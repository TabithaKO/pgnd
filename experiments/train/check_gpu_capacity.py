"""
check_gpu_capacity.py — GPU Usage Monitor and Parallel Training Capacity Assessment
=====================================================================================

Monitors GPU utilization, memory, and processes to assess whether multiple training
jobs can run in parallel.

Usage:
    python check_gpu_capacity.py
    python check_gpu_capacity.py --watch  # Continuous monitoring
"""

import argparse
import subprocess
import time
import json
from pathlib import Path


def get_gpu_info():
    """Get detailed GPU information using nvidia-smi."""
    try:
        # Query GPU metrics
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': float(parts[2]),
                    'memory_used_mb': float(parts[3]),
                    'memory_free_mb': float(parts[4]),
                    'gpu_utilization': float(parts[5]),
                    'memory_utilization': float(parts[6]),
                    'temperature': float(parts[7]) if parts[7] != 'N/A' else 0,
                    'power_draw': float(parts[8]) if parts[8] != 'N/A' else 0,
                    'power_limit': float(parts[9]) if parts[9] != 'N/A' else 0,
                })

        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def get_gpu_processes():
    """Get processes using GPUs."""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': float(parts[2]),
                    })

        return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []


def get_process_details(pid):
    """Get detailed information about a process."""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'cmd='],
                              capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Unknown"


def estimate_training_memory_requirement():
    """Estimate memory requirements for PGND training jobs."""
    # Based on typical PGND training runs
    base_pgnd = 8000  # MB - base PGND model + data

    estimates = {
        'PGND Baseline (no render loss)': base_pgnd,
        'Ablation 1 (LBS + frozen GS)': base_pgnd + 4000,  # +4GB for GS + LBS
        'Ablation 2 (Mesh-GS + trainable)': base_pgnd + 6000,  # +6GB for mesh + trainable GS
    }

    return estimates


def assess_parallel_capacity(gpus, processes):
    """Assess whether we can run multiple training jobs in parallel."""
    print("\n" + "="*80)
    print("PARALLEL TRAINING CAPACITY ASSESSMENT")
    print("="*80)

    # Get memory estimates
    mem_estimates = estimate_training_memory_requirement()

    print("\nEstimated Memory Requirements:")
    for job_type, mem_mb in mem_estimates.items():
        print(f"  {job_type}: {mem_mb/1024:.1f} GB")

    print("\nCurrent GPU Status:")

    recommendations = []

    for gpu in gpus:
        idx = gpu['index']
        name = gpu['name']
        mem_total_gb = gpu['memory_total_mb'] / 1024
        mem_used_gb = gpu['memory_used_mb'] / 1024
        mem_free_gb = gpu['memory_free_mb'] / 1024
        gpu_util = gpu['gpu_utilization']

        print(f"\n  GPU {idx} ({name}):")
        print(f"    Memory: {mem_used_gb:.1f}/{mem_total_gb:.1f} GB used ({mem_free_gb:.1f} GB free)")
        print(f"    Utilization: {gpu_util:.0f}%")
        print(f"    Temperature: {gpu['temperature']:.0f}°C")
        print(f"    Power: {gpu['power_draw']:.0f}W / {gpu['power_limit']:.0f}W")

        # Find processes on this GPU
        gpu_processes = [p for p in processes if any(
            str(idx) in get_process_details(p['pid']) for p in [p]
        )]

        if gpu_processes:
            print(f"    Active processes:")
            for proc in gpu_processes:
                cmd = get_process_details(proc['pid'])
                # Truncate command
                cmd_short = cmd[:80] + '...' if len(cmd) > 80 else cmd
                print(f"      PID {proc['pid']}: {cmd_short}")
                print(f"        GPU Memory: {proc['memory_mb']/1024:.1f} GB")

        # Assess capacity
        ablation1_mem = mem_estimates['Ablation 1 (LBS + frozen GS)'] / 1024
        ablation2_mem = mem_estimates['Ablation 2 (Mesh-GS + trainable)'] / 1024

        can_fit_ablation1 = mem_free_gb >= ablation1_mem
        can_fit_ablation2 = mem_free_gb >= ablation2_mem
        can_fit_both = mem_free_gb >= (ablation1_mem + ablation2_mem)

        if idx == 0:  # GPU currently running Ablation 1
            if can_fit_ablation2:
                recommendations.append({
                    'gpu': idx,
                    'action': 'add_ablation2',
                    'reason': f'GPU {idx} has {mem_free_gb:.1f} GB free, enough for Ablation 2 ({ablation2_mem:.1f} GB)'
                })
            else:
                recommendations.append({
                    'gpu': idx,
                    'action': 'wait',
                    'reason': f'GPU {idx} only has {mem_free_gb:.1f} GB free, need {ablation2_mem:.1f} GB for Ablation 2'
                })
        else:  # Other GPUs
            if can_fit_ablation2:
                recommendations.append({
                    'gpu': idx,
                    'action': 'use_gpu',
                    'reason': f'GPU {idx} has {mem_free_gb:.1f} GB free, can run Ablation 2'
                })

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    for rec in recommendations:
        if rec['action'] == 'add_ablation2':
            print(f"\n✅ GPU {rec['gpu']}: CAN RUN ABLATION 2 IN PARALLEL")
            print(f"   {rec['reason']}")
            print(f"   Command: python train_eval_ablation2.py gpus=[{rec['gpu']}] ...")
        elif rec['action'] == 'use_gpu':
            print(f"\n✅ GPU {rec['gpu']}: AVAILABLE FOR ABLATION 2")
            print(f"   {rec['reason']}")
            print(f"   Command: python train_eval_ablation2.py gpus=[{rec['gpu']}] ...")
        elif rec['action'] == 'wait':
            print(f"\n⚠️  GPU {rec['gpu']}: NOT ENOUGH MEMORY")
            print(f"   {rec['reason']}")
            print(f"   Options:")
            print(f"     1. Wait for Ablation 1 to finish")
            print(f"     2. Use a different GPU")
            print(f"     3. Reduce batch size or model size")

    # Overall recommendation
    print("\n" + "="*80)
    print("OVERALL RECOMMENDATION:")
    print("="*80)

    num_available = sum(1 for r in recommendations if r['action'] in ['add_ablation2', 'use_gpu'])

    if num_available > 0:
        print(f"\n✅ YES - You can run Ablation 2 in parallel!")
        print(f"   {num_available} GPU(s) have sufficient capacity")
        print(f"\nSuggested approach:")
        print(f"  1. Start Ablation 2 on an available GPU")
        print(f"  2. Monitor with: watch -n 5 nvidia-smi")
        print(f"  3. Both jobs will complete in ~30 hours (parallel)")
    else:
        print(f"\n⚠️  WAIT - Not enough GPU capacity for parallel training")
        print(f"   All GPUs are at capacity")
        print(f"\nSuggested approach:")
        print(f"  1. Let Ablation 1 complete first (~30 hours)")
        print(f"  2. Then start Ablation 2")
        print(f"  3. Total time: ~60 hours (sequential)")


def main():
    parser = argparse.ArgumentParser(description='Check GPU capacity for parallel training')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring (update every 5s)')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds (default: 5)')

    args = parser.parse_args()

    try:
        while True:
            gpus = get_gpu_info()
            processes = get_gpu_processes()

            if not gpus:
                print("No GPUs found or nvidia-smi not available")
                return 1

            # Clear screen for watch mode
            if args.watch:
                subprocess.run(['clear'])

            assess_parallel_capacity(gpus, processes)

            if not args.watch:
                break

            print(f"\nUpdating in {args.interval} seconds... (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        return 0

    return 0


if __name__ == '__main__':
    exit(main())
