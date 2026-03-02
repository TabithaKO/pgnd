"""
find_matching_episodes.py — Find Matching Episode + Splat Pairs
================================================================

Maps preprocessed training episodes (cloth_merged/sub_episodes_v) to their
source raw episodes with .splat files for Gaussian Splatting.

Usage:
    python find_matching_episodes.py --output matches.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def find_all_episodes_with_splats(data_cloth_root: Path) -> Dict[str, List[str]]:
    """Find all episodes with .splat files in each recording.

    Returns:
        Dict mapping recording_name -> list of episode_names with .splat files
    """
    recordings_with_splats = {}

    for recording_dir in data_cloth_root.iterdir():
        if not recording_dir.is_dir() or recording_dir.name == 'cloth_merged':
            continue

        episodes_with_splats = []
        for episode_dir in sorted(recording_dir.glob('episode_*')):
            gs_dir = episode_dir / 'gs'
            if gs_dir.exists() and list(gs_dir.glob('*.splat')):
                episodes_with_splats.append(episode_dir.name)

        if episodes_with_splats:
            recordings_with_splats[recording_dir.name] = episodes_with_splats

    return recordings_with_splats


def parse_meta_txt(meta_path: Path) -> Optional[Dict]:
    """Parse meta.txt file from preprocessed episode.

    Format:
        Line 0: source_episode_id
        Line 1: source_frame_start
        Line 2: num_frames (optional)
    """
    if not meta_path.exists():
        return None

    try:
        meta = np.loadtxt(str(meta_path))
        return {
            'source_episode_id': int(meta[0]),
            'source_frame_start': int(meta[1]),
            'num_frames': int(meta[2]) if len(meta) > 2 else None
        }
    except Exception as e:
        print(f"Warning: Could not parse {meta_path}: {e}")
        return None


def find_source_recording(episode_idx: int, metadata_json: Optional[Path]) -> Optional[str]:
    """Find which recording a preprocessed episode came from.

    Args:
        episode_idx: Index of episode in cloth_merged/sub_episodes_v/
        metadata_json: Path to metadata.json (if exists)

    Returns:
        Recording name (e.g., '1224_cloth_fold_processed') or None
    """
    if metadata_json and metadata_json.exists():
        try:
            with open(metadata_json) as f:
                metadata = json.load(f)

            # metadata can be list or dict
            if isinstance(metadata, list):
                if episode_idx < len(metadata):
                    entry = metadata[episode_idx]
                else:
                    return None
            else:
                entry = metadata.get(str(episode_idx), metadata.get(f'episode_{episode_idx:04d}'))

            if entry is None:
                return None

            # Extract recording name from path
            if isinstance(entry, dict):
                path = Path(entry.get('path', ''))
            else:
                path = Path(str(entry))

            # path example: 'experiments/log/data/1224_cloth_fold_processed/sub_episodes_v'
            # We want: '1224_cloth_fold_processed'
            parts = path.parts
            for i, part in enumerate(parts):
                if part == 'data' or part == 'data_cloth':
                    if i + 1 < len(parts):
                        return parts[i + 1]

            return None
        except Exception as e:
            print(f"Warning: Could not parse metadata.json: {e}")
            return None

    return None


def find_matching_pairs(
    cloth_merged_dir: Path,
    data_cloth_root: Path,
) -> List[Dict]:
    """Find all matching pairs of preprocessed episodes and raw episodes with .splats.

    Returns:
        List of dicts with:
            - episode_idx: Index in cloth_merged/sub_episodes_v
            - episode_name: Name (e.g., 'episode_0000')
            - traj_path: Path to traj.npz
            - source_recording: Recording name
            - source_episode_id: Episode ID in source recording
            - source_episode_name: Episode name in source recording
            - source_frame_start: Starting frame
            - has_splat: Whether source episode has .splat files
            - splat_dir: Path to .splat directory (if exists)
            - num_splats: Number of .splat files (if exists)
    """
    matches = []

    # Find all recordings with .splat files
    recordings_with_splats = find_all_episodes_with_splats(data_cloth_root)

    print(f"Recordings with .splat files:")
    for rec_name, episodes in recordings_with_splats.items():
        print(f"  {rec_name}: {episodes}")

    # Check metadata.json
    metadata_json = cloth_merged_dir / 'metadata.json'

    # Iterate through all preprocessed episodes
    sub_episodes_dir = cloth_merged_dir / 'sub_episodes_v'
    if not sub_episodes_dir.exists():
        print(f"Error: {sub_episodes_dir} not found")
        return matches

    for episode_dir in sorted(sub_episodes_dir.glob('episode_*')):
        episode_idx = int(episode_dir.name.split('_')[1])
        traj_path = episode_dir / 'traj.npz'

        if not traj_path.exists():
            continue

        # Parse meta.txt
        meta = parse_meta_txt(episode_dir / 'meta.txt')
        if meta is None:
            continue

        # Try to find source recording
        source_recording = find_source_recording(episode_idx, metadata_json)

        # If we couldn't find source_recording from metadata, try to infer it
        # by checking which recording has an episode with .splat files at the source_episode_id
        if source_recording is None:
            source_ep_name = f"episode_{meta['source_episode_id']:04d}"
            for rec_name, episodes in recordings_with_splats.items():
                if source_ep_name in episodes:
                    source_recording = rec_name
                    break

        # Build match info
        match = {
            'episode_idx': episode_idx,
            'episode_name': episode_dir.name,
            'traj_path': str(traj_path),
            'source_recording': source_recording,
            'source_episode_id': meta['source_episode_id'],
            'source_episode_name': f"episode_{meta['source_episode_id']:04d}",
            'source_frame_start': meta['source_frame_start'],
            'num_frames': meta.get('num_frames'),
            'has_splat': False,
            'splat_dir': None,
            'num_splats': 0,
        }

        # Check if source episode has .splat files
        if source_recording:
            source_ep_dir = data_cloth_root / source_recording / match['source_episode_name']
            splat_dir = source_ep_dir / 'gs'

            if splat_dir.exists():
                splat_files = list(splat_dir.glob('*.splat'))
                if splat_files:
                    match['has_splat'] = True
                    match['splat_dir'] = str(splat_dir)
                    match['num_splats'] = len(splat_files)

        matches.append(match)

    return matches


def main():
    parser = argparse.ArgumentParser(description='Find matching episode + splat pairs')
    parser.add_argument('--cloth_merged', type=str,
                       default='experiments/log/data_cloth/cloth_merged',
                       help='Path to cloth_merged directory')
    parser.add_argument('--data_cloth', type=str,
                       default='experiments/log/data_cloth',
                       help='Path to data_cloth root directory')
    parser.add_argument('--output', type=str, default='episode_matches.json',
                       help='Output JSON file with matches')
    parser.add_argument('--show_all', action='store_true',
                       help='Show all episodes (not just those with .splat files)')

    args = parser.parse_args()

    cloth_merged = Path(args.cloth_merged)
    data_cloth = Path(args.data_cloth)

    if not cloth_merged.exists():
        print(f"Error: {cloth_merged} not found")
        return 1

    if not data_cloth.exists():
        print(f"Error: {data_cloth} not found")
        return 1

    print("Searching for matching episodes...")
    matches = find_matching_pairs(cloth_merged, data_cloth)

    # Filter to only those with .splat files (unless --show_all)
    if not args.show_all:
        matches_with_splat = [m for m in matches if m['has_splat']]
        print(f"\nFound {len(matches_with_splat)} episodes with matching .splat files (out of {len(matches)} total):")
    else:
        matches_with_splat = matches
        print(f"\nFound {len(matches)} episodes:")

    # Print summary
    for match in matches_with_splat[:10]:  # Show first 10
        print(f"\n  {match['episode_name']}:")
        print(f"    Source: {match['source_recording']}/{match['source_episode_name']}")
        print(f"    Frame start: {match['source_frame_start']}")
        if match['has_splat']:
            print(f"    ✅ Has {match['num_splats']} .splat files")
        else:
            print(f"    ❌ No .splat files")

    if len(matches_with_splat) > 10:
        print(f"\n  ... and {len(matches_with_splat) - 10} more")

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(matches_with_splat, f, indent=2)

    print(f"\n✅ Saved matches to {output_path}")

    if matches_with_splat:
        print(f"\nExample usage:")
        match = matches_with_splat[0]
        print(f"  python test_mesh_gaussian_model.py \\")
        print(f"    --episode_dir {Path(match['traj_path']).parent} \\")

        if match['has_splat']:
            # Find a splat file
            splat_dir = Path(match['splat_dir'])
            splat_files = sorted(splat_dir.glob('*.splat'))
            if splat_files:
                # Choose splat near the source_frame_start
                closest_splat = min(
                    splat_files,
                    key=lambda p: abs(int(p.stem) - match['source_frame_start'])
                )
                print(f"    --splat_path {closest_splat}")

    return 0


if __name__ == '__main__':
    exit(main())
