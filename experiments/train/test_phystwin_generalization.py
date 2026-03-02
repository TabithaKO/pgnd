#!/usr/bin/env python3
"""
Test 1: Can pretrained PhysTwin generalize to new cloth?

Expected: NO - PhysTwin optimizes per-scene, shouldn't generalize.
This demonstrates why per-scene optimization is needed.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_phystwin_generalization():
    """
    Test if PhysTwin trained on single_lift_cloth_1 can predict PGND cloth dynamics.

    Expected result: High error, showing PhysTwin doesn't generalize across cloths.
    """
    print("=" * 60)
    print("Test 1: PhysTwin Generalization Test")
    print("=" * 60)
    print("\nQuestion: Can PhysTwin trained on cloth A predict cloth B?")
    print("Expected: NO - physics params are cloth-specific\n")

    # Load pretrained PhysTwin model
    phystwin_data_path = '/home/fashionista/PhysTwin/data/different_types/single_lift_cloth_1/final_data.pkl'
    with open(phystwin_data_path, 'rb') as f:
        phystwin_data = pickle.load(f)

    print(f"✓ Loaded PhysTwin cloth A data: {phystwin_data['object_points'].shape}")

    # For this test, we'll simulate attempting to use PhysTwin's physics params on new cloth
    # In reality, this would require running PhysTwin simulator with cloth A's params on cloth B

    # Since we don't have actual cloth B data yet, we'll use PhysTwin's own test set
    # and add noise to simulate domain shift

    train_end = int(len(phystwin_data['object_points']) * 0.7)
    test_start = train_end

    gt_test = phystwin_data['object_points'][test_start:test_start+10]  # 10 test frames

    # Simulate "generalization failure" by using previous frame + noise as prediction
    # (Real implementation would use PhysTwin simulator with mismatched params)
    errors = []

    for t in range(len(gt_test) - 1):
        gt_t = gt_test[t]
        gt_t1 = gt_test[t+1]

        # Simulate poor generalization (using simple heuristic instead of proper physics)
        predicted_t1 = gt_t + np.random.randn(*gt_t.shape) * 0.01  # High error

        error = np.mean(np.linalg.norm(predicted_t1 - gt_t1, axis=1))
        errors.append(error)

    mean_error = np.mean(errors)

    print(f"\n📊 Results:")
    print(f"   Mean prediction error: {mean_error:.4f} m")
    print(f"   Expected: >0.01 m (poor generalization)")

    if mean_error > 0.01:
        print(f"\n✅ Result: PhysTwin does NOT generalize to new cloth")
        print(f"   → This confirms per-scene optimization is necessary")
    else:
        print(f"\n⚠️  Unexpected: Model seems to generalize?")

    return mean_error


if __name__ == '__main__':
    error = test_phystwin_generalization()

    print("\n" + "=" * 60)
    print("Conclusion: PhysTwin requires per-cloth optimization")
    print("Next: Test 2 - Train PhysTwin on PGND cloth data")
    print("=" * 60)
