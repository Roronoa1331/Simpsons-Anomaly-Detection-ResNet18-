"""
Quick script to check the Simpsons dataset structure and statistics
"""

import os
from collections import defaultdict

def check_dataset(train_dir='dataset/simpsons_dataset',
                  test_normal_dir='dataset/test/normal',
                  test_anomaly_dir='dataset/test/anomaly'):
    """
    Check dataset structure and print statistics

    Args:
        train_dir: Path to the training dataset directory
        test_normal_dir: Path to normal test images
        test_anomaly_dir: Path to anomaly test images
    """
    print("\n" + "="*70)
    print("Simpsons Dataset Statistics")
    print("="*70)

    # Check training directory
    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory '{train_dir}' not found!")
        print("Please make sure the dataset is in the correct location.")
        return

    # Get all character directories
    character_dirs = [d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]

    if not character_dirs:
        print(f"ERROR: No character directories found in '{train_dir}'")
        return

    # Count images per character in training set
    character_counts = {}
    total_train_images = 0

    for char_dir in sorted(character_dirs):
        char_path = os.path.join(train_dir, char_dir)
        images = [f for f in os.listdir(char_path)
                 if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        character_counts[char_dir] = len(images)
        total_train_images += len(images)

    # Count test images
    test_normal_count = 0
    test_anomaly_count = 0

    if os.path.exists(test_normal_dir):
        test_normal_images = [f for f in os.listdir(test_normal_dir)
                             if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        test_normal_count = len(test_normal_images)

    if os.path.exists(test_anomaly_dir):
        test_anomaly_images = [f for f in os.listdir(test_anomaly_dir)
                              if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        test_anomaly_count = len(test_anomaly_images)

    # Print statistics
    print(f"\n{'TRAINING DATA':-^70}")
    print(f"Total Characters: {len(character_dirs)}")
    print(f"Total Training Images: {total_train_images}")
    print(f"Average Images per Character: {total_train_images / len(character_dirs):.1f}")

    print(f"\n{'TEST DATA':-^70}")
    print(f"Normal Test Images: {test_normal_count}")
    print(f"Anomaly Test Images: {test_anomaly_count}")
    print(f"Total Test Images: {test_normal_count + test_anomaly_count}")

    print(f"\n{'-'*70}")
    print("Training Images per Character:")
    print("-"*70)
    print(f"{'Character Name':<40} {'Image Count':>15}")
    print("-"*70)

    for char_name, count in sorted(character_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{char_name:<40} {count:>15}")

    print("-"*70)

    # Check for potential issues
    print("\n" + "="*70)
    print("Data Quality Checks")
    print("="*70)

    small_classes = [char for char, count in character_counts.items() if count < 20]
    if small_classes:
        print(f"\nWARNING: {len(small_classes)} characters have fewer than 20 images:")
        for char in small_classes[:5]:
            print(f"  - {char} ({character_counts[char]} images)")
        if len(small_classes) > 5:
            print(f"  ... and {len(small_classes) - 5} more")
    else:
        print("\n✓ All characters have sufficient images (>= 20)")

    large_classes = [char for char, count in character_counts.items() if count > 500]
    if large_classes:
        print(f"\n✓ {len(large_classes)} characters have >500 images (good for training):")
        for char in large_classes:
            print(f"  - {char} ({character_counts[char]} images)")

    # Test set checks
    if test_normal_count > 0 and test_anomaly_count > 0:
        print(f"\n✓ Test set is ready:")
        print(f"  - Normal samples: {test_normal_count}")
        print(f"  - Anomaly samples: {test_anomaly_count}")
        print(f"  - Anomaly ratio: {test_anomaly_count / (test_normal_count + test_anomaly_count) * 100:.1f}%")
    else:
        print(f"\nWARNING: Test set incomplete")
        if test_normal_count == 0:
            print(f"  - No normal test images found in '{test_normal_dir}'")
        if test_anomaly_count == 0:
            print(f"  - No anomaly test images found in '{test_anomaly_dir}'")

    print("\n" + "="*70)
    print("Dataset Structure:")
    print("="*70)
    print(f"Training data: {train_dir}/")
    print(f"  └── [character_name]/")
    print(f"      └── *.jpg")
    print(f"\nTest data:")
    print(f"  Normal: {test_normal_dir}/")
    print(f"  Anomaly: {test_anomaly_dir}/")
    print("="*70)

    print("\n" + "="*70)
    print("Dataset check complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    check_dataset()

