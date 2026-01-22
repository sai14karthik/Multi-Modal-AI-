#!/usr/bin/env python3
"""
Simple local test script for sample data.
Runs both forward and reverse orders with limited samples for quick testing.
"""

import subprocess
import sys
import os

# Default configuration
DEFAULT_MODEL = "openai/clip-vit-base-patch32"  # Fast, small model for testing
DEFAULT_ARCH = "clip"
DEFAULT_MAX_SAMPLES = 10  # Only 10 images per patient per modality
DEFAULT_DATA_ROOT = "data"
DEFAULT_CONFIG = "data/dataset_config.yaml"
DEFAULT_CLASS1 = "high_grade"
DEFAULT_CLASS2 = "low_grade"

def run_command(cmd, description):
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode

def main():
    # Parse arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model_arch = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ARCH
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_SAMPLES
    
    # Get from environment or use defaults
    data_root = os.environ.get('DATA_ROOT', DEFAULT_DATA_ROOT)
    dataset_config = os.environ.get('DATASET_CONFIG', DEFAULT_CONFIG)
    class1 = os.environ.get('CLASS1', DEFAULT_CLASS1)
    class2 = os.environ.get('CLASS2', DEFAULT_CLASS2)
    
    print("="*60)
    print("LOCAL TEST - Sample Data")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Architecture: {model_arch}")
    print(f"Data root: {data_root}")
    print(f"Dataset config: {dataset_config}")
    print(f"Classes: {class1}, {class2}")
    print(f"Max samples per patient per modality: {max_samples}")
    print(f"Modalities: CT PET")
    print()
    
    # Check if data directory exists
    if not os.path.isdir(data_root):
        print(f"❌ Error: Data root directory not found: {data_root}")
        print("   Please set DATA_ROOT environment variable or create the directory")
        return 1
    
    # Base command arguments
    base_args = [
        sys.executable, "-u", "-m", "src.main",
        "--data_root", data_root,
        "--model_arch", model_arch,
        "--model_name", model_name,
        "--output_dir", "results",
        "--batch_size", "4",  # Smaller batch for local testing
        "--dataset_config", dataset_config,
        "--class_names", class1, class2,
        "--temperature", "0.8",
        "--max_samples", str(max_samples)
    ]
    
    # Run forward order
    forward_cmd = base_args + ["--modalities", "CT", "PET"]
    forward_exit = run_command(forward_cmd, "Running FORWARD order: CT PET")
    
    # Run reverse order
    reverse_cmd = base_args + ["--modalities", "PET", "CT"]
    reverse_exit = run_command(reverse_cmd, "Running REVERSE order: PET CT")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Forward order (CT→PET): Exit code {forward_exit}")
    print(f"Reverse order (PET→CT): Exit code {reverse_exit}")
    
    if forward_exit == 0 and reverse_exit == 0:
        print("\n✅ Both orders completed successfully!")
        print(f"\nResults saved in: results/")
        print(f"Check results with: ls -lh results/*CT_PET*.json results/*PET_CT*.json")
        return 0
    else:
        print("\n❌ One or both orders failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
