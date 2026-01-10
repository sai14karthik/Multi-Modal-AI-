#!/usr/bin/env python3
"""
Local test script for Multi-Modal AI evaluation.
Tests the pipeline with a small sample of data using one model.

Usage:
    python test_local.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import main function
from src.main import main

def test_local():
    """Run a local test with minimal data and one model."""
    
    # Default arguments for local testing
    test_args = [
        '--data_root', 'data',  # Config expects root to be 'data', not 'data/Lung-PET-CT-Dx'
        '--dataset_config', 'data/dataset_config.yaml',  # Use the dataset config
        '--modalities', 'CT', 'PET',
        '--model_name', 'openai/clip-vit-base-patch32',  # Small, fast model
        '--model_arch', 'clip',
        '--output_dir', 'results',
        '--max_samples', '5',  # Only 5 images per patient per modality for testing
        '--batch_size', '1',
        '--class_names', 'high_grade', 'low_grade',
        '--temperature', '1.0',
        '--no_preprocess',  # Skip preprocessing for faster testing
    ]
    
    # Override sys.argv for argparse
    original_argv = sys.argv
    sys.argv = ['test_local.py'] + test_args
    
    try:
        print("="*80)
        print("LOCAL TEST - Multi-Modal AI Evaluation")
        print("="*80)
        print()
        print("Configuration:")
        print(f"  Model: openai/clip-vit-base-patch32")
        print(f"  Modalities: CT, PET")
        print(f"  Max samples per patient: 5")
        print(f"  Data root: data")
        print(f"  Dataset config: data/dataset_config.yaml")
        print()
        print("Starting test...")
        print()
        
        # Run main function
        main()
        
        print()
        print("="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("Check the 'results' directory for output files.")
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == '__main__':
    test_local()
