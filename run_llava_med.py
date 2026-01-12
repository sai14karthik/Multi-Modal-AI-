#!/usr/bin/env python3
"""
Standalone script to run LLaVA-Med model evaluation.
Designed to be run on Newton cluster with proper error handling and logging.
"""

import sys
import os
import argparse
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run LLaVA-Med model evaluation')
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/llava-med-v1.5-mistral-7b',
        help='LLaVA-Med model name (default: microsoft/llava-med-v1.5-mistral-7b)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Data root directory (default: data)'
    )
    
    parser.add_argument(
        '--dataset_config',
        type=str,
        default='data/dataset_config.yaml',
        help='Dataset config file (default: data/dataset_config.yaml)'
    )
    
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['CT', 'PET'],
        help='Modalities to use (default: CT PET)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples per modality (default: None = all)'
    )
    
    # Model arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature for generation (default: 0.8)'
    )
    
    # Preprocessing arguments
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Disable image preprocessing'
    )
    
    parser.add_argument(
        '--aggressive-preprocess',
        action='store_true',
        help='Use aggressive preprocessing'
    )
    
    # Ensemble arguments
    parser.add_argument(
        '--no-weighted-ensemble',
        action='store_true',
        help='Disable weighted ensemble'
    )
    
    parser.add_argument(
        '--no-swap-test',
        action='store_true',
        help='Disable swap testing'
    )
    
    # Class names
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=['high_grade', 'low_grade'],
        help='Class names (default: high_grade low_grade)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    # Hugging Face token
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face token (or set HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    # Import main function
    try:
        from src.main import main as run_main
    except ImportError:
        print("ERROR: Could not import main from src.main", file=sys.stderr)
        traceback.print_exc()
        return 1
    
    # Convert argparse to sys.argv format for main()
    sys.argv = [
        'main.py',
        '--model_arch', 'llava_med',
        '--model_name', args.model_name,
        '--data_root', args.data_root,
        '--dataset_config', args.dataset_config,
        '--modalities'] + args.modalities + [
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--temperature', str(args.temperature),
        '--class_names'] + args.class_names + [
        '--output_dir', args.output_dir
    ]
    
    if args.max_samples:
        sys.argv.extend(['--max_samples', str(args.max_samples)])
    if args.no_preprocess:
        sys.argv.append('--no_preprocess')
    if args.aggressive_preprocess:
        sys.argv.append('--aggressive_preprocess')
    if args.no_weighted_ensemble:
        sys.argv.append('--no_weighted_ensemble')
    if args.no_swap_test:
        sys.argv.append('--no_swap_test')
    if args.hf_token:
        sys.argv.extend(['--hf_token', args.hf_token])
    
    # Run main
    try:
        run_main()
        return 0
    except Exception as e:
        print(f"ERROR: Failed to run LLaVA-Med evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
