#!/usr/bin/env python3
"""
Test script to verify all supported models work correctly.
Tests model loading and basic inference with a small sample.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test configurations
TEST_MODELS = [
    # CLIP models
    {
        'name': 'openai/clip-vit-large-patch14',
        'arch': 'clip',
        'description': 'OpenAI CLIP ViT-Large (default)'
    },
    {
        'name': 'openai/clip-vit-base-patch32',
        'arch': 'clip',
        'description': 'OpenAI CLIP ViT-Base (fallback)'
    },
    {
        'name': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'arch': 'clip',
        'description': 'LAION CLIP ViT-Huge (best performance)'
    },
    {
        'name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'arch': 'clip',
        'description': 'Microsoft BiomedCLIP (medical-specific)'
    },
    # LLaVA models
    {
        'name': 'liuhaotian/llava-v1.6-mistral-7b',
        'arch': 'llava',
        'description': 'LLaVA v1.6 Mistral 7B'
    },
    # LLaVA-Med models
    {
        'name': 'microsoft/llava-med-v1.5-mistral-7b',
        'arch': 'llava_med',
        'description': 'LLaVA-Med v1.5 Mistral 7B (medical)'
    },
]

def check_data_available():
    """Check if test data is available."""
    data_root = 'data'
    config_path = 'data/dataset_config.yaml'
    
    if not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        return False
    
    if not os.path.exists(config_path):
        print(f"⚠️  Dataset config not found: {config_path}")
        print("   Will use default configuration")
    
    return True

def test_model(model_config, max_samples=10, timeout=300):
    """
    Test a single model.
    
    Args:
        model_config: Dictionary with 'name', 'arch', 'description'
        max_samples: Number of samples to test with
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (success: bool, error_message: str, output: str)
    """
    model_name = model_config['name']
    model_arch = model_config['arch']
    description = model_config['description']
    
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Model: {model_name}")
    print(f"Architecture: {model_arch}")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        sys.executable, '-m', 'src.main',
        '--data_root', 'data',
        '--modalities', 'CT', 'PET',
        '--model_name', model_name,
        '--model_arch', model_arch,
        '--output_dir', 'test_results',
        '--batch_size', '1',
        '--max_samples', str(max_samples),
        '--dataset_config', 'data/dataset_config.yaml',
        '--class_names', 'high_grade', 'low_grade'
    ]
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            # Check if results file was created
            model_safe_name = model_name.replace('/', '_')
            results_file = f'test_results/results_{model_safe_name}.json'
            
            if os.path.exists(results_file):
                # Read and validate results
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Check if results are valid
                    if 'step_accuracies' in results:
                        print(f"✅ Model loaded and inference completed successfully!")
                        print(f"   Results saved to: {results_file}")
                        
                        # Print accuracy summary
                        for step, metrics in results['step_accuracies'].items():
                            acc = metrics.get('accuracy', 0.0)
                            num_samples = metrics.get('num_samples', 0)
                            print(f"   {step}: Accuracy={acc:.4f}, Samples={num_samples}")
                        
                        return True, None, result.stdout
                    else:
                        return False, "Results file missing 'step_accuracies'", result.stdout
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON in results file: {e}", result.stdout
            else:
                return False, f"Results file not created: {results_file}", result.stdout
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, f"Command failed with return code {result.returncode}", error_msg
            
    except subprocess.TimeoutExpired:
        return False, f"Test timed out after {timeout} seconds", ""
    except Exception as e:
        return False, f"Exception during test: {str(e)}", ""

def main():
    """Run tests for all models."""
    print("="*80)
    print("Model Testing Suite")
    print("="*80)
    
    # Check if data is available
    if not check_data_available():
        print("\n❌ Cannot run tests without data. Please prepare the dataset first.")
        return 1
    
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Track results
    results = {
        'passed': [],
        'failed': [],
        'skipped': []
    }
    
    # Test each model
    for model_config in TEST_MODELS:
        model_name = model_config['name']
        description = model_config['description']
        
        try:
            success, error_msg, output = test_model(model_config, max_samples=10)
            
            if success:
                results['passed'].append({
                    'model': model_name,
                    'description': description,
                    'arch': model_config['arch']
                })
            else:
                results['failed'].append({
                    'model': model_name,
                    'description': description,
                    'arch': model_config['arch'],
                    'error': error_msg,
                    'output': output[:500] if output else ''  # Truncate long output
                })
                print(f"\n❌ Test failed: {error_msg}")
                if output:
                    print(f"   Output (last 500 chars):\n{output[-500:]}")
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            results['skipped'].extend([
                {
                    'model': m['name'],
                    'description': m['description'],
                    'arch': m['arch']
                }
                for m in TEST_MODELS[TEST_MODELS.index(model_config):]
            ])
            break
        except Exception as e:
            results['failed'].append({
                'model': model_name,
                'description': description,
                'arch': model_config['arch'],
                'error': f"Unexpected error: {str(e)}",
                'output': ''
            })
            print(f"\n❌ Unexpected error testing {model_name}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✅ Passed: {len(results['passed'])}")
    for result in results['passed']:
        print(f"   - {result['description']} ({result['model']})")
    
    print(f"\n❌ Failed: {len(results['failed'])}")
    for result in results['failed']:
        print(f"   - {result['description']} ({result['model']})")
        print(f"     Error: {result['error']}")
    
    if results['skipped']:
        print(f"\n⏭️  Skipped: {len(results['skipped'])}")
        for result in results['skipped']:
            print(f"   - {result['description']} ({result['model']})")
    
    # Save detailed results
    results_file = 'test_results/test_summary.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return exit code
    return 0 if len(results['failed']) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

