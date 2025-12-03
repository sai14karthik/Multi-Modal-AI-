#!/usr/bin/env python3
"""
Performance testing script for comparing different models.
Tests multiple models and generates a performance comparison report.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Test configurations - focusing on CLIP models for speed
TEST_MODELS = [
    {
        'name': 'openai/clip-vit-large-patch14',
        'arch': 'clip',
        'description': 'OpenAI CLIP ViT-Large'
    },
    {
        'name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'arch': 'clip',
        'description': 'Microsoft BiomedCLIP (Medical)'
    },
    {
        'name': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'arch': 'clip',
        'description': 'LAION CLIP ViT-Huge'
    },
]

def run_performance_test(model_config, max_samples=50, split='val'):
    """Run performance test for a single model."""
    model_name = model_config['name']
    model_arch = model_config['arch']
    description = model_config['description']
    
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Model: {model_name}")
    print(f"Architecture: {model_arch}")
    print(f"Sample size: {max_samples}")
    print(f"{'='*80}\n")
    
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, '-m', 'src.main',
        '--data_root', 'data',
        '--modalities', 'CT', 'PET',
        '--model_name', model_name,
        '--model_arch', model_arch,
        '--output_dir', output_dir,
        '--batch_size', '1',
        '--max_samples', str(max_samples),
        '--dataset_config', 'data/dataset_config.yaml',
        '--class_names', 'high_grade', 'low_grade',
        '--split', split
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            # Read results file
            model_safe_name = model_name.replace('/', '_')
            results_file = f'{output_dir}/results_{model_safe_name}.json'
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return True, results, None
            else:
                return False, None, "Results file not created"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, None, f"Command failed: {error_msg[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, None, f"Test timed out after 10 minutes"
    except Exception as e:
        return False, None, f"Exception: {str(e)}"

def generate_report(all_results):
    """Generate a performance comparison report."""
    print("\n" + "="*80)
    print("PERFORMANCE TEST REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Header
    print(f"{'Model':<50} {'CT Acc':<12} {'PET Acc':<12} {'CT+PET Acc':<12} {'Samples':<10}")
    print("-" * 80)
    
    for model_info, results in all_results:
        if results is None:
            print(f"{model_info['description']:<50} {'FAILED':<12}")
            continue
        
        step_accuracies = results.get('step_accuracies', {})
        ct_acc = step_accuracies.get('CT', {}).get('accuracy', 0.0)
        pet_acc = step_accuracies.get('PET', {}).get('accuracy', 0.0)
        ct_pet_acc = step_accuracies.get('CT+PET', {}).get('accuracy', 0.0)
        num_samples = step_accuracies.get('CT', {}).get('num_samples', 0)
        
        print(f"{model_info['description']:<50} {ct_acc:<12.4f} {pet_acc:<12.4f} {ct_pet_acc:<12.4f} {num_samples:<10}")
    
    print("="*80)
    
    # Find best performing model
    best_ct = None
    best_pet = None
    best_combined = None
    best_ct_score = 0
    best_pet_score = 0
    best_combined_score = 0
    
    for model_info, results in all_results:
        if results is None:
            continue
        step_accuracies = results.get('step_accuracies', {})
        ct_acc = step_accuracies.get('CT', {}).get('accuracy', 0.0)
        pet_acc = step_accuracies.get('PET', {}).get('accuracy', 0.0)
        ct_pet_acc = step_accuracies.get('CT+PET', {}).get('accuracy', 0.0)
        
        if ct_acc > best_ct_score:
            best_ct_score = ct_acc
            best_ct = model_info['description']
        if pet_acc > best_pet_score:
            best_pet_score = pet_acc
            best_pet = model_info['description']
        if ct_pet_acc > best_combined_score:
            best_combined_score = ct_pet_acc
            best_combined = model_info['description']
    
    print("\nBest Performance:")
    if best_ct:
        print(f"  CT only: {best_ct} ({best_ct_score:.4f})")
    if best_pet:
        print(f"  PET only: {best_pet} ({best_pet_score:.4f})")
    if best_combined:
        print(f"  CT+PET: {best_combined} ({best_combined_score:.4f})")
    
    # Save report to file
    report_file = 'test_results/performance_report.json'
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'models': []
    }
    
    for model_info, results in all_results:
        model_data = {
            'name': model_info['name'],
            'description': model_info['description'],
            'arch': model_info['arch'],
            'success': results is not None,
            'results': results if results else None
        }
        report_data['models'].append(model_data)
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

def main():
    """Run performance tests for all models."""
    print("="*80)
    print("Model Performance Testing Suite")
    print("="*80)
    print("\nThis will test multiple models on the validation set.")
    print("Note: This may take several minutes depending on sample size.\n")
    
    # Check if data is available
    if not os.path.exists('data'):
        print("❌ Data directory not found. Please ensure data is available.")
        return 1
    
    if not os.path.exists('data/dataset_config.yaml'):
        print("⚠️  Dataset config not found. Will use defaults.")
    
    all_results = []
    
    # Test each model
    for model_config in TEST_MODELS:
        success, results, error = run_performance_test(
            model_config,
            max_samples=50,  # Test with 50 samples for better statistics
            split='val'
        )
        
        if success:
            all_results.append((model_config, results))
            print(f"✅ {model_config['description']} completed successfully")
        else:
            all_results.append((model_config, None))
            print(f"❌ {model_config['description']} failed: {error}")
    
    # Generate report
    generate_report(all_results)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

