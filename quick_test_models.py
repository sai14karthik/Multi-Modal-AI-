#!/usr/bin/env python3
"""
Quick test to verify models can be loaded (without running full inference).
This is faster and helps identify model loading issues early.
"""

import sys
import os
import traceback

# Set up environment before imports
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.tf_mock import ensure_tensorflow_stub
ensure_tensorflow_stub()

import torch
from PIL import Image
import numpy as np

# Test configurations
TEST_MODELS = [
    # CLIP models
    {
        'name': 'openai/clip-vit-large-patch14',
        'arch': 'clip',
        'description': 'OpenAI CLIP ViT-Large'
    },
    {
        'name': 'openai/clip-vit-base-patch32',
        'arch': 'clip',
        'description': 'OpenAI CLIP ViT-Base'
    },
    {
        'name': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'arch': 'clip',
        'description': 'LAION CLIP ViT-Huge'
    },
    {
        'name': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'arch': 'clip',
        'description': 'Microsoft BiomedCLIP'
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
        'description': 'LLaVA-Med v1.5 Mistral 7B'
    },
]

def create_dummy_image():
    """Create a dummy RGB image for testing."""
    # Create a simple test image (224x224 RGB)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def test_model_loading(model_config):
    """Test if a model can be loaded."""
    model_name = model_config['name']
    model_arch = model_config['arch']
    description = model_config['description']
    
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Model: {model_name}")
    print(f"Architecture: {model_arch}")
    print(f"{'='*80}")
    
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        # Load model based on architecture
        if model_arch == 'llava':
            from src.models.llava_runner import LLaVARunner
            model = LLaVARunner(
                model_name=model_name,
                device=device,
                class_names=['high_grade', 'low_grade']
            )
        elif model_arch == 'llava_med':
            from src.models.llava_med_runner import LLaVAMedRunner
            model = LLaVAMedRunner(
                model_name=model_name,
                device=device,
                class_names=['high_grade', 'low_grade']
            )
        else:  # clip
            from src.models.model_wrapper import MultimodalModelWrapper
            model = MultimodalModelWrapper(
                model_name=model_name,
                device=device,
                class_names=['high_grade', 'low_grade']
            )
        
        print("✅ Model loaded successfully!")
        
        # Test inference with dummy image
        print("Testing inference with dummy image...")
        dummy_img = create_dummy_image()
        
        result = model.predict(
            images={'CT': dummy_img},
            available_modalities=['CT'],
            batch_size=1,
            preprocess=False,
            temperature=1.0,
            use_weighted_ensemble=True,
            try_both_swaps=True
        )
        
        # Validate result
        if 'prediction' in result and 'confidence' in result:
            print(f" Inference successful!")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            if 'probabilities' in result:
                print(f"   Probabilities: {result['probabilities']}")
            return True, None
        else:
            return False, "Result missing required fields"
            
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f" Error: {error_msg}")
        traceback.print_exc()
        return False, error_msg

def main():
    """Run quick tests for all models."""
    print("="*80)
    print("Quick Model Loading Test")
    print("="*80)
    print("\nThis test verifies that models can be loaded and perform basic inference.")
    print("Note: This may take several minutes as models are downloaded if not cached.\n")
    
    results = {
        'passed': [],
        'failed': []
    }
    
    for model_config in TEST_MODELS:
        try:
            success, error = test_model_loading(model_config)
            if success:
                results['passed'].append({
                    'model': model_config['name'],
                    'description': model_config['description'],
                    'arch': model_config['arch']
                })
            else:
                results['failed'].append({
                    'model': model_config['name'],
                    'description': model_config['description'],
                    'arch': model_config['arch'],
                    'error': error
                })
        except KeyboardInterrupt:
            print("\n\n Tests interrupted by user")
            break
        except Exception as e:
            results['failed'].append({
                'model': model_config['name'],
                'description': model_config['description'],
                'arch': model_config['arch'],
                'error': f"Unexpected error: {str(e)}"
            })
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n Passed: {len(results['passed'])}/{len(TEST_MODELS)}")
    for result in results['passed']:
        print(f"   ✓ {result['description']}")
    
    print(f"\n Failed: {len(results['failed'])}/{len(TEST_MODELS)}")
    for result in results['failed']:
        print(f"  {result['description']}")
        print(f"     Error: {result['error']}")
    
    return 0 if len(results['failed']) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

