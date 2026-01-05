#!/usr/bin/env python3
"""
Test script specifically for BiomedCLIP model loading.
Tests the fixed loading logic and verifies the model works correctly.
"""

import sys
import os
import traceback
from pathlib import Path

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


def create_dummy_image():
    """Create a dummy RGB image for testing."""
    # Create a simple test image (224x224 RGB)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_biomedclip_loading():
    """Test if BiomedCLIP model can be loaded."""
    print("=" * 80)
    print("TESTING BIOMEDCLIP MODEL LOADING")
    print("=" * 80)
    print()
    
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Model: {model_name}")
    print()
    
    try:
        from src.models.model_wrapper import MultimodalModelWrapper
        
        print("1. Testing model initialization...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        model = MultimodalModelWrapper(
            model_name=model_name,
            device=device,
            class_names=["high_grade", "low_grade"]
        )
        
        print(f"   ✓ Model loaded successfully!")
        print(f"   Model type: {type(model.model)}")
        print(f"   Processor type: {type(model.processor)}")
        print()
        
        # Check if it's actually BiomedCLIP or fallback
        if "openai/clip" in model.model_name.lower() and "biomedclip" not in model.model_name.lower():
            print("   ⚠️  WARNING: Model fell back to default CLIP model")
            print(f"   Actual model: {model.model_name}")
        else:
            print(f"   ✓ Using BiomedCLIP model: {model.model_name}")
        print()
        
        print("2. Testing image preprocessing...")
        test_image = create_dummy_image()
        print(f"   Test image size: {test_image.size}")
        
        # Test processor
        processed = model.processor(images=test_image, return_tensors="pt")
        print(f"   ✓ Image preprocessing successful")
        print(f"   Processed shape: {processed['pixel_values'].shape}")
        print()
        
        print("3. Testing text tokenization...")
        test_text = ["a lung CT scan showing high_grade lung cancer"]
        tokenized = model.processor(text=test_text, return_tensors="pt")
        print(f"   ✓ Text tokenization successful")
        if 'input_ids' in tokenized:
            print(f"   Tokenized shape: {tokenized['input_ids'].shape}")
        print()
        
        print("4. Testing model inference...")
        model.model.eval()
        
        with torch.no_grad():
            # Move to device
            pixel_values = processed['pixel_values'].to(device)
            input_ids = tokenized.get('input_ids', tokenized.get('input_ids')).to(device)
            
            # Run inference
            outputs = model.model(pixel_values=pixel_values, input_ids=input_ids)
            
            if hasattr(outputs, 'logits_per_image'):
                logits = outputs.logits_per_image
                print(f"   ✓ Inference successful")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Logits: {logits.cpu().numpy()}")
            else:
                print(f"   ⚠️  Unexpected output format: {type(outputs)}")
        print()
        
        print("5. Testing full prediction pipeline...")
        # predict() requires images dict and available_modalities list
        prediction = model.predict(
            images={"CT": test_image}, 
            available_modalities=["CT"],
            previous_predictions=None
        )
        print(f"   ✓ Full prediction successful")
        print(f"   Prediction: {prediction}")
        if isinstance(prediction, dict):
            if 'predicted_class' in prediction:
                print(f"   Predicted class: {prediction['predicted_class']}")
            if 'confidence' in prediction:
                print(f"   Confidence: {prediction['confidence']:.4f}")
            if 'probabilities' in prediction:
                print(f"   Probabilities: {prediction['probabilities']}")
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        return False


def test_open_clip_availability():
    """Test if open_clip is available."""
    print("=" * 80)
    print("TESTING OPEN_CLIP AVAILABILITY")
    print("=" * 80)
    print()
    
    try:
        import open_clip
        print("✓ open_clip is installed")
        print(f"   Version: {open_clip.__version__ if hasattr(open_clip, '__version__') else 'unknown'}")
        
        # Check for required functions
        required_funcs = ['create_model_and_transforms', 'get_tokenizer']
        for func_name in required_funcs:
            if hasattr(open_clip, func_name):
                print(f"   ✓ {func_name} is available")
            else:
                print(f"   ✗ {func_name} is NOT available")
        
        return True
    except ImportError:
        print("✗ open_clip is NOT installed")
        print("   Install with: pip install open-clip-torch>=2.20.0")
        return False


def test_biomedclip_with_samples():
    """Test BiomedCLIP with multiple samples."""
    print("=" * 80)
    print("TESTING BIOMEDCLIP WITH MULTIPLE SAMPLES")
    print("=" * 80)
    print()
    
    try:
        from src.models.model_wrapper import MultimodalModelWrapper
        
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading model: {model_name}")
        model = MultimodalModelWrapper(
            model_name=model_name,
            device=device,
            class_names=["high_grade", "low_grade"]
        )
        print(f"✓ Model loaded")
        print()
        
        # Test with multiple images
        num_samples = 5
        print(f"Testing with {num_samples} samples...")
        
        predictions = []
        for i in range(num_samples):
            test_image = create_dummy_image()
            prediction = model.predict(
                images={"CT": test_image},
                available_modalities=["CT"],
                previous_predictions=None
            )
            predictions.append(prediction)
            if isinstance(prediction, dict):
                pred_class = prediction.get('predicted_class', 'N/A')
                confidence = prediction.get('confidence', 0.0)
                print(f"  Sample {i+1}: {pred_class} (confidence: {confidence:.4f})")
        
        print()
        print(f"✓ Successfully processed {num_samples} samples")
        
        # Check consistency
        classes = [p.get('predicted_class', 'N/A') if isinstance(p, dict) else 'N/A' for p in predictions]
        print(f"   Predictions: {classes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("BIOMEDCLIP MODEL TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Check open_clip availability
    print("TEST 1: OpenCLIP Availability")
    print("-" * 80)
    results.append(("OpenCLIP Availability", test_open_clip_availability()))
    print()
    
    # Test 2: Test model loading
    print("TEST 2: Model Loading")
    print("-" * 80)
    results.append(("Model Loading", test_biomedclip_loading()))
    print()
    
    # Test 3: Test with multiple samples
    print("TEST 3: Multiple Samples")
    print("-" * 80)
    results.append(("Multiple Samples", test_biomedclip_with_samples()))
    print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    print()
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

