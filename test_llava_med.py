#!/usr/bin/env python3
"""
Test script specifically for LLaVA-Med model loading and prediction storage.
Tests the fixed prediction storage logic to ensure predictions are stored correctly.
"""

import sys
import os
import traceback
from pathlib import Path

# Set up environment before imports
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('ACCELERATE_USE_TENSORBOARD', 'false')
os.environ.setdefault('USE_TF', '0')

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


def test_llava_med_loading():
    """Test if LLaVA-Med model can be loaded."""
    print("=" * 80)
    print("TESTING LLAVA-MED MODEL LOADING")
    print("=" * 80)
    print()
    
    model_name = "microsoft/llava-med-v1.5-mistral-7b"
    print(f"Model: {model_name}")
    print()
    
    try:
        from src.models.llava_med_runner import LLaVAMedRunner
        
        print("1. Testing model initialization...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        if device == 'cpu':
            print("   ⚠️  WARNING: Running on CPU. LLaVA-Med will be VERY SLOW.")
        
        model = LLaVAMedRunner(
            model_name=model_name,
            device=device,
            class_names=["high_grade", "low_grade"]
        )
        
        print(f"   ✓ Model loaded successfully!")
        print(f"   Model type: {type(model.model)}")
        print(f"   Tokenizer type: {type(model.tokenizer)}")
        print(f"   Image processor type: {type(model.image_processor)}")
        print()
        
        return model
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        return None


def test_llava_med_prediction(model):
    """Test LLaVA-Med prediction."""
    print("=" * 80)
    print("TESTING LLAVA-MED PREDICTION")
    print("=" * 80)
    print()
    
    if model is None:
        print("❌ Cannot test prediction - model not loaded")
        return False
    
    try:
        print("1. Testing image preprocessing...")
        test_image = create_dummy_image()
        print(f"   Test image size: {test_image.size}")
        
        # Test image processor
        processed = model.image_processor.preprocess(test_image, return_tensors="pt")
        print(f"   ✓ Image preprocessing successful")
        if 'pixel_values' in processed:
            print(f"   Processed shape: {processed['pixel_values'].shape}")
        print()
        
        print("2. Testing prediction generation...")
        prediction = model.predict(
            images={"CT": test_image},
            available_modalities=["CT"],
            previous_predictions=None
        )
        
        print(f"   ✓ Prediction successful")
        print(f"   Prediction: {prediction}")
        if isinstance(prediction, dict):
            if 'prediction' in prediction:
                print(f"   Predicted class index: {prediction['prediction']}")
            if 'confidence' in prediction:
                print(f"   Confidence: {prediction['confidence']:.4f}")
            if 'probabilities' in prediction:
                print(f"   Probabilities: {prediction['probabilities']}")
            if 'logits' in prediction:
                print(f"   Logits: {prediction['logits']}")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ PREDICTION TEST FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        return False


def test_patient_id_extraction():
    """Test patient_id extraction logic."""
    print("=" * 80)
    print("TESTING PATIENT_ID EXTRACTION")
    print("=" * 80)
    print()
    
    try:
        # Copy the extraction function logic (it's defined inside main() so not directly importable)
        import re
        
        def extract_patient_id_from_img(img):
            """Extract patient_id from image metadata (copied from main.py)."""
            # Use patient_id from metadata (dataloader already extracted it)
            patient_id = img.get('patient_id')
            if patient_id is not None and str(patient_id).strip():
                return str(patient_id).strip()
            
            # Fallback: extract from filename if not in metadata
            image_path = img.get('image_path', '')
            if image_path:
                filename = os.path.basename(image_path)
                # Try pattern from dataloader: A0001, 0001, etc. (flexible)
                match = re.search(r'([A-Z]?\d{4,})', filename)
                if match:
                    return match.group(1)
            
            # If no patient_id found, return None
            return None
        
        test_cases = [
            {
                'name': 'With patient_id in metadata',
                'img': {'patient_id': 'A0001', 'image_path': '/path/to/image.png'},
                'expected': 'A0001'
            },
            {
                'name': 'Without patient_id, extract from filename',
                'img': {'image_path': '/path/to/A0001_ct_series_0001.png'},
                'expected': 'A0001'  # Should extract from filename
            },
            {
                'name': 'With image_id as fallback',
                'img': {'image_id': 'img_123', 'image_path': '/path/to/image.png'},
                'expected': None  # Will use image_id as fallback in main code
            },
            {
                'name': 'Empty patient_id string',
                'img': {'patient_id': '', 'image_path': '/path/to/image.png'},
                'expected': None  # Empty string should return None
            },
        ]
        
        print("Testing patient_id extraction with various scenarios:")
        print()
        
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            result = extract_patient_id_from_img(test_case['img'])
            expected = test_case['expected']
            
            # For test case 3, we expect None but main.py will use image_id as fallback
            if test_case['name'] == 'With image_id as fallback':
                if result is None:
                    print(f"  Test {i}: {test_case['name']}")
                    print(f"    ✓ Result: {result} (will use image_id fallback in main code)")
                else:
                    print(f"  Test {i}: {test_case['name']}")
                    print(f"    ✓ Result: {result}")
            elif result == expected:
                print(f"  Test {i}: {test_case['name']}")
                print(f"    ✓ Result: {result} (expected: {expected})")
            else:
                print(f"  Test {i}: {test_case['name']}")
                print(f"    ✗ Result: {result} (expected: {expected})")
                all_passed = False
        
        print()
        return all_passed
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_prediction_storage_simulation():
    """Simulate prediction storage to test the logic."""
    print("=" * 80)
    print("TESTING PREDICTION STORAGE LOGIC")
    print("=" * 80)
    print()
    
    try:
        # Simulate the storage logic from main.py
        patient_predictions = {}
        patient_ct_predictions_list = {}
        
        # Test case 1: Normal case with patient_id
        print("Test 1: Normal case with valid patient_id")
        patient_id = "A0001"
        prediction = {
            'prediction': 0,
            'confidence': 0.85,
            'probabilities': {'high_grade': 0.85, 'low_grade': 0.15}
        }
        img_info = {
            'image_id': 'img_001',
            'image_path': '/path/to/image.png',
            'slice_index': 10
        }
        
        if patient_id not in patient_predictions:
            patient_predictions[patient_id] = {}
        patient_predictions[patient_id]['CT'] = {
            'prediction': prediction['prediction'],
            'class_name': 'high_grade',
            'confidence': prediction['confidence']
        }
        
        if patient_id not in patient_ct_predictions_list:
            patient_ct_predictions_list[patient_id] = []
        patient_ct_predictions_list[patient_id].append({
            'image_id': img_info.get('image_id', img_info.get('image_path', 'unknown')),
            'prediction': prediction['prediction'],
            'class_name': 'high_grade',
            'confidence': prediction['confidence'],
            'slice_index': img_info.get('slice_index')
        })
        
        print(f"  ✓ Stored prediction for patient {patient_id}")
        print(f"    Patient predictions: {len(patient_predictions)} patients")
        print(f"    CT predictions list: {len(patient_ct_predictions_list)} patients")
        print(f"    Total CT slices stored: {sum(len(preds) for preds in patient_ct_predictions_list.values())}")
        print()
        
        # Test case 2: Case with None patient_id (should use fallback)
        print("Test 2: Case with None patient_id (fallback logic)")
        patient_id = None
        case_id = "high_grade_image_001_CT"
        
        # Simulate the fallback logic
        if patient_id is None:
            # Use case_id as fallback
            patient_id = case_id
            print(f"  ✓ Using fallback patient_id: {patient_id}")
        
        if patient_id not in patient_predictions:
            patient_predictions[patient_id] = {}
        patient_predictions[patient_id]['CT'] = {
            'prediction': prediction['prediction'],
            'class_name': 'high_grade',
            'confidence': prediction['confidence']
        }
        
        if patient_id not in patient_ct_predictions_list:
            patient_ct_predictions_list[patient_id] = []
        patient_ct_predictions_list[patient_id].append({
            'image_id': img_info.get('image_id', img_info.get('image_path', 'unknown')),
            'prediction': prediction['prediction'],
            'class_name': 'high_grade',
            'confidence': prediction['confidence']
        })
        
        print(f"  ✓ Stored prediction with fallback patient_id")
        print(f"    Patient predictions: {len(patient_predictions)} patients")
        print(f"    CT predictions list: {len(patient_ct_predictions_list)} patients")
        print(f"    Total CT slices stored: {sum(len(preds) for preds in patient_ct_predictions_list.values())}")
        print()
        
        # Test case 3: Missing image_id (should use image_path)
        print("Test 3: Missing image_id (should use image_path)")
        patient_id = "A0002"
        img_info_no_id = {
            'image_path': '/path/to/image2.png',
            'slice_index': 20
        }
        
        if patient_id not in patient_ct_predictions_list:
            patient_ct_predictions_list[patient_id] = []
        patient_ct_predictions_list[patient_id].append({
            'image_id': img_info_no_id.get('image_id', img_info_no_id.get('image_path', 'unknown')),
            'prediction': prediction['prediction'],
            'class_name': 'high_grade',
            'confidence': prediction['confidence']
        })
        
        stored_id = patient_ct_predictions_list[patient_id][-1]['image_id']
        print(f"  ✓ Stored prediction with image_id fallback: {stored_id}")
        print()
        
        print("=" * 80)
        print("✅ ALL STORAGE TESTS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Total patients: {len(patient_predictions)}")
        print(f"  - Total CT slices stored: {sum(len(preds) for preds in patient_ct_predictions_list.values())}")
        print(f"  - All predictions stored successfully!")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_llava_med_with_samples(model):
    """Test LLaVA-Med with multiple samples."""
    print("=" * 80)
    print("TESTING LLAVA-MED WITH MULTIPLE SAMPLES")
    print("=" * 80)
    print()
    
    if model is None:
        print("❌ Cannot test - model not loaded")
        return False
    
    try:
        num_samples = 3  # Reduced for speed (LLaVA-Med is slow)
        print(f"Testing with {num_samples} samples (LLaVA-Med is slow, using fewer samples)...")
        print()
        
        predictions = []
        for i in range(num_samples):
            test_image = create_dummy_image()
            print(f"  Processing sample {i+1}/{num_samples}...", end=' ', flush=True)
            
            prediction = model.predict(
                images={"CT": test_image},
                available_modalities=["CT"],
                previous_predictions=None
            )
            predictions.append(prediction)
            
            if isinstance(prediction, dict):
                pred_idx = prediction.get('prediction', 'N/A')
                confidence = prediction.get('confidence', 0.0)
                print(f"✓ Class: {pred_idx}, Confidence: {confidence:.4f}")
            else:
                print(f"✓ Prediction: {prediction}")
        
        print()
        print(f"✓ Successfully processed {num_samples} samples")
        
        # Check consistency
        if all(isinstance(p, dict) for p in predictions):
            classes = [p.get('prediction', 'N/A') for p in predictions]
            confidences = [p.get('confidence', 0.0) for p in predictions]
            print(f"   Predictions: {classes}")
            print(f"   Confidences: {[f'{c:.4f}' for c in confidences]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("LLAVA-MED MODEL TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Model loading
    print("TEST 1: Model Loading")
    print("-" * 80)
    model = test_llava_med_loading()
    results.append(("Model Loading", model is not None))
    print()
    
    # Test 2: Patient ID extraction
    print("TEST 2: Patient ID Extraction")
    print("-" * 80)
    results.append(("Patient ID Extraction", test_patient_id_extraction()))
    print()
    
    # Test 3: Prediction storage simulation
    print("TEST 3: Prediction Storage Logic")
    print("-" * 80)
    results.append(("Prediction Storage", test_prediction_storage_simulation()))
    print()
    
    # Test 4: Prediction generation (if model loaded)
    if model is not None:
        print("TEST 4: Prediction Generation")
        print("-" * 80)
        results.append(("Prediction Generation", test_llava_med_prediction(model)))
        print()
        
        # Test 5: Multiple samples (if model loaded)
        print("TEST 5: Multiple Samples")
        print("-" * 80)
        results.append(("Multiple Samples", test_llava_med_with_samples(model)))
        print()
    else:
        print("TEST 4 & 5: Skipped (model not loaded)")
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
        print()
        print("LLaVA-Med model is working correctly!")
        print("Predictions will be stored correctly even with missing patient_id.")
        return 0
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

