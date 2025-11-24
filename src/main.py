"""
Sequential modality evaluation script.
Evaluates model performance with CT only, MRI only, and CT+MRI.
"""

import os
import sys
import types
import importlib.util

# Suppress transformers warnings and prevent TensorFlow import errors
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mock TensorFlow before transformers tries to import it
if 'tensorflow' not in sys.modules:
    tf_mock = types.ModuleType('tensorflow')
    # Create a proper spec for the mock
    spec = importlib.util.spec_from_loader('tensorflow', loader=None)
    tf_mock.__spec__ = spec
    tf_mock.__version__ = '2.0.0'
    
    # Add common TensorFlow attributes that transformers might check
    class MockTensor:
        pass
    class MockVariable:
        pass
    tf_mock.Tensor = MockTensor
    tf_mock.Variable = MockVariable
    
    # Mock submodules
    tf_mock.image = types.ModuleType('tensorflow.image')
    tf_mock.nn = types.ModuleType('tensorflow.nn')
    
    sys.modules['tensorflow'] = tf_mock

import argparse
import random
import torch
from tqdm import tqdm
from PIL import Image

from src.data.dataloader import get_all_images_by_modality
from src.models.model_wrapper import MultimodalModelWrapper
from src.utils.evaluation import (
    evaluate_sequential_modalities,
    print_evaluation_results,
    save_results
)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Modality Evaluation for Multimodal Models"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing modality-specific folders'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['CT', 'MRI'],
        help='List of modalities to evaluate (e.g., CT MRI mammography)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='openai/clip-vit-large-patch14',
        help='HuggingFace model name (default: openai/clip-vit-large-patch14)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of images per modality (e.g., 40 means 40 MRI + 40 CT + 80 pairs). If None, processes all images'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1). Use larger values (e.g., 8, 16) for faster GPU processing'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nLoading data from {args.data_root}...")
    
    # Load model
    model = MultimodalModelWrapper(
        model_name=args.model_name,
        device=device
    )
    
    print("\nOrganizing images by modality...")
    
    mri_images = get_all_images_by_modality(args.data_root, args.modalities[1])
    ct_images = get_all_images_by_modality(args.data_root, args.modalities[0])
    
    print(f"  - Total {args.modalities[1]} images: {len(mri_images)}")
    print(f"  - Total {args.modalities[0]} images: {len(ct_images)}")
    
    print(f"  - Total {args.modalities[0]}+{args.modalities[1]} mix: {len(mri_images) + len(ct_images)}")
    
    if args.max_samples is not None:
        mri_images = mri_images[:args.max_samples] if len(mri_images) > args.max_samples else mri_images
        ct_images = ct_images[:args.max_samples] if len(ct_images) > args.max_samples else ct_images
        
    random.seed(42)
    random.shuffle(mri_images)
    random.shuffle(ct_images)
    ct_mri_mix = ct_images + mri_images
    random.shuffle(ct_mri_mix)
    
    print("\nRunning sequential modality evaluation...")
    print(f"Steps: 1. {args.modalities[1]} ({len(mri_images)} instances), 2. {args.modalities[0]} ({len(ct_images)} instances), 3. {args.modalities[0]}+{args.modalities[1]} mix ({len(ct_mri_mix)} instances)")
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Step 1: {args.modalities[1]} ({len(mri_images)} instances)")
    print(f"{'='*60}")
    
    for img_info in tqdm(mri_images, desc=f"Processing {args.modalities[1]}"):
        case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{args.modalities[1]}"
        label = img_info['label']
        
        if case_id not in results:
            results[case_id] = []
        
        try:
            img = Image.open(img_info['image_path']).convert('RGB')
            
            prediction = model.predict(
                images={args.modalities[1]: img},
                available_modalities=[args.modalities[1]],
                batch_size=args.batch_size
            )
            results[case_id].append({
                'modalities_used': [args.modalities[1]],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'label': label
            })
        except Exception as e:
            print(f"Error processing {case_id} with {args.modalities[1]}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Step 2: {args.modalities[0]} ({len(ct_images)} instances)")
    print(f"{'='*60}")
    
    for img_info in tqdm(ct_images, desc=f"Processing {args.modalities[0]}"):
        case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{args.modalities[0]}"
        label = img_info['label']
        
        if case_id not in results:
            results[case_id] = []
        
        try:
            img = Image.open(img_info['image_path']).convert('RGB')
            
            prediction = model.predict(
                images={args.modalities[0]: img},
                available_modalities=[args.modalities[0]],
                batch_size=args.batch_size
            )
            results[case_id].append({
                'modalities_used': [args.modalities[0]],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'label': label
            })
        except Exception as e:
            print(f"Error processing {case_id} with {args.modalities[0]}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Step 3: {args.modalities[0]}+{args.modalities[1]} mix ({len(ct_mri_mix)} instances)")
    print(f"{'='*60}")
    
    for img_info in tqdm(ct_mri_mix, desc=f"Processing {args.modalities[0]}+{args.modalities[1]}"):
        mod = img_info['modality']
        case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{mod}_mix"
        label = img_info['label']
        
        if case_id not in results:
            results[case_id] = []
        
        try:
            img = Image.open(img_info['image_path']).convert('RGB')
            
            prediction = model.predict(
                images={mod: img},
                available_modalities=[mod],
                batch_size=args.batch_size
            )
            results[case_id].append({
                'modalities_used': [mod],
                'step': f"{args.modalities[0]}+{args.modalities[1]}",
                'image_modality': mod,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'label': label
            })
        except Exception as e:
            print(f"Error processing {case_id} with {args.modalities}: {e}")
    
    # Evaluate results
    print("\nEvaluating results...")
    evaluation_results = evaluate_sequential_modalities(results, args.modalities)
    
    # Print results
    print_evaluation_results(evaluation_results)
    
    # Save results
    output_file = os.path.join(
        args.output_dir,
        f'results_{args.model_name.replace("/", "_")}.json'
    )
    


if __name__ == '__main__':
    main()

