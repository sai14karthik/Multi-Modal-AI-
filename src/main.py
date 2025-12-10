"""
Sequential modality evaluation script.
Evaluates model performance with single modalities and multimodal combinations.
Supports CT, PET, and other medical imaging modalities.
"""

# Set up stderr filtering FIRST, before any imports that might trigger warnings
import sys
import os
import warnings

# Suppress stderr for cadam32bit warning - must be before any imports
_original_stderr = sys.stderr
class FilteredStderr:
    def __init__(self):
        self._original = _original_stderr
    
    def write(self, s):
        if s:
            # Check if this is the specific warning we want to filter
            s_lower = s.lower()
            should_filter = (
                'cadam32bit' in s_lower or
                'text_config_dict' in s or
                'cliptextconfig' in s_lower or
                ('overriden' in s_lower and 'text_config' in s_lower) or
                ('nonetype' in s_lower and 'cadam32bit' in s_lower)
            )
            if not should_filter:
                self._original.write(s)
    
    def flush(self):
        self._original.flush()
    
    def __getattr__(self, name):
        # Forward any other attributes to original stderr
        return getattr(self._original, name)

sys.stderr = FilteredStderr()

# Set environment variables early to prevent warnings
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
import random
import traceback

from src.utils.tf_mock import ensure_tensorflow_stub
ensure_tensorflow_stub()

# Suppress additional warnings after imports
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*bitsandbytes.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')
warnings.filterwarnings('ignore', message='.*Redirects.*')
warnings.filterwarnings('ignore', message='.*trust_remote_code.*')
warnings.filterwarnings('ignore', message='.*text_config.*')
warnings.filterwarnings('ignore', message='.*cadam32bit.*')
warnings.filterwarnings('ignore', message='.*cadam32bit_grad_fp32.*')
warnings.filterwarnings('ignore', message='.*text_config_dict.*')
warnings.filterwarnings('ignore', message='.*CLIPTextConfig.*')
warnings.filterwarnings('ignore', message='.*will be overriden.*')
warnings.filterwarnings('ignore', message='.*overriden.*')
warnings.filterwarnings('ignore', message='.*id2label.*')
warnings.filterwarnings('ignore', message='.*bos_token_id.*')
warnings.filterwarnings('ignore', message='.*eos_token_id.*')

from PIL import Image
import torch
from tqdm import tqdm

from src.data.config import load_dataset_config
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
        '--model_arch',
        type=str,
        choices=['clip', 'llava', 'llava_med'],
        default='clip',
        help='Model architecture to use (clip, llava, or llava_med). Default: clip'
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
    parser.add_argument(
        '--no_preprocess',
        action='store_true',
        help='Disable medical image preprocessing (contrast enhancement, histogram equalization)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature scaling for logits (default: 0.8, lower = more confident predictions). Recommended: 0.7-0.9 for better calibration'
    )
    parser.add_argument(
        '--aggressive_preprocess',
        action='store_true',
        help='Use aggressive preprocessing (stronger contrast/sharpness enhancement). May improve difficult cases but can introduce artifacts'
    )
    parser.add_argument(
        '--no_weighted_ensemble',
        action='store_true',
        help='Disable weighted prompt ensemble (use simple mean instead)'
    )
    parser.add_argument(
        '--no_swap_test',
        action='store_true',
        help='Disable testing both swap strategies (use original swap only)'
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        default=None,
        help='Optional path to dataset_config.yaml describing modality folders and classes'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Optional split name (e.g., train/val/test) to filter images using metadata'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='Override class names used for prompts and labels (expects exactly two names)'
    )
    parser.add_argument(
        '--allow_single_modality',
        action='store_true',
        help='Permit running with a single modality (steps 2 and 3 will be skipped)'
    )
    
    args = parser.parse_args()
    
    if len(args.modalities) == 0:
        parser.error("Please provide at least one modality (e.g., CT).")
    
    # Use only the first two modalities for sequential evaluation
    args.modalities = args.modalities[:2]
    
    if len(args.modalities) < 2 and not args.allow_single_modality:
        parser.error("Please provide at least two distinct modalities (e.g., CT MRI).")
    
    if len(args.modalities) == 2 and len(set(args.modalities)) != 2:
        parser.error("Modalities for sequential evaluation must be distinct.")
    
    if not os.path.isdir(args.data_root):
        parser.error(f"Data root directory not found: {args.data_root}")

    dataset_config = load_dataset_config(args.dataset_config)

    if args.class_names is not None:
        if len(args.class_names) != 2:
            parser.error("Please provide exactly two class names (e.g., Adenocarcinoma Squamous).")
        class_names = args.class_names
    else:
        class_names = None
        for mod in args.modalities:
            mod_cfg = dataset_config.get('modalities', {}).get(mod, {})
            classes_cfg = mod_cfg.get('classes')
            if classes_cfg:
                # Sort by label index when available
                sorted_classes = sorted(classes_cfg.items(), key=lambda kv: kv[1])
                class_names = [name for name, _ in sorted_classes]
                break
        if class_names is None:
            class_names = ['Healthy', 'Tumor']
    args.class_names = class_names
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nLoading data from {args.data_root}...")
    
    # Load model
    if args.model_arch == 'llava':
        from src.models.llava_runner import LLaVARunner
        model = LLaVARunner(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names
        )
    elif args.model_arch == 'llava_med':
        from src.models.llava_med_runner import LLaVAMedRunner
        model = LLaVAMedRunner(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names
        )
    else:
        model = MultimodalModelWrapper(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names
        )
    
    print("\nOrganizing images by modality...")
    
    first_modality = args.modalities[0]
    second_modality = args.modalities[1] if len(args.modalities) > 1 else None
    
    print(f"  Loading {first_modality} images...")
    first_mod_images = get_all_images_by_modality(
        args.data_root,
        first_modality,
        classes=args.class_names,
        dataset_config_path=args.dataset_config,
        split=args.split
    )
    second_mod_images = []
    if second_modality:
        print(f"  Loading {second_modality} images...")
        second_mod_images = get_all_images_by_modality(
            args.data_root,
            second_modality,
            classes=args.class_names,
            dataset_config_path=args.dataset_config,
            split=args.split
        )
    
    if not first_mod_images:
        parser.error(f"No images found for modality '{first_modality}'. Check folder structure and casing.")
    if second_modality and not second_mod_images:
        parser.error(f"No images found for modality '{second_modality}'. Check folder structure and casing.")
    
    print(f"  - Total {first_modality} images: {len(first_mod_images)}")
    if second_modality:
        print(f"  - Total {second_modality} images: {len(second_mod_images)}")
    
    # Create patient-level pairs for multimodal evaluation FIRST (before limiting)
    paired_images = []
    if second_modality:
        # Group images by patient_id (or image_id if patient_id unavailable) and class
        first_mod_by_patient = {}
        for img in first_mod_images:
            # Use patient_id if available, otherwise fall back to image_id
            patient_key = img.get('patient_id') if img.get('patient_id') is not None else img.get('image_id')
            key = (patient_key, img['class'])
            if key not in first_mod_by_patient:
                first_mod_by_patient[key] = []
            first_mod_by_patient[key].append(img)
        
        second_mod_by_patient = {}
        for img in second_mod_images:
            # Use patient_id if available, otherwise fall back to image_id
            patient_key = img.get('patient_id') if img.get('patient_id') is not None else img.get('image_id')
            key = (patient_key, img['class'])
            if key not in second_mod_by_patient:
                second_mod_by_patient[key] = []
            second_mod_by_patient[key].append(img)
        
        # Create pairs from same patient/image_id and class
        for key in first_mod_by_patient:
            if key in second_mod_by_patient:
                first_imgs = first_mod_by_patient[key]
                second_imgs = second_mod_by_patient[key]
                # Pair up images (take minimum length to ensure pairs)
                min_len = min(len(first_imgs), len(second_imgs))
                for i in range(min_len):
                    paired_images.append({
                        first_modality: first_imgs[i],
                        second_modality: second_imgs[i],
                        'patient_id': key[0],
                        'class': key[1],
                        'label': first_imgs[i]['label']
                    })
        
        # Shuffle pairs with same seed for reproducibility
        random.seed(42)
        random.shuffle(paired_images)
        print(f"  - Total {first_modality}+{second_modality} pairs: {len(paired_images)}")
    
    # Now limit images AFTER pairing (so we have more pairs available)
    # Reset seed for single-modality shuffles (ensures reproducibility)
    random.seed(42)
    random.shuffle(first_mod_images)
    random.shuffle(second_mod_images)
    
    if args.max_samples is not None and args.max_samples > 0:
        first_mod_images = first_mod_images[:args.max_samples]
        second_mod_images = second_mod_images[:args.max_samples]
        # Also limit paired images
        if paired_images:
            paired_images = paired_images[:args.max_samples]
    
    print("\nRunning sequential modality evaluation...")
    if second_modality:
        print(f"Steps: 1. {first_modality} ({len(first_mod_images)} instances), 2. {second_modality} ({len(second_mod_images)} instances), 3. {first_modality}+{second_modality} pairs ({len(paired_images)} instances)")
    else:
        print(f"Steps: 1. {first_modality} ({len(first_mod_images)} instances)")
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Step 1: {first_modality} ({len(first_mod_images)} instances)")
    print(f"{'='*60}")
    
    for img_info in tqdm(first_mod_images, desc=f"Processing {first_modality}"):
        case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{first_modality}"
        label = img_info['label']
        
        if case_id not in results:
            results[case_id] = []
        
        try:
            img = Image.open(img_info['image_path']).convert('RGB')
            
            prediction = model.predict(
                images={first_modality: img},
                available_modalities=[first_modality],
                batch_size=args.batch_size,
                preprocess=not args.no_preprocess,
                temperature=args.temperature,
                use_weighted_ensemble=not args.no_weighted_ensemble,
                try_both_swaps=not args.no_swap_test,
                aggressive_preprocess=args.aggressive_preprocess
            )
            results[case_id].append({
                'modalities_used': [first_modality],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'label': label
            })
        except Exception as e:
            print(f"Error processing {case_id} with {first_modality}: {e}", file=sys.stderr)
            traceback.print_exc()
    
    if second_modality:
        print(f"\n{'='*60}")
        print(f"Step 2: {second_modality} ({len(second_mod_images)} instances)")
        print(f"{'='*60}")
        
        for img_info in tqdm(second_mod_images, desc=f"Processing {second_modality}"):
            case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{second_modality}"
            label = img_info['label']
            
            if case_id not in results:
                results[case_id] = []
            
            try:
                img = Image.open(img_info['image_path']).convert('RGB')
                
                prediction = model.predict(
                    images={second_modality: img},
                    available_modalities=[second_modality],
                    batch_size=args.batch_size,
                    preprocess=not args.no_preprocess,
                    temperature=args.temperature,
                    use_weighted_ensemble=not args.no_weighted_ensemble,
                    try_both_swaps=not args.no_swap_test,
                    aggressive_preprocess=args.aggressive_preprocess
                )
                results[case_id].append({
                    'modalities_used': [second_modality],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'label': label
                })
            except Exception as e:
                print(f"Error processing {case_id} with {second_modality}: {e}", file=sys.stderr)
                traceback.print_exc()
    
    if second_modality and paired_images:
        print(f"\n{'='*60}")
        print(f"Step 3: {first_modality}+{second_modality} pairs ({len(paired_images)} instances)")
        print(f"{'='*60}")
        
        for pair in tqdm(paired_images, desc=f"Processing {first_modality}+{second_modality}"):
            first_img_info = pair[first_modality]
            second_img_info = pair[second_modality]
            patient_id = pair.get('patient_id', 'unknown')
            class_name = pair['class']
            label = pair['label']
            
            case_id = f"{class_name.lower()}_{patient_id}_{first_modality}_{second_modality}_pair"
            
            if case_id not in results:
                results[case_id] = []
            
            try:
                # Load both images
                first_img = Image.open(first_img_info['image_path']).convert('RGB')
                second_img = Image.open(second_img_info['image_path']).convert('RGB')
                
                # Pass both images to model for multimodal evaluation
                prediction = model.predict(
                    images={
                        first_modality: first_img,
                        second_modality: second_img
                    },
                    available_modalities=[first_modality, second_modality],
                    batch_size=args.batch_size,
                    preprocess=not args.no_preprocess,
                    temperature=args.temperature,
                    use_weighted_ensemble=not args.no_weighted_ensemble,
                    try_both_swaps=not args.no_swap_test,
                    aggressive_preprocess=args.aggressive_preprocess
                )
                results[case_id].append({
                    'modalities_used': [first_modality, second_modality],
                    'step': f"{first_modality}+{second_modality}",
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'label': label
                })
            except Exception as e:
                print(f"Error processing {case_id} with {first_modality}+{second_modality}: {e}", file=sys.stderr)
                traceback.print_exc()
    
    # Evaluate results
    print("\nEvaluating results...")
    if not results:
        print("Warning: No results to evaluate. Check if images were processed successfully.", file=sys.stderr)
        return
    
    evaluation_results = evaluate_sequential_modalities(results, args.modalities)
    
    # Print results
    print_evaluation_results(evaluation_results)
    
    # Save results
    output_file = os.path.join(
        args.output_dir,
        f'results_{args.model_name.replace("/", "_")}.json'
    )
    save_results(evaluation_results, output_file)


if __name__ == '__main__':
    main()

