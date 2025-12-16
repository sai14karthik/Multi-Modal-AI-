"""
Sequential modality evaluation script.
Evaluates model performance with sequential modality evaluation:
1. CT evaluation (baseline)
2. PET evaluation (with CT context from Step 1)
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
import re
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

# Configure tqdm for better display in log files (Slurm jobs)
# Use file=sys.stderr for better compatibility with Slurm log files
import os
is_slurm = os.environ.get('SLURM_JOB_ID') is not None
tqdm_kwargs = {
    'file': sys.stderr if is_slurm else sys.stdout,  # stderr more reliable for Slurm
    'ncols': 120,  # Fixed width for better log file display
    'mininterval': 0.5,  # Update at least every 0.5 seconds
    'miniters': 1,  # Update on every iteration (for file output)
    'disable': False  # Always show progress bars
}

from src.data.config import load_dataset_config
from src.data.dataloader import get_all_images_by_modality
from src.models.model_wrapper import MultimodalModelWrapper
from src.utils.evaluation import (
    evaluate_sequential_modalities,
    print_evaluation_results,
    save_results,
    aggregate_patient_predictions
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
        help='Maximum number of images per patient per modality (e.g., 100 means 100 CT + 100 PET per patient). If None, loads ALL images from all patients'
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
    
    # Helper function to extract patient_id consistently
    # Use the patient_id that's already in the image metadata (from dataloader)
    def extract_patient_id_from_img(img):
        """Extract patient_id from image metadata.
        The dataloader already extracts patient_id correctly, so we use it directly."""
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
    
    def has_valid_patient_id(img):
        """Check if image has a valid patient_id (not None)."""
        patient_id = extract_patient_id_from_img(img)
        return patient_id is not None
    
    # Ensure all patients with both CT and PET are included
    # When max_samples=100: Distribute 100 CT and 100 PET across ALL patients that have both
    # Each patient gets both CT and PET scans
    # When max_samples=None: Process ALL images from ALL patients (still with patient matching)
    if second_modality:
        # Filter: Only keep images with valid patient_id
        first_mod_images_with_pid = [img for img in first_mod_images if has_valid_patient_id(img)]
        second_mod_images_with_pid = [img for img in second_mod_images if has_valid_patient_id(img)]
        
        # Only check if max_samples is specified
        if args.max_samples is not None:
            # This is a rough check - we'll do proper validation after grouping by patient
            if len(first_mod_images_with_pid) < args.max_samples:
                print(f"  Warning: Only {len(first_mod_images_with_pid)} {first_modality} images with valid patient_id available.")
            if len(second_mod_images_with_pid) < args.max_samples:
                print(f"  Warning: Only {len(second_mod_images_with_pid)} {second_modality} images with valid patient_id available.")
        
        # Group images by patient_id for both modalities (only valid patient_ids)
        first_mod_by_patient = {}
        for img in first_mod_images_with_pid:
            patient_id = extract_patient_id_from_img(img)
            if patient_id is not None:
                if patient_id not in first_mod_by_patient:
                    first_mod_by_patient[patient_id] = []
                first_mod_by_patient[patient_id].append(img)
        
        second_mod_by_patient = {}
        for img in second_mod_images_with_pid:
            patient_id = extract_patient_id_from_img(img)
            if patient_id is not None:
                if patient_id not in second_mod_by_patient:
                    second_mod_by_patient[patient_id] = []
                second_mod_by_patient[patient_id].append(img)
        
        # Find ALL patients that have BOTH CT and PET images
        # When max_samples=100: Try to get 100 different patients, each with 1 CT and 1 PET
        # Result: Patient 1 (CT + PET), Patient 2 (CT + PET), ..., Patient N (CT + PET)
        # If fewer patients available, use all available patients
        matched_patients = []
        for patient_id in first_mod_by_patient:
            if patient_id in second_mod_by_patient:
                # Patient must have at least 1 CT and 1 PET
                if len(first_mod_by_patient[patient_id]) >= 1 and len(second_mod_by_patient[patient_id]) >= 1:
                    matched_patients.append(patient_id)
        
        if len(matched_patients) == 0:
            parser.error(f"No patients found with both {first_modality} and {second_modality} images. Check patient_id matching.")
        
        # Shuffle for reproducibility
        random.seed(42)
        random.shuffle(matched_patients)
        
        # Select all available patients (52 patients with both CT and PET)
        # max_samples controls how many images per patient, not how many patients
        selected_patients = matched_patients
        
        # For each selected patient, take images based on max_samples
        # If max_samples is None: take ALL images from each patient
        # If max_samples is specified: take up to max_samples CT and max_samples PET per patient
        selected_first_mod = []
        selected_second_mod = []
        
        # Shuffle patient images for variety (seed once before loop for reproducibility)
        for patient_id in selected_patients:
            patient_ct_images = first_mod_by_patient[patient_id].copy()
            patient_pet_images = second_mod_by_patient[patient_id].copy()
            
            # Shuffle patient's images for variety
            random.shuffle(patient_ct_images)
            random.shuffle(patient_pet_images)
            
            if args.max_samples is None:
                # Take ALL CT and ALL PET images from this patient
                selected_first_mod.extend(patient_ct_images)
                selected_second_mod.extend(patient_pet_images)
            else:
                # Take up to max_samples CT and max_samples PET images from this patient
                # If patient has fewer images, take all available
                ct_to_take = min(args.max_samples, len(patient_ct_images))
                pet_to_take = min(args.max_samples, len(patient_pet_images))
                
                selected_first_mod.extend(patient_ct_images[:ct_to_take])
                selected_second_mod.extend(patient_pet_images[:pet_to_take])
        
        # Update image lists
        first_mod_images = selected_first_mod
        second_mod_images = selected_second_mod
    else:
        # Original behavior: shuffle and limit independently
        random.seed(42)
        random.shuffle(first_mod_images)
        if second_modality:
            random.shuffle(second_mod_images)
        
        if args.max_samples is not None and args.max_samples > 0:
            first_mod_images = first_mod_images[:args.max_samples]
            if second_modality:
                second_mod_images = second_mod_images[:args.max_samples]
    
    print("\nRunning sequential modality evaluation...")
    if second_modality:
        print(f"Steps: 1. {first_modality} ({len(first_mod_images)} instances), 2. {second_modality} ({len(second_mod_images)} instances) with {first_modality} context")
    else:
        print(f"Steps: 1. {first_modality} ({len(first_mod_images)} instances)")
    
    results = {}
    # Store CT predictions by patient_id for sequential context
    # Format: {patient_id: {'prediction': int, 'class_name': str}}
    # Each patient gets ONE aggregated CT prediction from all their slices
    # All PET images from same patient will use this aggregated CT prediction
    patient_predictions = {}
    
    # Store CT predictions by image for better matching and aggregation
    # Format: {patient_id: [{'image_id': str, 'prediction': int, 'class_name': str, 'confidence': float}, ...]}
    patient_ct_predictions_list = {}
    
    print(f"\n{'='*60}")
    print(f"Step 1: {first_modality} ({len(first_mod_images)} instances)")
    print(f"{'='*60}")
    
    # Add periodic status prints for better visibility in log files
    total_images = len(first_mod_images)
    print_interval = max(100, total_images // 20)  # Print every 5% or every 100 images
    
    for idx, img_info in enumerate(tqdm(first_mod_images, desc=f"Processing {first_modality}", **tqdm_kwargs), 1):
        # Print progress every N images
        if idx % print_interval == 0 or idx == total_images:
            progress_pct = (idx / total_images) * 100
            print(f"[Progress] Processed {idx}/{total_images} {first_modality} images ({progress_pct:.1f}%)", file=sys.stderr, flush=True)
        case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{first_modality}"
        label = img_info['label']
        patient_id = extract_patient_id_from_img(img_info)
        
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
            
            # Store CT prediction result (y_i) for this patient i
            # This CT output will be given to PET scan from the SAME patient i in Step 2
            pred_class_name = args.class_names[prediction['prediction']]  # CT output: e.g., "high_grade" or "low_grade"
            
            # Store CT prediction for patient i
            # When processing PET scan from patient i, it will get this CT result (y_i)
            # STRATEGY: If patient has multiple CT scans, keep the one with HIGHEST confidence
            # This ensures PET gets the most reliable CT prediction
            if patient_id is None:
                # Skip if no valid patient_id
                continue
            
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = {}
            
            # If this is the first CT for this patient, or if this CT has higher confidence, use it
            if first_modality not in patient_predictions[patient_id]:
                patient_predictions[patient_id][first_modality] = {
                    'prediction': prediction['prediction'],  # CT prediction result (y_i) for patient i
                    'class_name': pred_class_name,  # CT output class name for patient i: "high_grade" or "low_grade"
                    'confidence': prediction['confidence']  # Store confidence to compare with other CT scans
                }
            else:
                # Patient already has a CT prediction - keep the one with HIGHER confidence
                # This ensures PET gets the most reliable CT prediction from the same patient
                existing_confidence = patient_predictions[patient_id][first_modality].get('confidence', 0.0)
                if prediction['confidence'] > existing_confidence:
                    # This CT scan is more confident - use it instead
                    patient_predictions[patient_id][first_modality] = {
                        'prediction': prediction['prediction'],
                        'class_name': pred_class_name,
                        'confidence': prediction['confidence']
                    }
            
            # Store ALL CT predictions with their identifiers for 1-to-1 matching
            # This allows each PET image to get its corresponding CT image's prediction
            if patient_id not in patient_ct_predictions_list:
                patient_ct_predictions_list[patient_id] = []
            
            # Store CT prediction with slice_index and series_uid for matching
            ct_prediction_entry = {
                'image_id': img_info['image_id'],
                'prediction': prediction['prediction'],
                'class_name': pred_class_name,
                'confidence': prediction['confidence']
            }
            
            # Add slice_index and series_uid if available in image metadata
            if 'slice_index' in img_info:
                ct_prediction_entry['slice_index'] = img_info['slice_index']
            if 'series_uid' in img_info:
                ct_prediction_entry['series_uid'] = img_info['series_uid']
            
            patient_ct_predictions_list[patient_id].append(ct_prediction_entry)
            
            results[case_id].append({
                'modalities_used': [first_modality],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'label': label
            })
        except Exception as e:
            print(f"Error processing {case_id} with {first_modality}: {e}", file=sys.stderr)
            traceback.print_exc()
    
    # Aggregate predictions per patient for patient-level evaluation
    patient_aggregated_ct = {}
    for patient_id, slices in patient_ct_predictions_list.items():
        if patient_id is None:
            continue  # Skip invalid patient_ids
        try:
            patient_aggregated_ct[patient_id] = aggregate_patient_predictions(slices)
            # Update patient_predictions with aggregated result
            if patient_id and patient_id in patient_predictions:
                patient_predictions[patient_id][first_modality] = {
                    'prediction': patient_aggregated_ct[patient_id]['prediction'],
                    'class_name': args.class_names[patient_aggregated_ct[patient_id]['prediction']],
                    'confidence': patient_aggregated_ct[patient_id]['confidence']
                }
        except Exception as e:
            print(f"Warning: Failed to aggregate CT predictions for patient {patient_id}: {e}", file=sys.stderr)
            # Use first slice as fallback
            if slices:
                first_slice = slices[0]
                if patient_id and patient_id in patient_predictions:
                    patient_predictions[patient_id][first_modality] = {
                        'prediction': first_slice.get('prediction', 0),
                        'class_name': args.class_names[first_slice.get('prediction', 0)],
                        'confidence': first_slice.get('confidence', 0.5)
                    }
    
    # Summary of Step 1
    unique_patients_ct = len(patient_predictions)
    total_ct_predictions = sum(len(preds) for preds in patient_ct_predictions_list.values())
    print(f"\n✓ Step 1 Complete: Processed {len(first_mod_images)} {first_modality} images")
    print(f"  Stored {total_ct_predictions} {first_modality} slice predictions for {unique_patients_ct} unique patients")
    print(f"  Aggregated to {unique_patients_ct} patient-level predictions using weighted voting")
    
    if second_modality:
        print(f"\n{'='*60}")
        print(f"Step 2: {second_modality} ({len(second_mod_images)} instances)")
        print(f"Using {first_modality} predictions as context for each patient...")
        print(f"{'='*60}")
        
        # Match PET images to CT predictions by patient_id
        # Goal: Each PET sample should get CT information from the same patient
        # Patient i's PET scan gets Patient i's CT result (y_i)
        filtered_second_mod_images = []
        patients_with_ct = set(patient_predictions.keys())
        
        # Match each PET image to its corresponding CT prediction by patient_id
        # IMPORTANT: Use the SAME extraction function as Step 1 to ensure matching
        for img_info in second_mod_images:
            patient_id = extract_patient_id_from_img(img_info)  # Use same function as Step 1
            
            # Check if this patient has CT prediction from Step 1
            if patient_id in patients_with_ct and first_modality in patient_predictions[patient_id]:
                # This PET image will get CT information from the same patient
                # Patient i's PET gets Patient i's CT result
                filtered_second_mod_images.append(img_info)
        
        # All PET samples should have matching CT (due to patient matching in sampling)
        # No need for verbose output - matching is guaranteed
        
        # Initialize PET predictions list for aggregation
        patient_pet_predictions_list = {}
        
        context_used_count = 0
        total_pet_images = len(filtered_second_mod_images)
        pet_print_interval = max(100, total_pet_images // 20)  # Print every 5% or every 100 images
        
        for idx, img_info in enumerate(tqdm(filtered_second_mod_images, desc=f"Processing {second_modality} with {first_modality} context", **tqdm_kwargs), 1):
            # Print progress every N images
            if idx % pet_print_interval == 0 or idx == total_pet_images:
                progress_pct = (idx / total_pet_images) * 100
                print(f"[Progress] Processed {idx}/{total_pet_images} {second_modality} images ({progress_pct:.1f}%)", file=sys.stderr, flush=True)
            case_id = f"{img_info['class'].lower()}_{img_info['image_id']}_{second_modality}"
            label = img_info['label']
            patient_id = extract_patient_id_from_img(img_info)  # Use same extraction as Step 1
            
            if case_id not in results:
                results[case_id] = []
            
            try:
                img = Image.open(img_info['image_path']).convert('RGB')
                
                # Get CT prediction result (y_i) from patient i (from Step 1)
                # BEST STRATEGY: 1-to-1 matching with intelligent fallbacks
                # 1. Try exact match by slice_index
                # 2. If not found, try nearest slice_index
                # 3. If still not found, use patient-level (highest confidence CT)
                previous_predictions = None
                context_used = False
                
                if patient_id and patient_id in patient_predictions and first_modality in patient_predictions[patient_id]:
                    matched_ct_prediction = None
                    pet_slice_index = img_info.get('slice_index')
                    
                    if patient_id in patient_ct_predictions_list and pet_slice_index is not None:
                        # Strategy 1: Try exact match by slice_index
                        for ct_pred in patient_ct_predictions_list[patient_id]:
                            if 'slice_index' in ct_pred and ct_pred['slice_index'] == pet_slice_index:
                                matched_ct_prediction = ct_pred
                                break
                        
                        # Strategy 2: If no exact match, find nearest slice_index
                        if matched_ct_prediction is None:
                            min_distance = float('inf')
                            for ct_pred in patient_ct_predictions_list[patient_id]:
                                if 'slice_index' in ct_pred:
                                    distance = abs(ct_pred['slice_index'] - pet_slice_index)
                                    if distance < min_distance:
                                        min_distance = distance
                                        matched_ct_prediction = ct_pred
                    
                    # Strategy 3: Use matched CT prediction, or fallback to patient-level
                    if matched_ct_prediction is not None:
                        # Use 1-to-1 matched CT prediction (include confidence for boosting)
                        patient_i_ct_result = {
                            'prediction': matched_ct_prediction['prediction'],
                            'class_name': matched_ct_prediction['class_name'],
                            'confidence': matched_ct_prediction.get('confidence', 0.5)  # Include confidence for PET boosting
                        }
                    else:
                        # Fallback: Use patient-level CT prediction (highest confidence)
                        patient_i_ct_result = patient_predictions[patient_id][first_modality]
                        # Ensure confidence is present
                        if 'confidence' not in patient_i_ct_result:
                            patient_i_ct_result['confidence'] = 0.5
                    
                    previous_predictions = {
                        first_modality: patient_i_ct_result  # Patient i's CT result → Patient i's PET
                    }
                    context_used = True
                    context_used_count += 1
                
                # Model.predict() receives:
                # 1. Patient i's PET image
                # 2. Patient i's CT output result (y_i) from Step 1
                # The model uses prompts like: "Given that the CT scan showed high_grade, this PET scan shows..."
                # Simple: "hey CT gave this result, what's for PET?"
                prediction = model.predict(
                    images={second_modality: img},  # Patient i's PET image
                    available_modalities=[second_modality],
                    batch_size=args.batch_size,
                    preprocess=not args.no_preprocess,
                    temperature=args.temperature,
                    use_weighted_ensemble=not args.no_weighted_ensemble,
                    try_both_swaps=not args.no_swap_test,
                    aggressive_preprocess=args.aggressive_preprocess,
                    previous_predictions=previous_predictions  # Patient i's CT result (y_i) → given to Patient i's PET
                )
                
                # Store prediction for this patient
                if patient_id is None:
                    # Skip if no valid patient_id
                    continue
                
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = {}
                pred_class_name = args.class_names[prediction['prediction']]
                
                # Store all PET predictions for aggregation
                if patient_id not in patient_pet_predictions_list:
                    patient_pet_predictions_list[patient_id] = []
                
                pet_prediction_entry = {
                    'image_id': img_info['image_id'],
                    'prediction': prediction['prediction'],
                    'class_name': pred_class_name,
                    'confidence': prediction['confidence']
                }
                
                if 'slice_index' in img_info:
                    pet_prediction_entry['slice_index'] = img_info['slice_index']
                if 'series_uid' in img_info:
                    pet_prediction_entry['series_uid'] = img_info['series_uid']
                
                patient_pet_predictions_list[patient_id].append(pet_prediction_entry)
                
                results[case_id].append({
                    'modalities_used': [second_modality],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'label': label,
                    'used_context': context_used,
                    'context_from': [first_modality] if context_used else []
                })
            except Exception as e:
                print(f"Error processing {case_id} with {second_modality}: {e}", file=sys.stderr)
                traceback.print_exc()
        
        # Aggregate PET predictions per patient
        patient_aggregated_pet = {}
        for patient_id, slices in patient_pet_predictions_list.items():
            if patient_id is None:
                continue  # Skip invalid patient_ids
            try:
                patient_aggregated_pet[patient_id] = aggregate_patient_predictions(slices)
                # Update patient_predictions with aggregated result
                if patient_id and patient_id in patient_predictions:
                    patient_predictions[patient_id][second_modality] = {
                        'prediction': patient_aggregated_pet[patient_id]['prediction'],
                        'class_name': args.class_names[patient_aggregated_pet[patient_id]['prediction']],
                        'confidence': patient_aggregated_pet[patient_id]['confidence']
                    }
            except Exception as e:
                print(f"Warning: Failed to aggregate PET predictions for patient {patient_id}: {e}", file=sys.stderr)
                # Use first slice as fallback
                if slices:
                    first_slice = slices[0]
                    if patient_id and patient_id in patient_predictions:
                        patient_predictions[patient_id][second_modality] = {
                            'prediction': first_slice.get('prediction', 0),
                            'class_name': args.class_names[first_slice.get('prediction', 0)],
                            'confidence': first_slice.get('confidence', 0.5)
                        }
        
        total_pet_predictions = sum(len(preds) for preds in patient_pet_predictions_list.values())
        unique_patients_pet = len(patient_pet_predictions_list)
        print(f"\n✓ Processed {context_used_count} {second_modality} images using {first_modality} context")
        print(f"  Aggregated {total_pet_predictions} {second_modality} slice predictions to {unique_patients_pet} patient-level predictions")
    
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

