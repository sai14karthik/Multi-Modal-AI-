"""
Data loader for sequential modality feeding.
"""


import os
import re
import glob
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from src.data.config import load_dataset_config, resolve_metadata_path


def _extract_image_number(filename):
    """Extract number from filename like 'ct_healthy (123).jpg' -> 123"""
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


_METADATA_CACHE: Dict[str, pd.DataFrame] = {}


def _load_metadata(metadata_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not metadata_path:
        return None
    if metadata_path in _METADATA_CACHE:
        return _METADATA_CACHE[metadata_path]
    if not os.path.exists(metadata_path):
        return None
    df = pd.read_csv(metadata_path)
    _METADATA_CACHE[metadata_path] = df
    return df


def get_all_images_by_modality(
    data_root: str,
    modality: str,
    classes: Optional[List[str]] = None,
    dataset_config_path: Optional[str] = None,
    split: Optional[str] = None
) -> List[Dict]:
    """
    Get ALL images from a specific modality folder, honoring dataset_config metadata.
    
    Args:
        data_root: Root directory containing modality-specific folders
        modality: Modality name (e.g., 'CT', 'MRI')
        classes: Optional list of class folders to search
        dataset_config_path: Optional YAML path describing folders/classes metadata
        split: Optional split name to filter using metadata.csv
    
    Returns:
        List of dictionaries with 'image_path', 'class', 'label', and 'image_id'
    """
    dataset_config = load_dataset_config(dataset_config_path)
    modality_cfg = dataset_config.get('modalities', {}).get(modality, {})
    folder_name = modality_cfg.get('folder', modality)
    class_mapping = modality_cfg.get('classes', {})

    if classes is None:
        classes = list(class_mapping.keys()) if class_mapping else ['Healthy', 'Tumor']

    # Ensure labels exist for classes encountered
    class_to_label = {cls: class_mapping.get(cls, idx) for idx, cls in enumerate(classes)}
    all_images = []
    
    metadata_path = resolve_metadata_path(dataset_config, data_root)
    metadata_df = _load_metadata(metadata_path)
    metadata_cfg = dataset_config.get('metadata', {})
    split_column = metadata_cfg.get('split_column', 'split')
    image_column = metadata_cfg.get('image_column', 'image_path')
    filtered_paths = None
    metadata_lookup = {}  # Optimize: create lookup dict instead of filtering each time

    if metadata_df is not None:
        df = metadata_df
        if split:
            df = df[df[split_column] == split]
        filtered_paths = set(df[image_column].tolist())
        # Create fast lookup dictionary: image_path -> {patient_id, slice_index, series_uid, ...}
        for _, row in df.iterrows():
            img_path = row.get(image_column)
            if img_path:
                metadata_lookup[img_path] = {
                    'patient_id': row.get('patient_id', None),
                    'slice_index': row.get('slice_index', None),
                    'series_uid': row.get('series_uid', None)
                }
    
    folder_root = os.path.join(data_root, folder_name)
    existing_dirs = {}
    if os.path.isdir(folder_root):
        for entry in os.listdir(folder_root):
            potential_path = os.path.join(folder_root, entry)
            if os.path.isdir(potential_path):
                existing_dirs[entry.lower()] = entry

    for class_name in classes:
        folder_override = existing_dirs.get(class_name.lower(), class_name)
        class_path = os.path.join(data_root, folder_name, folder_override)
        if not os.path.exists(class_path):
            continue
        
        image_files = glob.glob(os.path.join(class_path, '*'))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        
        # Use tqdm for progress indication
        for img_path in tqdm(image_files, desc=f"Loading {modality}/{class_name}", leave=False):
            filename = os.path.basename(img_path)
            img_number = _extract_image_number(filename)
            rel_path = os.path.relpath(img_path, data_root)

            if filtered_paths is not None and rel_path not in filtered_paths:
                continue

            # Extract metadata if available (optimized lookup)
            patient_id = None
            slice_index = None
            series_uid = None
            
            if metadata_lookup:
                metadata_entry = metadata_lookup.get(rel_path)
                if metadata_entry:
                    if isinstance(metadata_entry, dict):
                        patient_id = metadata_entry.get('patient_id')
                        slice_index = metadata_entry.get('slice_index')
                        series_uid = metadata_entry.get('series_uid')
                    else:
                        # Backward compatibility: if it's just patient_id
                        patient_id = metadata_entry
            
            # If patient_id not in metadata, try to extract from filename
            if patient_id is None:
                # Try common patterns: A0001, patient_001, etc.
                try:
                    patient_match = re.search(r'([A-Z]?\d{4,})', filename)
                    if patient_match:
                        patient_id = patient_match.group(1)
                except Exception:
                    # If regex fails, continue without patient_id
                    pass
            
            # Convert slice_index to int if it's a valid number
            if slice_index is not None:
                try:
                    slice_index = int(float(slice_index))  # Handle both int and float strings
                except (ValueError, TypeError):
                    slice_index = None
            
            all_images.append({
                'image_path': img_path,
                'class': class_name,
                'label': class_to_label.get(class_name, -1),
                'image_id': img_number,
                'modality': modality,
                'patient_id': patient_id,
                'slice_index': slice_index,
                'series_uid': series_uid
            })
    
    return all_images

