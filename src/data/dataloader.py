"""
Data loader for sequential modality feeding.
"""


import os
import re
import glob
from typing import List, Dict


def _extract_image_number(filename):
    """Extract number from filename like 'ct_healthy (123).jpg' -> 123"""
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def get_all_images_by_modality(
    data_root: str,
    modality: str,
    classes: List[str] = ['Healthy', 'Tumor']
) -> List[Dict]:
    """
    Get ALL images from a specific modality folder, regardless of matching.
    
    Args:
        data_root: Root directory containing modality-specific folders
        modality: Modality name (e.g., 'CT', 'MRI')
        classes: List of class folders to search (e.g., ['Healthy', 'Tumor'])
    
    Returns:
        List of dictionaries with 'image_path', 'class', 'label', and 'image_id'
    """
    modality_folders = {
        'CT': 'Brain Tumor CT scan Images',
        'MRI': 'Brain Tumor MRI images',
    }
    
    folder_name = modality_folders.get(modality, modality)
    class_to_label = {'Healthy': 0, 'Tumor': 1}
    all_images = []
    
    for class_name in classes:
        class_path = os.path.join(data_root, folder_name, class_name)
        if not os.path.exists(class_path):
            continue
        
        image_files = glob.glob(os.path.join(class_path, '*'))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            img_number = _extract_image_number(filename)
            
            all_images.append({
                'image_path': img_path,
                'class': class_name,
                'label': class_to_label.get(class_name, -1),
                'image_id': img_number,
                'modality': modality
            })
    
    return all_images

