"""
Evaluation utilities for sequential modality feeding.
Calculates accuracy for CT, MRI, and CT+MRI.
"""

import numpy as np
from typing import List, Dict


def calculate_accuracy(
    predictions: List[int],
    labels: List[int]
) -> float:
    """Calculate accuracy given predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def evaluate_sequential_modalities(
    results: Dict[str, List[Dict]],
    modalities: List[str]
) -> Dict:
    """
    Evaluate model performance for each modality (and optional mix).
    
    Args:
        results: Dictionary with keys as case_ids and values as lists of predictions
                 Each prediction dict should have 'modalities_used', 'prediction', 'label'
        modalities: Ordered list of modalities supplied via CLI (length 1 or 2)
    
    Returns:
        Dictionary with accuracy metrics for each step
    """
    # Organize by modality combinations
    step_data = {
        modalities[0]: {'predictions': [], 'labels': []},
    }
    if len(modalities) > 1:
        step_data[modalities[1]] = {'predictions': [], 'labels': []}
        step_data['+'.join(modalities)] = {'predictions': [], 'labels': []}
    
    for case_id, case_results in results.items():
        for result in case_results:
            mods_used = result.get('modalities_used', [])
            prediction = result['prediction']
            label = result['label']
            
            # Determine step name (explicit step overrides modality inference)
            step_name = result.get('step')
            if step_name is None:
                if len(mods_used) == 1 and mods_used[0] in step_data:
                    step_name = mods_used[0]
                elif len(mods_used) == len(modalities):
                    step_name = '+'.join(modalities)
            
            if step_name in step_data:
                step_data[step_name]['predictions'].append(prediction)
                step_data[step_name]['labels'].append(label)
    
    # Calculate accuracy for each step
    step_accuracies = {}
    for step_name, data in step_data.items():
        if len(data['predictions']) > 0:
            acc = calculate_accuracy(data['predictions'], data['labels'])
            step_accuracies[step_name] = {
                'accuracy': acc,
                'num_samples': len(data['predictions'])
            }
        else:
            step_accuracies[step_name] = {
                'accuracy': 0.0,
                'num_samples': 0
            }
    
    return {
        'step_accuracies': step_accuracies,
        'modalities': modalities
    }


def print_evaluation_results(evaluation_results: Dict):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("Sequential Modality Evaluation Results")
    print("="*60)
    
    step_accuracies = evaluation_results['step_accuracies']
    modalities = evaluation_results.get('modalities', [])
    
    print(f"\n{'Modalities':<20} {'Accuracy':<15} {'Samples':<10}")
    print("-" * 60)
    
    for step_name in list(step_accuracies.keys()):
        acc = step_accuracies[step_name]['accuracy']
        num_samples = step_accuracies[step_name]['num_samples']
        
        # For multimodal cases (e.g., "CT+PET"), show total images (pairs × modalities)
        if '+' in step_name and len(modalities) > 1:
            # Each pair contains one image from each modality
            total_images = num_samples * len(modalities)
            print(f"{step_name:<20} {acc:<15.4f} {total_images:<10}")
        else:
            print(f"{step_name:<20} {acc:<15.4f} {num_samples:<10}")
    
    print("="*60 + "\n")


def save_results(results: Dict, output_path: str):
    """Save evaluation results to file."""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")

