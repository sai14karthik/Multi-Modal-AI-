"""
Evaluation utilities for sequential modality feeding.
Calculates accuracy for CT and PET (with CT context).
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


def aggregate_patient_predictions(patient_slices: List[Dict]) -> Dict:
    """
    Aggregate predictions from multiple slices per patient.
    
    Strategy: Weighted voting by confidence
    - Each slice's prediction is weighted by its confidence
    - Final prediction is the class with highest weighted sum
    - Tie-breaking: Use class with higher average confidence
    
    Args:
        patient_slices: List of prediction dicts, each with 'prediction' and 'confidence'
    
    Returns:
        Dictionary with aggregated 'prediction', 'confidence', and 'num_slices'
    """
    if not patient_slices:
        return {'prediction': 0, 'confidence': 0.0, 'num_slices': 0}
    
    # Weighted voting: sum confidence for each class
    class_weights = {0: 0.0, 1: 0.0}
    class_confidences = {0: [], 1: []}  # Track individual confidences for tie-breaking
    total_confidence = 0.0
    
    for slice_pred in patient_slices:
        # Validate prediction is 0 or 1
        pred = slice_pred.get('prediction', 0)
        pred = max(0, min(1, int(pred)))  # Clamp to valid range [0, 1]
        
        # Validate confidence is in valid range [0, 1]
        conf = slice_pred.get('confidence', 0.5)
        conf = max(0.0, min(1.0, float(conf)))  # Clamp to valid range
        
        class_weights[pred] += conf
        class_confidences[pred].append(conf)
        total_confidence += conf
    
    # Get prediction with highest weighted sum
    # Handle tie-breaking: if weights are equal, use class with higher average confidence
    if class_weights[0] == class_weights[1]:
        # Tie-breaking: compare average confidence
        avg_conf_0 = sum(class_confidences[0]) / len(class_confidences[0]) if class_confidences[0] else 0.0
        avg_conf_1 = sum(class_confidences[1]) / len(class_confidences[1]) if class_confidences[1] else 0.0
        aggregated_pred = 1 if avg_conf_1 > avg_conf_0 else 0
    else:
        aggregated_pred = max(class_weights.items(), key=lambda x: x[1])[0]
    
    # Calculate aggregated confidence (normalized)
    if total_confidence > 0:
        aggregated_conf = class_weights[aggregated_pred] / total_confidence
    else:
        aggregated_conf = 0.5
    
    return {
        'prediction': aggregated_pred,
        'confidence': aggregated_conf,
        'num_slices': len(patient_slices)
    }


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
        # Note: Removed CT+PET multimodal step - only CT and PET (with CT context) are evaluated
    
    for case_id, case_results in results.items():
        for result in case_results:
            mods_used = result.get('modalities_used', [])
            prediction = result.get('prediction')
            label = result.get('label')
            
            # Skip if required fields are missing
            if prediction is None or label is None:
                continue
            
            # Determine step name (explicit step overrides modality inference)
            step_name = result.get('step')
            if step_name is None:
                if len(mods_used) == 1 and mods_used[0] in step_data:
                    step_name = mods_used[0]
                else:
                    # Skip if we can't determine the step
                    continue
            
            if step_name in step_data:
                step_data[step_name]['predictions'].append(prediction)
                step_data[step_name]['labels'].append(label)
    
    # Calculate accuracy for each step
    step_accuracies = {}
    for step_name, data in step_data.items():
        if len(data['predictions']) > 0:
            acc = calculate_accuracy(data['predictions'], data['labels'])
            num_predictions = len(data['predictions'])
            num_samples = num_predictions
            
            step_accuracies[step_name] = {
                'accuracy': acc,
                'num_samples': num_samples
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
