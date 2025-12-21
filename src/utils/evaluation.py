"""
Evaluation utilities for sequential modality feeding.
Calculates accuracy for CT and PET (with CT context).
Focuses on certainty dynamics: confidence scores, entropy, logit distributions.
"""

import numpy as np
from typing import List, Dict, Optional


def calculate_accuracy(
    predictions: List[int],
    labels: List[int]
) -> float:
    """Calculate accuracy given predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate entropy of probability distribution.
    Higher entropy = more uncertainty.
    
    Args:
        probabilities: Array of probabilities (should sum to 1)
    
    Returns:
        Entropy value (bits)
    """
    # Ensure probabilities sum to 1 and are non-negative
    probs = np.array(probabilities)
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    probs = probs / probs.sum()  # Renormalize
    
    # Calculate entropy: H(X) = -sum(p(x) * log2(p(x)))
    log_probs = np.log2(probs)
    entropy = -np.sum(probs * log_probs)
    
    return float(entropy)


def calculate_probability_gap(probabilities: np.ndarray) -> float:
    """
    Calculate the gap between top-1 and top-2 probabilities.
    Larger gap = more confident prediction.
    
    Args:
        probabilities: Array of probabilities
    
    Returns:
        Gap between top-1 and top-2 probabilities
    """
    probs = np.array(probabilities)
    sorted_probs = np.sort(probs)[::-1]  # Descending order
    
    if len(sorted_probs) < 2:
        return 0.0
    
    return float(sorted_probs[0] - sorted_probs[1])


def calculate_logit_magnitude(logits: np.ndarray) -> float:
    """
    Calculate the magnitude (L2 norm) of logits.
    Larger magnitude = stronger signal.
    
    Args:
        logits: Array of logits
    
    Returns:
        L2 norm of logits
    """
    logits_array = np.array(logits)
    return float(np.linalg.norm(logits_array))


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Used for comparing logit distributions across modalities.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity (range: -1 to 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


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
    Focuses on certainty dynamics, not just accuracy.
    
    Args:
        results: Dictionary with keys as case_ids and values as lists of predictions
                 Each prediction dict should have 'modalities_used', 'prediction', 'label',
                 'confidence', 'probabilities', and optionally 'logits'
        modalities: Ordered list of modalities supplied via CLI (length 1 or 2)
    
    Returns:
        Dictionary with accuracy and certainty metrics for each step
    """
    # Organize by modality combinations
    step_data = {
        modalities[0]: {'predictions': [], 'labels': [], 'full_predictions': []},
    }
    if len(modalities) > 1:
        step_data[modalities[1]] = {'predictions': [], 'labels': [], 'full_predictions': []}
    
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
                step_data[step_name]['full_predictions'].append(result)  # Store full prediction dict
    
    # Calculate accuracy and certainty metrics for each step
    step_results = {}
    for step_name, data in step_data.items():
        if len(data['predictions']) > 0:
            acc = calculate_accuracy(data['predictions'], data['labels'])
            num_predictions = len(data['predictions'])
            
            # Analyze certainty metrics
            certainty_metrics = analyze_certainty_metrics(data['full_predictions'], step_name)
            
            step_results[step_name] = {
                'accuracy': acc,
                'num_samples': num_predictions,
                'certainty_metrics': certainty_metrics
            }
        else:
            step_results[step_name] = {
                'accuracy': 0.0,
                'num_samples': 0,
                'certainty_metrics': analyze_certainty_metrics([], step_name)
            }
    
    # Analyze modality agreement if we have both modalities
    agreement_metrics = None
    if len(modalities) >= 2:
        mod1 = modalities[0]
        mod2 = modalities[1]
        if mod1 in step_data and mod2 in step_data:
            mod1_preds = step_data[mod1]['full_predictions']
            mod2_preds = step_data[mod2]['full_predictions']
            
            # Extract patient IDs if available
            patient_ids = None
            if mod1_preds and len(mod1_preds) > 0:
                first_pred = mod1_preds[0]
                if isinstance(first_pred, dict) and 'patient_id' in first_pred:
                    patient_ids = [p.get('patient_id') for p in mod1_preds if isinstance(p, dict) and p.get('patient_id') is not None]
            
            agreement_metrics = analyze_modality_agreement(mod1_preds, mod2_preds, patient_ids)
    
    return {
        'step_results': step_results,
        'modalities': modalities,
        'agreement_metrics': agreement_metrics
    }


def print_evaluation_results(evaluation_results: Dict):
    """Print evaluation results with emphasis on certainty dynamics."""
    print("\n" + "="*80)
    print("Sequential Modality Evaluation Results - Certainty Analysis")
    print("="*80)
    
    step_results = evaluation_results.get('step_results', {})
    modalities = evaluation_results.get('modalities', [])
    agreement_metrics = evaluation_results.get('agreement_metrics')
    patient_agreement = evaluation_results.get('patient_level_agreement')
    
    # Print summary table (requested format)
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    print(f"{'Modality':<15} {'Avg confidence':<18} {'Entropy':<12} {'Disagreement rate':<20}")
    print("-"*80)
    
    cert_comparison = {}
    for step_name in list(step_results.keys()):
        step_data = step_results[step_name]
        cert_metrics = step_data.get('certainty_metrics', {})
        avg_conf = cert_metrics.get('avg_confidence', 0.0)
        avg_entropy = cert_metrics.get('avg_entropy', 0.0)
        
        # Get disagreement rate for this modality (same for both in binary comparison)
        disagreement_rate = 0.0
        if agreement_metrics:
            disagreement_rate = agreement_metrics.get('disagreement_rate', 0.0)
        
        cert_comparison[step_name] = {
            'avg_conf': avg_conf,
            'avg_entropy': avg_entropy
        }
        
        print(f"{step_name:<15} {avg_conf:<18.4f} {avg_entropy:<12.4f} {disagreement_rate:<20.4f}")
    
    # Print accuracy (secondary metric)
    print("\n" + "-"*80)
    print("ACCURACY (Secondary Metric)")
    print("-"*80)
    for step_name in list(step_results.keys()):
        step_data = step_results[step_name]
        acc = step_data.get('accuracy', 0.0)
        print(f"{step_name}: {acc:.4f}")
    
    print("="*80 + "\n")


def analyze_certainty_metrics(
    predictions: List[Dict],
    modality_name: str
) -> Dict:
    """
    Analyze certainty metrics for a set of predictions.
    
    Args:
        predictions: List of prediction dicts, each should have:
            - 'confidence': float
            - 'probabilities': dict or array
            - 'logits': optional array
            - 'prediction': int
        modality_name: Name of modality (e.g., 'CT', 'PET')
    
    Returns:
        Dictionary with certainty metrics
    """
    if not predictions:
        return {
            'modality': modality_name,
            'avg_confidence': 0.0,
            'avg_entropy': 0.0,
            'avg_probability_gap': 0.0,
            'avg_logit_magnitude': 0.0,
            'num_samples': 0
        }
    
    confidences = []
    entropies = []
    probability_gaps = []
    logit_magnitudes = []
    
    for pred in predictions:
        # Extract probabilities - for certainty analysis, use BEFORE boosting (more realistic)
        # This shows the true model behavior without artificial boosting effects
        probs_before_boosting = pred.get('probabilities_before_boosting')
        prob_array = None
        conf = None
        
        if probs_before_boosting is not None and len(probs_before_boosting) >= 2:
            # Use probabilities BEFORE boosting for certainty analysis (shows real model behavior)
            prob_array = np.array(probs_before_boosting)
            # Calculate confidence from probabilities before boosting (not from boosted value)
            conf = float(np.max(prob_array))
        elif pred.get('probabilities_array') is not None:
            # Fallback to probabilities_array if before_boosting not available (e.g., CT)
            probs_array = pred.get('probabilities_array')
            if len(probs_array) >= 2:
                prob_array = np.array(probs_array)
                conf = float(np.max(prob_array))
        else:
            # Fallback to probabilities dict
            probs = pred.get('probabilities', {})
            if isinstance(probs, dict) and len(probs) >= 2:
                # Convert dict to array (assume binary classification)
                # Try common class name keys
                keys = list(probs.keys())
                if len(keys) >= 2:
                    prob_array = np.array([probs[keys[0]], probs[keys[1]]])
                else:
                    prob_array = np.array(list(probs.values()))
                # If still empty, try specific names
                if prob_array.sum() == 0 or len(prob_array) < 2:
                    prob_array = np.array([
                        probs.get('high_grade', probs.get('healthy', 0.0)),
                        probs.get('low_grade', probs.get('tumor', 0.0))
                    ])
            elif isinstance(probs, (list, np.ndarray)) and len(probs) >= 2:
                prob_array = np.array(probs)
            else:
                # Default to uniform distribution if no valid probabilities found
                prob_array = np.array([0.5, 0.5])
                conf = 0.5
        
        # If we still don't have confidence, calculate from prob_array
        if conf is None:
            if prob_array is not None and len(prob_array) >= 2:
                conf = float(np.max(prob_array))
            else:
                conf = pred.get('confidence', 0.0)
        
        # Ensure prob_array is defined
        if prob_array is None:
            prob_array = np.array([0.5, 0.5])
        
        confidences.append(conf)
        
        # Ensure probabilities are valid
        if prob_array.sum() == 0 or len(prob_array) < 2:
            prob_array = np.array([0.5, 0.5])
        else:
            # Renormalize to ensure they sum to 1
            prob_array = prob_array / max(prob_array.sum(), 1e-10)  # Avoid division by zero
        
        # Calculate entropy
        entropies.append(calculate_entropy(prob_array))
        
        # Calculate probability gap
        probability_gaps.append(calculate_probability_gap(prob_array))
        
        # Extract logits if available
        logits = pred.get('logits')
        if logits is not None and len(logits) > 0:
            logit_array = np.array(logits)
            logit_magnitudes.append(calculate_logit_magnitude(logit_array))
    
    return {
        'modality': modality_name,
        'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
        'std_confidence': float(np.std(confidences)) if confidences else 0.0,
        'avg_entropy': float(np.mean(entropies)) if entropies else 0.0,
        'std_entropy': float(np.std(entropies)) if entropies else 0.0,
        'avg_probability_gap': float(np.mean(probability_gaps)) if probability_gaps else 0.0,
        'std_probability_gap': float(np.std(probability_gaps)) if probability_gaps else 0.0,
        'avg_logit_magnitude': float(np.mean(logit_magnitudes)) if logit_magnitudes else 0.0,
        'std_logit_magnitude': float(np.std(logit_magnitudes)) if logit_magnitudes else 0.0,
        'num_samples': len(predictions)
    }


def analyze_ct_context_influence(
    pet_predictions: List[Dict],
    ct_predictions: Optional[List[Dict]] = None
) -> Dict:
    """
    Analyze whether CT context influences PET predictions.
    Compares PET probabilities before and after boosting to see if CT context changes predictions.
    
    Args:
        pet_predictions: List of PET prediction dicts with 'probabilities_before_boosting'
        ct_predictions: Optional list of CT predictions for comparison
    
    Returns:
        Dictionary with context influence metrics
    """
    if not pet_predictions:
        return {
            'context_changes_prediction': 0,
            'context_increases_confidence': 0,
            'context_decreases_confidence': 0,
            'avg_confidence_change': 0.0,
            'num_samples': 0
        }
    
    changes_prediction = 0
    increases_conf = 0
    decreases_conf = 0
    confidence_changes = []
    
    for pet_pred in pet_predictions:
        probs_before = pet_pred.get('probabilities_before_boosting')
        probs_after = pet_pred.get('probabilities_array')
        
        if probs_before is not None and probs_after is not None:
            probs_before_arr = np.array(probs_before)
            probs_after_arr = np.array(probs_after)
            
            # Calculate confidence from probabilities (not from stored confidence which is after boosting)
            conf_before = float(np.max(probs_before_arr))
            conf_after = float(np.max(probs_after_arr))
            
            # Check if prediction changed
            pred_before = np.argmax(probs_before_arr)
            pred_after = np.argmax(probs_after_arr)
            
            if pred_before != pred_after:
                changes_prediction += 1
            
            # Check confidence change
            conf_change = conf_after - conf_before
            confidence_changes.append(conf_change)
            
            if conf_change > 0.01:  # Significant increase
                increases_conf += 1
            elif conf_change < -0.01:  # Significant decrease
                decreases_conf += 1
    
    total = len(pet_predictions)
    
    return {
        'context_changes_prediction': changes_prediction,
        'context_changes_prediction_rate': changes_prediction / total if total > 0 else 0.0,
        'context_increases_confidence': increases_conf,
        'context_decreases_confidence': decreases_conf,
        'avg_confidence_change': float(np.mean(confidence_changes)) if confidence_changes else 0.0,
        'std_confidence_change': float(np.std(confidence_changes)) if confidence_changes else 0.0,
        'num_samples': total
    }


def analyze_modality_agreement(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze agreement/disagreement between CT and PET predictions.
    
    Args:
        ct_predictions: List of CT prediction dicts (should have 'prediction', 'confidence')
        pet_predictions: List of PET prediction dicts (should have 'prediction', 'confidence')
        patient_ids: Optional list of patient IDs for matching (if None, assumes same order)
    
    Returns:
        Dictionary with agreement metrics
    """
    if len(ct_predictions) != len(pet_predictions):
        # Try to match by patient_id if provided
        if patient_ids is None:
            return {
                'agreement_rate': 0.0,
                'disagreement_rate': 0.0,
                'ct_dominates': 0,
                'pet_dominates': 0,
                'num_pairs': 0
            }
    
    # Match predictions by patient_id or by index
    matched_pairs = []
    if patient_ids is not None:
        # Create lookup by patient_id
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient:
                matched_pairs.append((ct_by_patient[pid], pet_by_patient[pid]))
    else:
        # Match by index
        min_len = min(len(ct_predictions), len(pet_predictions))
        matched_pairs = list(zip(ct_predictions[:min_len], pet_predictions[:min_len]))
    
    if not matched_pairs:
        return {
            'agreement_rate': 0.0,
            'disagreement_rate': 0.0,
            'ct_dominates': 0,
            'pet_dominates': 0,
            'num_pairs': 0
        }
    
    agreements = 0
    disagreements = 0
    ct_dominates = 0
    pet_dominates = 0
    
    for ct_pred, pet_pred in matched_pairs:
        ct_pred_class = ct_pred.get('prediction')
        pet_pred_class = pet_pred.get('prediction')
        ct_conf = ct_pred.get('confidence', 0.0)
        pet_conf = pet_pred.get('confidence', 0.0)
        
        if ct_pred_class == pet_pred_class:
            agreements += 1
        else:
            disagreements += 1
            
            # Check which modality has higher confidence when they disagree
            if ct_conf > pet_conf:
                ct_dominates += 1
            elif pet_conf > ct_conf:
                pet_dominates += 1
    
    total = len(matched_pairs)
    
    return {
        'agreement_rate': agreements / total if total > 0 else 0.0,
        'disagreement_rate': disagreements / total if total > 0 else 0.0,
        'ct_dominates': ct_dominates,
        'pet_dominates': pet_dominates,
        'num_pairs': total
    }


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
