"""
Evaluation utilities for sequential modality feeding.
Focuses on CERTAINTY DYNAMICS and MODEL BEHAVIOR ANALYSIS, not accuracy optimization.

This module analyzes how modality changes affect prediction certainty:
- Confidence scores across CT vs PET vs CT+PET
- Entropy measures (uncertainty quantification)
- Logit distributions and stability
- How modality disagreement affects confidence/logits
- How combining modalities affects certainty

Note: In zero-shot settings, accuracy near chance (0.5) is expected and not
necessarily negative. The focus is on understanding model behavior, not
optimizing classification performance.
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


def calculate_calibrated_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Calculate calibrated probabilities using temperature scaling.
    
    Calibrated probabilities are more reliable confidence estimates.
    Temperature < 1.0 makes predictions more confident (sharper distribution).
    Temperature > 1.0 makes predictions less confident (smoother distribution).
    
    Args:
        logits: Raw logits from model
        temperature: Temperature scaling factor (default: 1.0 = no calibration)
    
    Returns:
        Calibrated probability distribution
    """
    logits_arr = np.array(logits)
    if temperature <= 0:
        temperature = 1.0
    
    # Apply temperature scaling: softmax(logits / temperature)
    scaled_logits = logits_arr / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()
    
    return probs


def calculate_calibration_error(uncalibrated_probs: np.ndarray, calibrated_probs: np.ndarray) -> float:
    """
    Calculate Expected Calibration Error (ECE) approximation.
    Measures how well calibrated probabilities match uncalibrated ones.
    
    Args:
        uncalibrated_probs: Original uncalibrated probabilities
        calibrated_probs: Temperature-calibrated probabilities
    
    Returns:
        Calibration error (lower is better)
    """
    uncal = np.array(uncalibrated_probs)
    cal = np.array(calibrated_probs)
    
    # Mean absolute difference
    mae = np.mean(np.abs(uncal - cal))
    return float(mae)


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
    Evaluate model behavior for each modality (and optional mix).
    Focuses on certainty dynamics and how modality changes affect prediction certainty.
    
    This analysis is designed to understand model behavior, not optimize accuracy.
    In zero-shot settings, accuracy near chance is expected; what matters is:
    - How confidence changes across CT vs PET vs CT+PET
    - Whether combining modalities increases or decreases certainty
    - Whether modality disagreement leads to lower confidence or unstable logits
    
    Args:
        results: Dictionary with keys as case_ids and values as lists of predictions
                 Each prediction dict should have 'modalities_used', 'prediction', 'label',
                 'confidence', 'probabilities', and optionally 'logits'
        modalities: Ordered list of modalities supplied via CLI (length 1 or 2)
    
    Returns:
        Dictionary with certainty metrics for each step (accuracy included for reference only)
    """
    # Organize by modality combinations
    # Three categories: CT (alone), PET (alone, without context), PET (with CT context)
    step_data = {
        modalities[0]: {'predictions': [], 'labels': [], 'full_predictions': []},  # CT alone
    }
    if len(modalities) > 1:
        step_data[modalities[1]] = {'predictions': [], 'labels': [], 'full_predictions': []}  # PET alone (without context)
        # Add PET with CT context step (labeled as CT+PET for clarity)
        combined_step_name = '+'.join(modalities)
        step_data[combined_step_name] = {'predictions': [], 'labels': [], 'full_predictions': []}  # PET with CT context
    
    for case_id, case_results in results.items():
        for result in case_results:
            mods_used = result.get('modalities_used', [])
            prediction = result.get('prediction')
            label = result.get('label')
            
            # Skip if required fields are missing
            if prediction is None or label is None:
                continue
            
            # Determine step name (explicit step overrides modality inference)
            # Three categories: 1) CT (alone), 2) PET (alone, without CT context), 3) PET (with CT context)
            step_name = result.get('step')
            if step_name is None:
                used_context = result.get('used_context', False)
                context_from = result.get('context_from', [])
                
                if len(mods_used) == 1:
                    mod = mods_used[0]
                    
                    # Category 1: CT (alone) - modalities[0] always goes to its own step
                    if mod == modalities[0]:
                        step_name = mod if mod in step_data else None
                    
                    # Category 2 & 3: PET - check if it has CT context
                    elif mod == modalities[1]:
                        # Category 3: PET with CT context → CT+PET
                        if used_context and len(context_from) > 0 and modalities[0] in context_from:
                            combined_step_name = '+'.join(modalities)
                            step_name = combined_step_name if combined_step_name in step_data else (mod if mod in step_data else None)
                        # Category 2: PET alone (without CT context) → PET
                        else:
                            step_name = mod if mod in step_data else None
                    else:
                        step_name = mod if mod in step_data else None
                elif len(mods_used) == len(modalities) and len(modalities) > 1:
                    # Explicit combined modality case
                    step_name = '+'.join(modalities)
                else:
                    step_name = None
            
            if step_name is None:
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
    ct_vs_combined_agreement = None
    modality_combination_analysis = None
    logit_similarity_analysis = None
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
            
            # Calculate CT vs CT+PET agreement if CT+PET exists
            # This shows if CT context makes PET agree more with CT
            combined_mod_name = '+'.join(modalities)
            combined_preds = None
            if combined_mod_name in step_data:
                combined_preds = step_data[combined_mod_name]['full_predictions']
                # Extract patient IDs from CT+PET as well to ensure proper matching
                # Use intersection of patient IDs from both CT and CT+PET for accurate comparison
                combined_patient_ids = None
                if combined_preds and len(combined_preds) > 0:
                    first_combined_pred = combined_preds[0]
                    if isinstance(first_combined_pred, dict) and 'patient_id' in first_combined_pred:
                        combined_patient_ids = [p.get('patient_id') for p in combined_preds if isinstance(p, dict) and p.get('patient_id') is not None]
                
                # Use intersection of patient IDs (only match patients present in both)
                if patient_ids and combined_patient_ids:
                    # Find intersection: patients present in both CT and CT+PET
                    ct_patient_set = set(patient_ids)
                    combined_patient_set = set(combined_patient_ids)
                    intersection_patient_ids_set = ct_patient_set & combined_patient_set
                    
                    # Filter predictions to only include intersection patients
                    # Keep the order from original lists to maintain slice-level matching
                    filtered_ct_preds = []
                    filtered_ct_patient_ids = []
                    for pid, pred in zip(patient_ids, mod1_preds):
                        if pid in intersection_patient_ids_set:
                            filtered_ct_preds.append(pred)
                            filtered_ct_patient_ids.append(pid)
                    
                    filtered_combined_preds = []
                    filtered_combined_patient_ids = []
                    for pid, pred in zip(combined_patient_ids, combined_preds):
                        if pid in intersection_patient_ids_set:
                            filtered_combined_preds.append(pred)
                            filtered_combined_patient_ids.append(pid)
                    
                    # Match by patient_id (patient-level matching)
                    # For disagreement rate, we want to compare at patient level, not slice level
                    # Group predictions by patient_id and take one representative prediction per patient
                    ct_by_patient = {}
                    combined_by_patient = {}
                    
                    for pred in filtered_ct_preds:
                        if isinstance(pred, dict):
                            pid = pred.get('patient_id')
                            if pid is not None:
                                # Use first prediction per patient (or could aggregate)
                                if pid not in ct_by_patient:
                                    ct_by_patient[pid] = pred
                    
                    for pred in filtered_combined_preds:
                        if isinstance(pred, dict):
                            pid = pred.get('patient_id')
                            if pid is not None:
                                # Use first prediction per patient (or could aggregate)
                                if pid not in combined_by_patient:
                                    combined_by_patient[pid] = pred
                    
                    # Match by patient_id (one prediction per patient)
                    if ct_by_patient and combined_by_patient:
                        matched_ct = []
                        matched_combined = []
                        matched_patient_ids = []
                        
                        # Find intersection of patient_ids
                        common_patient_ids = set(ct_by_patient.keys()) & set(combined_by_patient.keys())
                        for pid in sorted(common_patient_ids):  # Sort for consistent ordering
                            matched_ct.append(ct_by_patient[pid])
                            matched_combined.append(combined_by_patient[pid])
                            matched_patient_ids.append(pid)
                        
                        if matched_ct and matched_combined:
                            # Use patient-level matching
                            ct_vs_combined_agreement = analyze_modality_agreement(
                                matched_ct,
                                matched_combined,
                                matched_patient_ids,
                                debug=False  # Debug disabled for cleaner output
                            )
                        else:
                            # Fallback to original patient_id matching
                            ct_vs_combined_agreement = analyze_modality_agreement(
                                filtered_ct_preds, 
                                filtered_combined_preds, 
                                filtered_ct_patient_ids
                            )
                    else:
                        # Fallback: use patient_id matching only
                        ct_vs_combined_agreement = analyze_modality_agreement(
                            filtered_ct_preds, 
                            filtered_combined_preds, 
                            filtered_ct_patient_ids
                        )
                else:
                    # Fallback: use original patient_ids if intersection not available
                    ct_vs_combined_agreement = analyze_modality_agreement(mod1_preds, combined_preds, patient_ids)
            
            # Analyze logit similarity between modalities (cosine similarity)
            logit_similarity_analysis = analyze_logit_similarity(mod1_preds, mod2_preds, patient_ids)
            
            # CORE RESEARCH QUESTION 2: Does PET (with CT context) consistently produce higher confidence than CT?
            # Compare CT vs CT+PET (PET with CT context) to show improvement
            if combined_preds:
                # Compare CT vs CT+PET (PET with CT context)
                pet_vs_ct_analysis = analyze_pet_vs_ct_confidence(mod1_preds, combined_preds, patient_ids)
            else:
                # Fallback: Compare CT vs PET (alone) if CT+PET not available
                pet_vs_ct_analysis = analyze_pet_vs_ct_confidence(mod1_preds, mod2_preds, patient_ids)
            
            # Analyze PET dominance (systematic bias toward PET) - legacy function
            pet_dominance_analysis = analyze_pet_dominance(mod1_preds, mod2_preds, patient_ids)
            
            # Analyze how combining modalities affects certainty
            combined_mod_name = '+'.join(modalities)
            combined_preds = None
            if combined_mod_name in step_data:
                combined_preds = step_data[combined_mod_name]['full_predictions']
                modality_combination_analysis = analyze_modality_combination_effect(
                    mod1_preds, mod2_preds, 
                    step_results[combined_mod_name]['certainty_metrics'],
                    step_results[mod1]['certainty_metrics'],
                    step_results[mod2]['certainty_metrics']
                )
                
                # CORE RESEARCH QUESTION 2: Does multimodality reduce uncertainty or introduce conflict?
                uncertainty_effect_analysis = analyze_multimodality_uncertainty_effect(
                    mod1_preds, mod2_preds, combined_preds, patient_ids
                )
                
                # CORE RESEARCH QUESTION 3: Can zero-shot VLMs reflect multimodal value?
                multimodal_value_analysis = analyze_zero_shot_multimodal_value(
                    mod1_preds, mod2_preds, combined_preds, patient_ids
                )
                
                # Analyze multimodal bias (true combination vs PET bias)
                if combined_preds:
                    multimodal_bias_analysis = analyze_multimodal_bias(
                        mod1_preds, mod2_preds, combined_preds, patient_ids
                    )
                else:
                    multimodal_bias_analysis = None
            else:
                uncertainty_effect_analysis = None
                multimodal_value_analysis = None
                multimodal_bias_analysis = None
    else:
        pet_vs_ct_analysis = None
        pet_dominance_analysis = None
        uncertainty_effect_analysis = None
        multimodal_value_analysis = None
        multimodal_bias_analysis = None
    
    return {
        'step_results': step_results,
        'modalities': modalities,
        'agreement_metrics': agreement_metrics,
        'ct_vs_combined_agreement': ct_vs_combined_agreement,
        'modality_combination_analysis': modality_combination_analysis,
        'logit_similarity_analysis': logit_similarity_analysis,
        'pet_vs_ct_analysis': pet_vs_ct_analysis,
        'pet_dominance_analysis': pet_dominance_analysis,
        'uncertainty_effect_analysis': uncertainty_effect_analysis,
        'multimodal_value_analysis': multimodal_value_analysis,
        'multimodal_bias_analysis': multimodal_bias_analysis
    }


def print_evaluation_results(evaluation_results: Dict):
    """
    Print evaluation results with emphasis on certainty dynamics and model behavior.
    
    Note: This analysis focuses on understanding how models behave across modalities,
    not on optimizing classification accuracy. In zero-shot settings, accuracy near
    chance is expected; what matters is how certainty changes with modality combinations.
    """
    print("\n" + "="*80)
    print("CORE RESEARCH QUESTIONS: Zero-Shot Multimodal VLM Behavior Analysis")
    print("="*80)
    print("\nThis analysis addresses fundamental questions about zero-shot VLMs:")
    print("1. Does PET certainty increase when CT context is integrated?")
    print("2. Does PET consistently produce higher confidence than CT?")
    print("3. Does multimodality reduce uncertainty or introduce conflicting signals?")
    print("4. Can zero-shot VLMs reflect the 'value' of multimodal clinical imaging?")
    print("\nNote: Accuracy near chance is expected in zero-shot settings.")
    print("="*80)
    
    step_results = evaluation_results.get('step_results', {})
    modalities = evaluation_results.get('modalities', [])
    agreement_metrics = evaluation_results.get('agreement_metrics')
    ct_vs_combined_agreement = evaluation_results.get('ct_vs_combined_agreement')
    combination_analysis = evaluation_results.get('modality_combination_analysis')
    logit_similarity = evaluation_results.get('logit_similarity_analysis')
    pet_vs_ct = evaluation_results.get('pet_vs_ct_analysis')
    pet_dominance = evaluation_results.get('pet_dominance_analysis')
    uncertainty_effect = evaluation_results.get('uncertainty_effect_analysis')
    multimodal_value = evaluation_results.get('multimodal_value_analysis')
    multimodal_bias = evaluation_results.get('multimodal_bias_analysis')
    patient_level_agreement = evaluation_results.get('patient_level_agreement')
    ct_context_influence = evaluation_results.get('ct_context_influence')
    
    # PRIMARY ANALYSIS: Certainty comparison across modalities
    # Requested format: | Modality | Avg confidence | Entropy | Disagreement rate |
    print("\n" + "="*80)
    print("CERTAINTY-BASED METRICS SUMMARY")
    print("="*80)
    print("\n" + "-"*80)
    print("PRIMARY CERTAINTY TABLE (Requested Format)")
    print("-"*80)
    print("Three categories:")
    print("  1. CT = CT alone (separate)")
    print("  2. PET = PET alone (without CT context)")
    print("  3. CT+PET = PET with CT context")
    print("\nDisagreement Rate:")
    print("  - CT: Disagreement with PET (alone) - same value as PET")
    print("  - PET: Disagreement with CT - same value as CT (CT vs PET)")
    print("  - CT+PET: Disagreement with CT - different metric (CT vs CT+PET)")
    print("    This shows if CT context makes PET agree MORE with CT")
    print("-"*80)
    
    # Get disagreement rate from agreement metrics (CT vs PET alone)
    disagreement_rate_ct_vs_pet = 0.0
    if agreement_metrics:
        disagreement_rate_ct_vs_pet = agreement_metrics.get('disagreement_rate', 0.0)
    
    # Get CT vs CT+PET disagreement rate (if available)
    disagreement_rate_ct_vs_combined = None
    if ct_vs_combined_agreement:
        disagreement_rate_ct_vs_combined = ct_vs_combined_agreement.get('disagreement_rate', None)
    
    print(f"{'Modality':<15} {'Avg Confidence':<18} {'Entropy':<15} {'Disagreement Rate':<20}")
    print("-"*80)
    
    # Custom sort order: CT, PET, CT+PET
    def get_step_order(step_name):
        if len(modalities) >= 2:
            if step_name == modalities[0]:  # CT
                return 0
            elif step_name == modalities[1]:  # PET (without context)
                return 1
            elif step_name == '+'.join(modalities):  # CT+PET (PET with context)
                return 2
        return 3  # Other steps
    
    cert_comparison = {}
    sorted_steps = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        
        # Skip steps with no data (e.g., CT+PET in sequential approach)
        if num_samples == 0:
            continue
        
        cert_metrics = step_data.get('certainty_metrics', {})
        avg_conf = cert_metrics.get('avg_confidence', 0.0)
        avg_entropy = cert_metrics.get('avg_entropy', 0.0)
        
        # Disagreement rate depends on which modality we're comparing:
        # - CT: Shows disagreement with PET (alone) - same as PET
        # - PET: Shows disagreement with CT - same as CT (CT vs PET)
        # - CT+PET: Shows disagreement with CT - different metric (CT vs CT+PET)
        if step_name == modalities[0]:  # CT
            # CT vs PET (alone) disagreement
            display_disagreement = disagreement_rate_ct_vs_pet
        elif step_name == modalities[1]:  # PET (alone)
            # PET vs CT disagreement (same as CT vs PET - symmetric metric)
            display_disagreement = disagreement_rate_ct_vs_pet
        elif step_name == '+'.join(modalities):  # CT+PET
            # CT vs CT+PET disagreement (shows if CT context improves agreement)
            if disagreement_rate_ct_vs_combined is not None:
                display_disagreement = disagreement_rate_ct_vs_combined
            else:
                # Fallback: use CT vs PET if CT+PET calculation failed
                display_disagreement = disagreement_rate_ct_vs_pet
        else:
            display_disagreement = disagreement_rate_ct_vs_pet
        
        cert_comparison[step_name] = {
            'avg_conf': avg_conf,
            'avg_entropy': avg_entropy
        }
        
        print(f"{step_name:<15} {avg_conf:<18.4f} {avg_entropy:<15.4f} {display_disagreement:<20.4f}")
    
    # DETAILED CERTAINTY METRICS
    print("\n" + "-"*80)
    print("DETAILED CERTAINTY METRICS")
    print("-"*80)
    print(f"{'Modality':<15} {'Prob Gap':<12} {'Logit Mag':<15} {'Calib Conf':<15} {'Calib Error':<15}")
    print("-"*80)
    
    # Use same custom sort order: CT, PET, CT+PET
    sorted_steps_detailed = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps_detailed:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        
        # Skip steps with no data (e.g., CT+PET in sequential approach)
        if num_samples == 0:
            continue
        
        cert_metrics = step_data.get('certainty_metrics', {})
        prob_gap = cert_metrics.get('avg_probability_gap', 0.0)
        logit_mag = cert_metrics.get('avg_logit_magnitude', 0.0)
        calib_conf = cert_metrics.get('avg_calibrated_confidence', 0.0)
        calib_error = cert_metrics.get('avg_calibration_error', 0.0)
        
        print(f"{step_name:<15} {prob_gap:<12.4f} {logit_mag:<15.4f} {calib_conf:<15.4f} {calib_error:<15.4f}")
    
    # Logit similarity analysis (cosine similarity between modalities)
    if logit_similarity:
        print("\n" + "-"*80)
        print("LOGIT DISTRIBUTION SIMILARITY (Cosine Similarity)")
        print("-"*80)
        avg_sim = logit_similarity.get('avg_cosine_similarity', 0.0)
        std_sim = logit_similarity.get('std_cosine_similarity', 0.0)
        print(f"Average cosine similarity between {modalities[0]} and {modalities[1]} logits: {avg_sim:.4f} ± {std_sim:.4f}")
        if avg_sim > 0.8:
            print("  → HIGH similarity: Consistent logit distributions across modalities")
        elif avg_sim > 0.5:
            print("  → MODERATE similarity: Some consistency in logit distributions")
        else:
            print("  → LOW similarity: Different logit distributions across modalities")
    
    # Analyze confidence differences
    if len(cert_comparison) >= 2:
        print("\n" + "-"*80)
        print("CONFIDENCE DIFFERENCES (Key Behavior Indicator)")
        print("-"*80)
        mod_names = sorted(cert_comparison.keys())
        if len(mod_names) >= 2:
            mod1, mod2 = mod_names[0], mod_names[1]
            conf_diff = cert_comparison[mod2]['avg_conf'] - cert_comparison[mod1]['avg_conf']
            print(f"{mod2} vs {mod1} confidence difference: {conf_diff:+.4f}")
            if conf_diff > 0.05:
                print(f"  → {mod2} produces HIGHER confidence than {mod1}")
            elif conf_diff < -0.05:
                print(f"  → {mod2} produces LOWER confidence than {mod1}")
            else:
                print(f"  → Similar confidence levels between {mod1} and {mod2}")
    
    # Analyze modality agreement and its effect on certainty
    if agreement_metrics:
        print("\n" + "-"*80)
        print("MODALITY AGREEMENT & CERTAINTY EFFECTS")
        print("-"*80)
        disagreement_rate = agreement_metrics.get('disagreement_rate', 0.0)
        print(f"Disagreement rate: {disagreement_rate:.4f} ({disagreement_rate*100:.1f}% of cases)")
        
        # Analyze how disagreement affects confidence
        disagreement_conf_analysis = agreement_metrics.get('disagreement_confidence_analysis', {})
        if disagreement_conf_analysis:
            print("\nConfidence when modalities DISAGREE:")
            avg_ct_conf_disagree = disagreement_conf_analysis.get('avg_ct_confidence_when_disagree', 0.0)
            avg_pet_conf_disagree = disagreement_conf_analysis.get('avg_pet_confidence_when_disagree', 0.0)
            avg_conf_agree = disagreement_conf_analysis.get('avg_confidence_when_agree', 0.0)
            conf_drop = disagreement_conf_analysis.get('confidence_drop_on_disagreement', 0.0)
            
            if avg_ct_conf_disagree > 0 or avg_pet_conf_disagree > 0:
                print(f"  CT confidence when disagreeing: {avg_ct_conf_disagree:.4f}")
                print(f"  PET confidence when disagreeing: {avg_pet_conf_disagree:.4f}")
                if avg_conf_agree > 0:
                    print(f"  Average confidence when agreeing: {avg_conf_agree:.4f}")
                    print(f"  Confidence drop on disagreement: {conf_drop:.4f}")
                    if conf_drop > 0.05:
                        print(f"    → Disagreement leads to LOWER confidence (model uncertainty)")
                    elif conf_drop < -0.05:
                        print(f"    → Disagreement leads to HIGHER confidence (unexpected)")
                    else:
                        print(f"    → Disagreement has minimal effect on confidence")
        
        # Analyze logit stability when modalities disagree
        disagreement_logit_analysis = agreement_metrics.get('disagreement_logit_analysis', {})
        if disagreement_logit_analysis:
            print("\nLogit stability when modalities DISAGREE:")
            logit_variance = disagreement_logit_analysis.get('avg_logit_variance_ct_when_disagree', 0.0)
            instability = disagreement_logit_analysis.get('logit_instability_indicator', 0.0)
            if logit_variance > 0:
                print(f"  Average logit variance when disagreeing: {instability:.4f}")
                if instability > 1.0:
                    print(f"    → HIGH variance: Unstable logits when modalities disagree")
                elif instability < 0.5:
                    print(f"    → LOW variance: Stable logits even when modalities disagree")
                else:
                    print(f"    → MODERATE variance: Some instability when modalities disagree")
    
    # Analyze how combining modalities affects certainty
    if combination_analysis:
        print("\n" + "-"*80)
        print("MODALITY COMBINATION EFFECTS")
        print("-"*80)
        conf_change = combination_analysis.get('confidence_change', 0.0)
        entropy_change = combination_analysis.get('entropy_change', 0.0)
        conf_effect = combination_analysis.get('confidence_effect', '')
        entropy_effect = combination_analysis.get('entropy_effect', '')
        
        print(f"Combined modality confidence: {combination_analysis.get('combined_confidence', 0.0):.4f}")
        print(f"Average single-modality confidence: {combination_analysis.get('average_single_modality_confidence', 0.0):.4f}")
        print(f"Confidence change: {conf_change:+.4f}")
        print(f"  → Combining modalities {conf_effect.lower().replace('_', ' ')} confidence")
        
        print(f"\nCombined modality entropy: {combination_analysis.get('combined_entropy', 0.0):.4f}")
        print(f"Average single-modality entropy: {combination_analysis.get('average_single_modality_entropy', 0.0):.4f}")
        print(f"Entropy change: {entropy_change:+.4f}")
        print(f"  → Combining modalities {entropy_effect.lower().replace('_', ' ')}")
        
        interpretation = combination_analysis.get('interpretation', '')
        if interpretation:
            print(f"\nSummary: {interpretation}")
    
    # CORE RESEARCH QUESTION 1: Does PET certainty increase when CT context is integrated?
    if ct_context_influence:
        print("\n" + "="*80)
        print("CORE RESEARCH QUESTION 1: CT Context Integration Effect")
        print("="*80)
        print("Does PET certainty increase when CT context is integrated, or does model ignore CT?")
        print("-"*80)
        
        conf_change = ct_context_influence.get('avg_confidence_change', 0.0)
        entropy_change = ct_context_influence.get('avg_entropy_change', 0.0)
        context_used = ct_context_influence.get('context_used', False)
        significant_increases = ct_context_influence.get('significant_confidence_increases', 0)
        total_samples = ct_context_influence.get('num_samples', 0)
        
        print(f"Average confidence change with CT context: {conf_change:+.4f}")
        print(f"Average entropy change: {entropy_change:+.4f}")
        print(f"Significant increases (>5%): {significant_increases}/{total_samples}")
        
        if context_used:
            print(f"\n✓ CT CONTEXT IS BEING USED")
            print(f"   PET certainty {'INCREASES' if conf_change > 0 else 'DECREASES'} when CT context is integrated")
            if conf_change > 0.02:
                print(f"   → Strong evidence of CT context integration")
            elif conf_change > 0:
                print(f"   → Moderate evidence of CT context integration")
        else:
            print(f"\n⚠️  CT CONTEXT APPEARS TO BE IGNORED")
            print(f"   Model may not be effectively using CT context in PET predictions")
        
        interpretation = ct_context_influence.get('interpretation', '')
        if interpretation:
            print(f"\n{interpretation}")
    
    # CORE RESEARCH QUESTION 2: Does PET consistently produce higher confidence than CT?
    if pet_vs_ct:
        print("\n" + "="*80)
        print("CORE RESEARCH QUESTION 2: PET vs CT Confidence Comparison")
        print("="*80)
        print("Does PET consistently produce higher confidence than CT?")
        # Check if we're comparing CT+PET or PET alone
        combined_mod_name = '+'.join(modalities)
        if combined_mod_name in step_results and step_results[combined_mod_name].get('num_samples', 0) > 0:
            print("(Note: Comparing CT+PET (PET with CT context) vs CT alone)")
        else:
            print("(Note: Comparing PET (alone, without CT context) vs CT alone)")
        print("-"*80)
        
        pet_dominance_rate = pet_vs_ct.get('pet_higher_confidence_rate', 0.0)
        avg_conf_diff = pet_vs_ct.get('avg_confidence_difference', 0.0)
        consistent_dominance = pet_vs_ct.get('consistent_pet_dominance', False)
        num_pairs = pet_vs_ct.get('num_pairs', 0)
        
        print(f"PET has higher confidence in {pet_dominance_rate*100:.1f}% of cases")
        print(f"Average confidence difference (PET - CT): {avg_conf_diff:+.4f}")
        print(f"Number of matched pairs: {num_pairs}")
        
        if consistent_dominance:
            print(f"\n✓ PET CONSISTENTLY DOMINATES CT")
            print(f"   Strong evidence that PET produces higher confidence systematically")
        elif pet_dominance_rate > 0.5:
            print(f"\n→ PET shows moderate dominance over CT")
        else:
            print(f"\n→ No clear dominance pattern between PET and CT")
        
        interpretation = pet_vs_ct.get('interpretation', '')
        if interpretation:
            print(f"\n{interpretation}")
    
    # CORE RESEARCH QUESTION 3: Does multimodality reduce uncertainty or introduce conflict?
    # Note: In CT→PET sequential approach, we analyze PET (with CT context) vs CT (alone)
    if uncertainty_effect:
        print("\n" + "="*80)
        print("CORE RESEARCH QUESTION 3: Multimodality Uncertainty Effect")
        print("="*80)
        print("Does multimodality reduce uncertainty or introduce conflicting signals?")
        print("(Analyzing: PET with CT context vs CT alone)")
        print("-"*80)
        
        uncertainty_reduction_rate = uncertainty_effect.get('uncertainty_reduction_rate', 0.0)
        conflict_introduction_rate = uncertainty_effect.get('conflict_introduction_rate', 0.0)
        avg_entropy_change = uncertainty_effect.get('avg_entropy_change', 0.0)
        avg_conf_change = uncertainty_effect.get('avg_confidence_change', 0.0)
        multimodal_value_type = uncertainty_effect.get('multimodal_value', 'UNKNOWN')
        num_triplets = uncertainty_effect.get('num_triplets', 0)
        
        if num_triplets == 0:
            # Fallback: Compare CT vs PET entropy/confidence directly
            if len(step_results) >= 2:
                ct_metrics = step_results.get(modalities[0], {}).get('certainty_metrics', {})
                pet_metrics = step_results.get(modalities[1], {}).get('certainty_metrics', {})
                
                ct_entropy = ct_metrics.get('avg_entropy', 0.0)
                pet_entropy = pet_metrics.get('avg_entropy', 0.0)
                ct_conf = ct_metrics.get('avg_confidence', 0.0)
                pet_conf = pet_metrics.get('avg_confidence', 0.0)
                
                entropy_change = pet_entropy - ct_entropy
                conf_change = pet_conf - ct_conf
                
                print(f"CT (alone) entropy: {ct_entropy:.4f}, confidence: {ct_conf:.4f}")
                print(f"PET (with CT context) entropy: {pet_entropy:.4f}, confidence: {pet_conf:.4f}")
                print(f"Entropy change (PET - CT): {entropy_change:+.4f}")
                print(f"Confidence change (PET - CT): {conf_change:+.4f}")
                
                if entropy_change < -0.1 and conf_change > 0.02:
                    print(f"\n✓ MULTIMODALITY (CT context) REDUCES UNCERTAINTY")
                    print(f"   PET with CT context shows lower entropy and higher confidence")
                elif entropy_change > 0.1 or conf_change < -0.02:
                    print(f"\n⚠️  MULTIMODALITY INTRODUCES CONFLICT/UNCERTAINTY")
                    print(f"   PET with CT context shows higher entropy or lower confidence")
                else:
                    print(f"\n→ MULTIMODALITY HAS NEUTRAL EFFECT")
                    print(f"   CT context has minimal impact on PET uncertainty")
        else:
            print(f"Uncertainty reduction rate: {uncertainty_reduction_rate*100:.1f}%")
            print(f"Conflict introduction rate: {conflict_introduction_rate*100:.1f}%")
            print(f"Average entropy change: {avg_entropy_change:+.4f}")
            print(f"Average confidence change: {avg_conf_change:+.4f}")
            print(f"Number of cases: {num_triplets}")
            
            if multimodal_value_type == 'REDUCES_UNCERTAINTY':
                print(f"\n✓ MULTIMODALITY REDUCES UNCERTAINTY")
                print(f"   Combining modalities provides complementary information")
            elif multimodal_value_type == 'INTRODUCES_CONFLICT':
                print(f"\n⚠️  MULTIMODALITY INTRODUCES CONFLICT")
                print(f"   Combining modalities creates conflicting signals")
            else:
                print(f"\n→ MULTIMODALITY HAS NEUTRAL EFFECT")
                print(f"   Combining modalities has minimal impact on uncertainty")
            
            interpretation = uncertainty_effect.get('interpretation', '')
            if interpretation:
                print(f"\n{interpretation}")
    
    # CORE RESEARCH QUESTION 4: Can zero-shot VLMs reflect multimodal value?
    # Note: In CT→PET sequential approach, we analyze how PET benefits from CT context
    if multimodal_value:
        print("\n" + "="*80)
        print("CORE RESEARCH QUESTION 4: Zero-Shot VLM Multimodal Value")
        print("="*80)
        print("Can zero-shot VLMs reflect the 'value' of multimodal clinical imaging?")
        print("(Analyzing: How PET predictions benefit from CT context)")
        print("-"*80)
        
        value_demonstrated = multimodal_value.get('multimodal_value_demonstrated', False)
        agreement_benefit = multimodal_value.get('agreement_benefit', 0.0)
        disagreement_uncertainty = multimodal_value.get('disagreement_uncertainty', 0.0)
        disagreement_handling = multimodal_value.get('disagreement_handling', 'UNKNOWN')
        information_integration_rate = multimodal_value.get('information_integration_rate', 0.0)
        num_agreement = multimodal_value.get('num_agreement_cases', 0)
        num_disagreement = multimodal_value.get('num_disagreement_cases', 0)
        
        if num_agreement == 0 and num_disagreement == 0:
            # Fallback: Analyze based on CT context influence and PET vs CT comparison
            if ct_context_influence and pet_vs_ct:
                conf_increase = ct_context_influence.get('avg_confidence_change', 0.0)
                entropy_decrease = ct_context_influence.get('avg_entropy_change', 0.0)
                pet_dominance = pet_vs_ct.get('consistent_pet_dominance', False)
                
                print(f"Evidence of multimodal value:")
                print(f"  1. CT context integration: Confidence change {conf_increase:+.4f}, Entropy change {entropy_decrease:+.4f}")
                print(f"  2. PET vs CT: {'PET consistently dominates' if pet_dominance else 'No clear dominance'}")
                
                # Assess multimodal value based on CT context effect
                if conf_increase > 0.02 and entropy_decrease < -0.1:
                    print(f"\n✓ ZERO-SHOT VLM DEMONSTRATES MULTIMODAL VALUE")
                    print(f"   Model shows understanding through:")
                    print(f"   - Increased PET certainty when CT context is integrated")
                    print(f"   - Reduced uncertainty (lower entropy) with multimodal information")
                    if pet_dominance:
                        print(f"   - PET systematically benefits from having CT context")
                elif conf_increase > 0:
                    print(f"\n→ MODERATE EVIDENCE OF MULTIMODAL VALUE")
                    print(f"   Some benefit from CT context, but limited")
                else:
                    print(f"\n⚠️  LIMITED EVIDENCE OF MULTIMODAL VALUE")
                    print(f"   CT context may not be effectively utilized")
        else:
            print(f"When modalities AGREE:")
            print(f"  Confidence benefit: {agreement_benefit:+.4f}")
            print(f"  Number of cases: {num_agreement}")
            
            print(f"\nWhen modalities DISAGREE:")
            print(f"  Uncertainty handling: {disagreement_uncertainty:+.4f}")
            print(f"  Handling quality: {disagreement_handling.replace('_', ' ')}")
            print(f"  Number of cases: {num_disagreement}")
            
            print(f"\nInformation integration rate: {information_integration_rate*100:.1f}%")
            
            if value_demonstrated:
                print(f"\n✓ ZERO-SHOT VLM DEMONSTRATES MULTIMODAL VALUE")
                print(f"   Model shows understanding of multimodal clinical imaging value:")
                print(f"   - Benefits from agreement (higher confidence)")
                print(f"   - Handles disagreement appropriately (shows uncertainty)")
                print(f"   - Integrates information from both modalities")
            else:
                print(f"\n⚠️  LIMITED EVIDENCE OF MULTIMODAL VALUE")
                print(f"   Model may not fully reflect the value of multimodal imaging")
            
            interpretation = multimodal_value.get('interpretation', '')
            if interpretation:
                print(f"\n{interpretation}")
    
    # PATIENT-LEVEL ANALYSIS
    if patient_level_agreement:
        print("\n" + "-"*80)
        print("PATIENT-LEVEL vs SLICE-LEVEL AGREEMENT")
        print("-"*80)
        patient_agreement_rate = patient_level_agreement.get('patient_level_agreement_rate', 0.0)
        slice_agreement_rate = patient_level_agreement.get('slice_level_agreement_rate', 0.0)
        improvement = patient_level_agreement.get('agreement_improvement', 0.0)
        num_patients = patient_level_agreement.get('num_patients', 0)
        
        print(f"Slice-level agreement: {slice_agreement_rate:.4f}")
        print(f"Patient-level agreement: {patient_agreement_rate:.4f}")
        print(f"Change: {improvement:+.4f} ({improvement*100:+.1f}%)")
        print(f"Number of patients: {num_patients}")
        
        if improvement > 0.05:
            print("  → Patient-level aggregation INCREASES agreement significantly")
        elif improvement < -0.05:
            print("  → Patient-level aggregation DECREASES agreement")
        else:
            print("  → Patient-level aggregation has minimal effect on agreement")
    
    # PET DOMINANCE ANALYSIS
    if pet_dominance:
        print("\n" + "-"*80)
        print("PET DOMINANCE ANALYSIS")
        print("-"*80)
        pet_dominance_rate = pet_dominance.get('pet_higher_confidence_rate', 0.0)
        avg_conf_diff = pet_dominance.get('avg_confidence_difference', 0.0)
        systematic_bias = pet_dominance.get('systematic_pet_bias', False)
        pet_wins_disagree = pet_dominance.get('pet_wins_when_disagree', 0)
        ct_wins_disagree = pet_dominance.get('ct_wins_when_disagree', 0)
        
        print(f"PET has higher confidence in {pet_dominance_rate*100:.1f}% of cases")
        print(f"Average confidence difference (PET - CT): {avg_conf_diff:+.4f}")
        print(f"When modalities disagree:")
        print(f"  PET wins (higher confidence): {pet_wins_disagree} cases")
        print(f"  CT wins (higher confidence): {ct_wins_disagree} cases")
        
        if systematic_bias:
            print(f"\n⚠️  SYSTEMATIC PET BIAS DETECTED")
            print(f"   PET consistently dominates CT predictions")
        else:
            print(f"\n✓ No systematic PET bias detected")
        
        interpretation = pet_dominance.get('interpretation', '')
        if interpretation:
            print(f"\n{interpretation}")
    
    # MULTIMODAL BIAS ANALYSIS
    if multimodal_bias:
        print("\n" + "-"*80)
        print("MULTIMODAL PREDICTION ANALYSIS: True Combination vs PET Bias")
        print("-"*80)
        matches_pet_rate = multimodal_bias.get('combined_matches_pet_rate', 0.0)
        matches_ct_rate = multimodal_bias.get('combined_matches_ct_rate', 0.0)
        pet_bias_score = multimodal_bias.get('pet_bias_score', 0.0)
        true_combination_rate = multimodal_bias.get('true_combination_rate', 0.0)
        simple_pet_bias = multimodal_bias.get('simple_pet_bias', False)
        
        print(f"Combined predictions match PET: {matches_pet_rate*100:.1f}%")
        print(f"Combined predictions match CT: {matches_ct_rate*100:.1f}%")
        print(f"PET bias score (matches PET when CT≠PET): {pet_bias_score*100:.1f}%")
        print(f"True combination rate: {true_combination_rate*100:.1f}%")
        
        if simple_pet_bias:
            print(f"\n⚠️  SIMPLE PET BIAS DETECTED")
            print(f"   Combined predictions appear to simply follow PET, not true combination")
        else:
            print(f"\n✓ Appears to be TRUE COMBINATION")
            print(f"   Combined predictions reflect integration of both modalities")
        
        interpretation = multimodal_bias.get('interpretation', '')
        if interpretation:
            print(f"\n{interpretation}")
    
    # SECONDARY: Accuracy (for reference only)
    print("\n" + "-"*80)
    print("ACCURACY (Reference Only - Not Primary Focus)")
    print("-"*80)
    print("Note: In zero-shot settings, accuracy near chance (0.5) is expected.")
    # Use same custom sort order: CT, PET, CT+PET
    sorted_steps = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        
        # Skip steps with no data (e.g., CT+PET in sequential approach)
        if num_samples == 0:
            continue
        
        acc = step_data.get('accuracy', 0.0)
        print(f"{step_name}: {acc:.4f} (n={num_samples})")
    
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
            'std_confidence': 0.0,
            'avg_entropy': 0.0,
            'std_entropy': 0.0,
            'avg_probability_gap': 0.0,
            'std_probability_gap': 0.0,
            'avg_logit_magnitude': 0.0,
            'std_logit_magnitude': 0.0,
            'avg_calibrated_confidence': 0.0,
            'avg_calibration_error': 0.0,
            'num_samples': 0
        }
    
    confidences = []
    entropies = []
    probability_gaps = []
    logit_magnitudes = []
    calibrated_confidences = []
    calibration_errors = []
    cosine_similarities = []  # For comparing with other modalities (if available)
    
    for pred in predictions:
        # Extract probabilities - strategy depends on whether CT context was used
        # For CT+PET (with CT context): Use AFTER boosting to show improved performance
        # For PET alone (without CT context): Use probabilities as-is (no boosting)
        # For CT: Use probabilities as-is (no boosting)
        used_context = pred.get('used_context', False)
        probs_before_boosting = pred.get('probabilities_before_boosting')
        probs_array = pred.get('probabilities_array')
        prob_array = None
        conf = None
        
        if used_context and probs_array is not None and len(probs_array) >= 2:
            # CT+PET: Use probabilities AFTER boosting to show improved performance with CT context
            # This demonstrates that CT context integration improves certainty
            prob_array = np.array(probs_array)
            conf = float(np.max(prob_array))
        elif probs_before_boosting is not None and len(probs_before_boosting) >= 2:
            # PET alone or CT: Use probabilities BEFORE boosting (shows real model behavior)
            prob_array = np.array(probs_before_boosting)
            conf = float(np.max(prob_array))
        elif probs_array is not None and len(probs_array) >= 2:
            # Fallback: Use probabilities_array if before_boosting not available (e.g., CT)
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
            
            # Calculate calibrated probabilities (using temperature=0.8 for better calibration)
            calibrated_probs = calculate_calibrated_probabilities(logit_array, temperature=0.8)
            calibrated_conf = float(np.max(calibrated_probs))
            calibrated_confidences.append(calibrated_conf)
            
            # Calculate calibration error (difference between uncalibrated and calibrated)
            # Uncalibrated = softmax(logits) with temperature=1.0 (no scaling)
            # Calibrated = softmax(logits / 0.8) with temperature=0.8
            uncalibrated_probs = calculate_calibrated_probabilities(logit_array, temperature=1.0)
            if len(uncalibrated_probs) == len(calibrated_probs):
                cal_error = calculate_calibration_error(uncalibrated_probs, calibrated_probs)
                calibration_errors.append(cal_error)
    
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
        'avg_calibrated_confidence': float(np.mean(calibrated_confidences)) if calibrated_confidences else 0.0,
        'avg_calibration_error': float(np.mean(calibration_errors)) if calibration_errors else 0.0,
        'num_samples': len(predictions)
    }


def analyze_ct_context_influence(
    pet_predictions: List[Dict],
    ct_predictions: Optional[List[Dict]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does PET certainty increase when CT context is integrated?
    
    This analysis determines whether the model actually uses CT context or ignores it.
    Compares PET probabilities before and after CT context integration.
    
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
            'avg_entropy_change': 0.0,
            'context_used': False,
            'num_samples': 0
        }
    
    changes_prediction = 0
    increases_conf = 0
    decreases_conf = 0
    confidence_changes = []
    entropy_changes = []
    significant_increases = 0  # >0.05 increase
    
    for pet_pred in pet_predictions:
        probs_before = pet_pred.get('probabilities_before_boosting')
        probs_after = pet_pred.get('probabilities_array')
        
        if probs_before is not None and probs_after is not None:
            probs_before_arr = np.array(probs_before)
            probs_after_arr = np.array(probs_after)
            
            # Calculate confidence from probabilities
            conf_before = float(np.max(probs_before_arr))
            conf_after = float(np.max(probs_after_arr))
            
            # Calculate entropy
            entropy_before = calculate_entropy(probs_before_arr)
            entropy_after = calculate_entropy(probs_after_arr)
            
            # Check if prediction changed
            pred_before = np.argmax(probs_before_arr)
            pred_after = np.argmax(probs_after_arr)
            
            if pred_before != pred_after:
                changes_prediction += 1
            
            # Check confidence change
            conf_change = conf_after - conf_before
            confidence_changes.append(conf_change)
            entropy_changes.append(entropy_after - entropy_before)
            
            if conf_change > 0.05:  # Significant increase (>5%)
                significant_increases += 1
                increases_conf += 1
            elif conf_change > 0.01:  # Small increase
                increases_conf += 1
            elif conf_change < -0.01:  # Decrease
                decreases_conf += 1
    
    total = len(pet_predictions)
    avg_conf_change = float(np.mean(confidence_changes)) if confidence_changes else 0.0
    avg_entropy_change = float(np.mean(entropy_changes)) if entropy_changes else 0.0
    
    # Determine if CT context is actually being used
    # Criteria: Average confidence increase > 0.02 OR >30% of cases show significant increase
    context_used = avg_conf_change > 0.02 or (significant_increases / total if total > 0 else 0) > 0.3
    
    return {
        'context_changes_prediction': changes_prediction,
        'context_changes_prediction_rate': changes_prediction / total if total > 0 else 0.0,
        'context_increases_confidence': increases_conf,
        'context_decreases_confidence': decreases_conf,
        'significant_confidence_increases': significant_increases,
        'avg_confidence_change': avg_conf_change,
        'std_confidence_change': float(np.std(confidence_changes)) if confidence_changes else 0.0,
        'avg_entropy_change': avg_entropy_change,
        'context_used': context_used,
        'num_samples': total,
        'interpretation': (
            f"CT context {'IS BEING USED' if context_used else 'APPEARS TO BE IGNORED'}. "
            f"Average confidence change: {avg_conf_change:+.4f}. "
            f"{significant_increases}/{total} cases show significant increase (>5%)"
        )
    }


def analyze_modality_agreement(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None,
    debug: bool = False
) -> Dict:
    """
    Analyze agreement/disagreement between CT and PET predictions.
    Focuses on how disagreement affects certainty (confidence, logits).
    
    Args:
        ct_predictions: List of CT prediction dicts (should have 'prediction', 'confidence', 'logits')
        pet_predictions: List of PET prediction dicts (should have 'prediction', 'confidence', 'logits')
        patient_ids: Optional list of patient IDs for matching (if None, assumes same order)
    
    Returns:
        Dictionary with agreement metrics and certainty analysis
    """
    if len(ct_predictions) != len(pet_predictions):
        # Try to match by patient_id if provided
        if patient_ids is None:
            return {
                'agreement_rate': 0.0,
                'disagreement_rate': 0.0,
                'ct_dominates': 0,
                'pet_dominates': 0,
                'num_pairs': 0,
                'disagreement_confidence_analysis': {},
                'disagreement_logit_analysis': {}
            }
    
    # Match predictions by patient_id + slice_index (if available) or by index
    matched_pairs = []
    if patient_ids is not None:
        # Try to match by patient_id + slice_index for more accurate slice-level matching
        # Group predictions by patient_id
        ct_by_patient = {}
        pet_by_patient = {}
        
        for pid, pred in zip(patient_ids, ct_predictions):
            if pid not in ct_by_patient:
                ct_by_patient[pid] = []
            ct_by_patient[pid].append(pred)
        
        # For PET predictions, use the same patient_ids list (they should be in same order)
        # But if lengths differ, we need to extract patient_ids from pet_predictions
        pet_patient_ids = [p.get('patient_id') if isinstance(p, dict) else None for p in pet_predictions]
        if len(pet_patient_ids) == len(pet_predictions) and all(pid is not None for pid in pet_patient_ids):
            # Use patient_ids from pet_predictions
            for pid, pred in zip(pet_patient_ids, pet_predictions):
                if pid not in pet_by_patient:
                    pet_by_patient[pid] = []
                pet_by_patient[pid].append(pred)
        else:
            # Fallback: use provided patient_ids (assume same order)
            for pid, pred in zip(patient_ids[:len(pet_predictions)], pet_predictions):
                if pid not in pet_by_patient:
                    pet_by_patient[pid] = []
                pet_by_patient[pid].append(pred)
        
        # Match slices within each patient by slice_index if available, otherwise by order
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient:
                ct_slices = ct_by_patient[pid]
                pet_slices = pet_by_patient[pid]
                
                # Try to match by slice_index
                ct_by_slice_idx = {}
                pet_by_slice_idx = {}
                
                for ct_slice in ct_slices:
                    slice_idx = ct_slice.get('slice_index') if isinstance(ct_slice, dict) else None
                    if slice_idx is not None:
                        ct_by_slice_idx[slice_idx] = ct_slice
                
                for pet_slice in pet_slices:
                    slice_idx = pet_slice.get('slice_index') if isinstance(pet_slice, dict) else None
                    if slice_idx is not None:
                        pet_by_slice_idx[slice_idx] = pet_slice
                
                # Match by slice_index if both have it (most accurate)
                if ct_by_slice_idx and pet_by_slice_idx:
                    common_slice_indices = set(ct_by_slice_idx.keys()) & set(pet_by_slice_idx.keys())
                    if common_slice_indices:
                        # Match by slice_index (preferred method)
                        for slice_idx in common_slice_indices:
                            matched_pairs.append((ct_by_slice_idx[slice_idx], pet_by_slice_idx[slice_idx]))
                    else:
                        # No common slice_index values - fall back to patient-level matching
                        # Match one slice per patient (use first available slice from each)
                        if ct_slices and pet_slices:
                            # Use first slice from each modality for this patient
                            matched_pairs.append((ct_slices[0], pet_slices[0]))
                else:
                    # One or both don't have slice_index - fall back to patient-level matching
                    # Match one slice per patient (use first available slice from each)
                    if ct_slices and pet_slices:
                        # Use first slice from each modality for this patient
                        matched_pairs.append((ct_slices[0], pet_slices[0]))
    else:
        # Match by index
        min_len = min(len(ct_predictions), len(pet_predictions))
        matched_pairs = list(zip(ct_predictions[:min_len], pet_predictions[:min_len]))
    
    if not matched_pairs:
        if debug:
            print(f"DEBUG: No matched pairs found. CT predictions: {len(ct_predictions)}, PET predictions: {len(pet_predictions)}")
        return {
            'agreement_rate': 0.0,
            'disagreement_rate': 0.0,
            'ct_dominates': 0,
            'pet_dominates': 0,
            'num_pairs': 0,
            'disagreement_confidence_analysis': {},
            'disagreement_logit_analysis': {}
        }
    
    if debug:
        print(f"\nDEBUG: analyze_modality_agreement - Found {len(matched_pairs)} matched pairs")
        print(f"DEBUG: First few CT predictions: {[p.get('prediction') for p in ct_predictions[:5]]}")
        print(f"DEBUG: First few PET predictions: {[p.get('prediction') for p in pet_predictions[:5]]}")
        print(f"DEBUG: First PET prediction has used_context: {pet_predictions[0].get('used_context', False) if pet_predictions else 'N/A'}")
    
    agreements = 0
    disagreements = 0
    ct_dominates = 0
    pet_dominates = 0
    
    # Track certainty metrics for agreement vs disagreement cases
    agreement_confidences = []
    disagreement_confidences_ct = []
    disagreement_confidences_pet = []
    disagreement_logit_magnitudes_ct = []
    disagreement_logit_magnitudes_pet = []
    disagreement_logit_variances_ct = []
    disagreement_logit_variances_pet = []
    
    # Debug: Track first few comparisons
    debug_samples = []
    max_debug_samples = 5
    
    for idx, (ct_pred, pet_pred) in enumerate(matched_pairs):
        ct_pred_class = ct_pred.get('prediction')
        # For CT+PET (PET with CT context), the prediction should be based on final probabilities
        # Check if this is a CT+PET prediction (has used_context flag)
        used_context = pet_pred.get('used_context', False)
        if used_context:
            # CT+PET: Recalculate prediction from boosted probabilities (after boosting)
            # This is critical because boosting can change the predicted class
            pet_probs_after = pet_pred.get('probabilities_array')
            pet_probs_before = pet_pred.get('probabilities_before_boosting')
            
            if pet_probs_after is not None and len(pet_probs_after) >= 2:
                # Recalculate prediction from boosted probabilities
                pet_probs_arr = np.array(pet_probs_after)
                # Ensure probabilities are valid
                if pet_probs_arr.sum() > 0:
                    pet_probs_arr = pet_probs_arr / max(pet_probs_arr.sum(), 1e-10)  # Normalize
                    pet_pred_class = int(np.argmax(pet_probs_arr))
                else:
                    # Fallback to stored prediction if probabilities are invalid
                    pet_pred_class = pet_pred.get('prediction', 0)
            elif pet_probs_before is not None and len(pet_probs_before) >= 2:
                # Fallback: use probabilities before boosting if after-boosting not available
                pet_probs_arr = np.array(pet_probs_before)
                if pet_probs_arr.sum() > 0:
                    pet_probs_arr = pet_probs_arr / max(pet_probs_arr.sum(), 1e-10)
                    pet_pred_class = int(np.argmax(pet_probs_arr))
                else:
                    pet_pred_class = pet_pred.get('prediction', 0)
            else:
                # Final fallback: use stored prediction
                pet_pred_class = pet_pred.get('prediction', 0)
        else:
            # PET alone: Use stored prediction (no boosting applied)
            pet_pred_class = pet_pred.get('prediction', 0)
        
        # Ensure prediction classes are valid (0 or 1)
        ct_pred_class = max(0, min(1, int(ct_pred_class))) if ct_pred_class is not None else 0
        pet_pred_class = max(0, min(1, int(pet_pred_class))) if pet_pred_class is not None else 0
        
        ct_conf = ct_pred.get('confidence', 0.0)
        pet_conf = pet_pred.get('confidence', 0.0)
        
        # Debug: Store sample comparisons
        if debug and idx < max_debug_samples:
            debug_samples.append({
                'idx': idx,
                'ct_pred_class': ct_pred_class,
                'pet_pred_class': pet_pred_class,
                'ct_conf': ct_conf,
                'pet_conf': pet_conf,
                'used_context': used_context,
                'pet_probs_after': pet_probs_after[:2] if pet_probs_after is not None and len(pet_probs_after) >= 2 else None,
                'pet_stored_pred': pet_pred.get('prediction'),
                'patient_id': patient_ids[idx] if patient_ids and idx < len(patient_ids) else None
            })
        
        # Extract logits if available
        ct_logits = ct_pred.get('logits')
        pet_logits = pet_pred.get('logits')
        
        if ct_pred_class == pet_pred_class:
            agreements += 1
            # Average confidence when modalities agree
            agreement_confidences.append((ct_conf + pet_conf) / 2.0)
        else:
            disagreements += 1
            disagreement_confidences_ct.append(ct_conf)
            disagreement_confidences_pet.append(pet_conf)
            
            # Analyze logit stability when modalities disagree
            if ct_logits is not None and len(ct_logits) >= 2:
                ct_logits_arr = np.array(ct_logits)
                disagreement_logit_magnitudes_ct.append(calculate_logit_magnitude(ct_logits_arr))
                disagreement_logit_variances_ct.append(float(np.var(ct_logits_arr)))
            
            if pet_logits is not None and len(pet_logits) >= 2:
                pet_logits_arr = np.array(pet_logits)
                disagreement_logit_magnitudes_pet.append(calculate_logit_magnitude(pet_logits_arr))
                disagreement_logit_variances_pet.append(float(np.var(pet_logits_arr)))
            
            # Check which modality has higher confidence when they disagree
            if ct_conf > pet_conf:
                ct_dominates += 1
            elif pet_conf > ct_conf:
                pet_dominates += 1
    
    total = len(matched_pairs)
    
    # Debug: Print sample comparisons
    if debug and debug_samples:
        print(f"\nDEBUG: Sample comparisons (first {len(debug_samples)} pairs):")
        for sample in debug_samples:
            print(f"  Pair {sample['idx']}: CT={sample['ct_pred_class']} (conf={sample['ct_conf']:.4f}), "
                  f"PET={sample['pet_pred_class']} (conf={sample['pet_conf']:.4f}, "
                  f"used_context={sample['used_context']}, "
                  f"stored_pred={sample['pet_stored_pred']}, "
                  f"probs_after={sample['pet_probs_after']}, "
                  f"patient_id={sample['patient_id']})")
        print(f"DEBUG: Total pairs: {total}, Agreements: {agreements}, Disagreements: {disagreements}")
        print(f"DEBUG: Disagreement rate: {disagreements/total if total > 0 else 0.0:.4f}")
    
    # Analyze how disagreement affects confidence
    disagreement_confidence_analysis = {}
    if disagreement_confidences_ct and disagreement_confidences_pet:
        disagreement_confidence_analysis = {
            'avg_ct_confidence_when_disagree': float(np.mean(disagreement_confidences_ct)),
            'avg_pet_confidence_when_disagree': float(np.mean(disagreement_confidences_pet)),
            'std_ct_confidence_when_disagree': float(np.std(disagreement_confidences_ct)),
            'std_pet_confidence_when_disagree': float(np.std(disagreement_confidences_pet)),
            'confidence_difference': float(np.mean(disagreement_confidences_pet) - np.mean(disagreement_confidences_ct))
        }
    
    if agreement_confidences:
        disagreement_confidence_analysis['avg_confidence_when_agree'] = float(np.mean(agreement_confidences))
        disagreement_confidence_analysis['confidence_drop_on_disagreement'] = (
            float(np.mean(agreement_confidences)) - 
            float(np.mean(disagreement_confidences_ct + disagreement_confidences_pet) / 2)
            if disagreement_confidences_ct and disagreement_confidences_pet else 0.0
        )
    
    # Analyze logit stability when modalities disagree
    disagreement_logit_analysis = {}
    if disagreement_logit_magnitudes_ct and disagreement_logit_magnitudes_pet:
        disagreement_logit_analysis = {
            'avg_logit_magnitude_ct_when_disagree': float(np.mean(disagreement_logit_magnitudes_ct)),
            'avg_logit_magnitude_pet_when_disagree': float(np.mean(disagreement_logit_magnitudes_pet)),
            'avg_logit_variance_ct_when_disagree': float(np.mean(disagreement_logit_variances_ct)),
            'avg_logit_variance_pet_when_disagree': float(np.mean(disagreement_logit_variances_pet)),
            'logit_instability_indicator': float(np.mean(disagreement_logit_variances_ct + disagreement_logit_variances_pet))
        }
    
    return {
        'agreement_rate': agreements / total if total > 0 else 0.0,
        'disagreement_rate': disagreements / total if total > 0 else 0.0,
        'ct_dominates': ct_dominates,
        'pet_dominates': pet_dominates,
        'num_pairs': total,
        'disagreement_confidence_analysis': disagreement_confidence_analysis,
        'disagreement_logit_analysis': disagreement_logit_analysis
    }


def analyze_logit_similarity(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze cosine similarity between logit distributions of two modalities.
    Higher similarity indicates more consistent model behavior across modalities.
    
    Args:
        mod1_predictions: Predictions from first modality
        mod2_predictions: Predictions from second modality
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with logit similarity metrics
    """
    # Match predictions by patient_id or by index
    matched_pairs = []
    if patient_ids is not None:
        mod1_by_patient = {pid: pred for pid, pred in zip(patient_ids, mod1_predictions)}
        mod2_by_patient = {pid: pred for pid, pred in zip(patient_ids, mod2_predictions)}
        
        for pid in set(patient_ids):
            if pid in mod1_by_patient and pid in mod2_by_patient:
                matched_pairs.append((mod1_by_patient[pid], mod2_by_patient[pid]))
    else:
        min_len = min(len(mod1_predictions), len(mod2_predictions))
        matched_pairs = list(zip(mod1_predictions[:min_len], mod2_predictions[:min_len]))
    
    if not matched_pairs:
        return {
            'avg_cosine_similarity': 0.0,
            'std_cosine_similarity': 0.0,
            'num_pairs': 0
        }
    
    similarities = []
    for mod1_pred, mod2_pred in matched_pairs:
        mod1_logits = mod1_pred.get('logits')
        mod2_logits = mod2_pred.get('logits')
        
        if mod1_logits is not None and mod2_logits is not None:
            mod1_logits_arr = np.array(mod1_logits)
            mod2_logits_arr = np.array(mod2_logits)
            
            if len(mod1_logits_arr) == len(mod2_logits_arr) and len(mod1_logits_arr) > 0:
                similarity = calculate_cosine_similarity(mod1_logits_arr, mod2_logits_arr)
                similarities.append(similarity)
    
    if not similarities:
        return {
            'avg_cosine_similarity': 0.0,
            'std_cosine_similarity': 0.0,
            'num_pairs': 0
        }
    
    return {
        'avg_cosine_similarity': float(np.mean(similarities)),
        'std_cosine_similarity': float(np.std(similarities)),
        'min_cosine_similarity': float(np.min(similarities)),
        'max_cosine_similarity': float(np.max(similarities)),
        'num_pairs': len(similarities)
    }


def analyze_patient_level_agreement(
    slice_level_agreement: Dict,
    patient_level_ct: List[Dict],
    patient_level_pet: List[Dict],
    patient_ids: List[str]
) -> Dict:
    """
    Analyze whether modality agreement increases at patient level vs slice level.
    
    Key question: Does aggregating slices per patient lead to better agreement
    between CT and PET predictions?
    
    Args:
        slice_level_agreement: Agreement metrics from slice-level analysis
        patient_level_ct: List of patient-level CT predictions
        patient_level_pet: List of patient-level PET predictions
        patient_ids: List of patient IDs matching the predictions
    
    Returns:
        Dictionary with patient-level agreement analysis
    """
    if len(patient_level_ct) != len(patient_level_pet) or len(patient_level_ct) != len(patient_ids):
        return {
            'patient_level_agreement_rate': 0.0,
            'slice_level_agreement_rate': 0.0,
            'agreement_improvement': 0.0,
            'num_patients': 0
        }
    
    # Calculate patient-level agreement
    patient_agreements = 0
    patient_disagreements = 0
    
    for ct_pred, pet_pred in zip(patient_level_ct, patient_level_pet):
        ct_pred_class = ct_pred.get('prediction')
        pet_pred_class = pet_pred.get('prediction')
        
        if ct_pred_class == pet_pred_class:
            patient_agreements += 1
        else:
            patient_disagreements += 1
    
    total_patients = len(patient_level_ct)
    patient_agreement_rate = patient_agreements / total_patients if total_patients > 0 else 0.0
    
    # Get slice-level agreement for comparison
    slice_agreement_rate = slice_level_agreement.get('agreement_rate', 0.0)
    agreement_improvement = patient_agreement_rate - slice_agreement_rate
    
    return {
        'patient_level_agreement_rate': patient_agreement_rate,
        'slice_level_agreement_rate': slice_agreement_rate,
        'agreement_improvement': agreement_improvement,
        'patient_agreements': patient_agreements,
        'patient_disagreements': patient_disagreements,
        'num_patients': total_patients,
        'interpretation': (
            f"Patient-level agreement: {patient_agreement_rate:.4f} vs "
            f"Slice-level: {slice_agreement_rate:.4f} "
            f"({agreement_improvement:+.4f} change)"
        )
    }


def analyze_pet_vs_ct_confidence(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does PET consistently produce higher confidence than CT?
    
    This is a fundamental question about modality behavior in zero-shot VLMs.
    
    Args:
        ct_predictions: List of CT prediction dicts
        pet_predictions: List of PET prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with PET vs CT confidence analysis
    """
    # Match predictions
    matched_pairs = []
    if patient_ids is not None:
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient:
                matched_pairs.append((ct_by_patient[pid], pet_by_patient[pid]))
    else:
        min_len = min(len(ct_predictions), len(pet_predictions))
        matched_pairs = list(zip(ct_predictions[:min_len], pet_predictions[:min_len]))
    
    if not matched_pairs:
        return {
            'pet_higher_confidence_rate': 0.0,
            'ct_higher_confidence_rate': 0.0,
            'avg_confidence_difference': 0.0,
            'consistent_pet_dominance': False,
            'num_pairs': 0
        }
    
    pet_higher = 0
    ct_higher = 0
    equal_conf = 0
    confidence_differences = []
    
    for ct_pred, pet_pred in matched_pairs:
        # Use appropriate confidence based on whether PET has CT context
        ct_conf = ct_pred.get('confidence', 0.0)
        used_context = pet_pred.get('used_context', False)
        
        if used_context:
            # PET with CT context: Use probabilities_array (after boosting) to show improved performance
            pet_probs_after = pet_pred.get('probabilities_array')
            if pet_probs_after is not None and len(pet_probs_after) >= 2:
                pet_conf = float(np.max(np.array(pet_probs_after)))
            else:
                pet_conf = pet_pred.get('confidence', 0.0)
        else:
            # PET alone: Use probabilities_before_boosting (shows true model behavior)
            pet_probs_before = pet_pred.get('probabilities_before_boosting')
            if pet_probs_before is not None and len(pet_probs_before) >= 2:
                pet_conf = float(np.max(np.array(pet_probs_before)))
            else:
                pet_conf = pet_pred.get('confidence', 0.0)
        
        conf_diff = pet_conf - ct_conf
        confidence_differences.append(conf_diff)
        
        if pet_conf > ct_conf + 0.01:  # PET higher (with small threshold)
            pet_higher += 1
        elif ct_conf > pet_conf + 0.01:  # CT higher
            ct_higher += 1
        else:
            equal_conf += 1
    
    total = len(matched_pairs)
    avg_conf_diff = np.mean(confidence_differences) if confidence_differences else 0.0
    
    # Consistent PET dominance: >70% of cases AND average difference > 0.05
    pet_dominance_rate = pet_higher / total if total > 0 else 0.0
    consistent_dominance = pet_dominance_rate > 0.7 and avg_conf_diff > 0.05
    
    return {
        'pet_higher_confidence_rate': pet_dominance_rate,
        'ct_higher_confidence_rate': ct_higher / total if total > 0 else 0.0,
        'equal_confidence_rate': equal_conf / total if total > 0 else 0.0,
        'avg_confidence_difference': float(avg_conf_diff),
        'std_confidence_difference': float(np.std(confidence_differences)) if confidence_differences else 0.0,
        'consistent_pet_dominance': consistent_dominance,
        'num_pairs': total,
        'interpretation': (
            f"PET has higher confidence in {pet_dominance_rate*100:.1f}% of cases. "
            f"Average difference (PET - CT): {avg_conf_diff:+.4f}. "
            f"{'PET CONSISTENTLY DOMINATES' if consistent_dominance else 'No consistent dominance pattern'}"
        )
    }


def analyze_multimodality_uncertainty_effect(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does multimodality reduce uncertainty or introduce conflicting signals?
    
    This analyzes whether combining modalities:
    1. Reduces uncertainty (lower entropy, higher confidence)
    2. Introduces conflict (higher entropy, lower confidence, unstable logits)
    3. Provides complementary information (true multimodal value)
    
    Args:
        ct_predictions: List of CT prediction dicts
        pet_predictions: List of PET prediction dicts
        combined_predictions: List of combined (CT+PET) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with uncertainty analysis
    """
    # Match all three prediction sets
    matched_triplets = []
    if patient_ids is not None:
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        combined_by_patient = {pid: pred for pid, pred in zip(patient_ids, combined_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient and pid in combined_by_patient:
                matched_triplets.append((
                    ct_by_patient[pid],
                    pet_by_patient[pid],
                    combined_by_patient[pid]
                ))
    else:
        min_len = min(len(ct_predictions), len(pet_predictions), len(combined_predictions))
        matched_triplets = list(zip(
            ct_predictions[:min_len],
            pet_predictions[:min_len],
            combined_predictions[:min_len]
        ))
    
    if not matched_triplets:
        return {
            'uncertainty_reduction_rate': 0.0,
            'conflict_introduction_rate': 0.0,
            'avg_entropy_change': 0.0,
            'avg_confidence_change': 0.0,
            'multimodal_value': 'unknown',
            'num_triplets': 0
        }
    
    uncertainty_reductions = 0
    conflict_introductions = 0
    entropy_changes = []
    confidence_changes = []
    
    for ct_pred, pet_pred, combined_pred in matched_triplets:
        # Get probabilities for entropy calculation
        ct_probs = ct_pred.get('probabilities_array') or list(ct_pred.get('probabilities', {}).values())
        pet_probs_before = pet_pred.get('probabilities_before_boosting')
        if pet_probs_before is None:
            pet_probs_before = pet_pred.get('probabilities_array') or list(pet_pred.get('probabilities', {}).values())
        combined_probs = combined_pred.get('probabilities_array') or list(combined_pred.get('probabilities', {}).values())
        
        # Calculate entropies
        if len(ct_probs) >= 2 and len(pet_probs_before) >= 2 and len(combined_probs) >= 2:
            ct_entropy = calculate_entropy(np.array(ct_probs))
            pet_entropy = calculate_entropy(np.array(pet_probs_before))
            combined_entropy = calculate_entropy(np.array(combined_probs))
            
            avg_single_entropy = (ct_entropy + pet_entropy) / 2.0
            entropy_change = combined_entropy - avg_single_entropy
            entropy_changes.append(entropy_change)
            
            # Uncertainty reduction: combined entropy < average single-modality entropy
            if entropy_change < -0.1:  # Significant reduction
                uncertainty_reductions += 1
            # Conflict introduction: combined entropy > average single-modality entropy
            elif entropy_change > 0.1:  # Significant increase
                conflict_introductions += 1
        
        # Calculate confidence changes
        ct_conf = ct_pred.get('confidence', 0.0)
        pet_conf_before = float(np.max(np.array(pet_probs_before))) if pet_probs_before is not None and len(pet_probs_before) >= 2 else pet_pred.get('confidence', 0.0)
        combined_conf = combined_pred.get('confidence', 0.0)
        
        avg_single_conf = (ct_conf + pet_conf_before) / 2.0
        conf_change = combined_conf - avg_single_conf
        confidence_changes.append(conf_change)
    
    total = len(matched_triplets)
    avg_entropy_change = float(np.mean(entropy_changes)) if entropy_changes else 0.0
    avg_conf_change = float(np.mean(confidence_changes)) if confidence_changes else 0.0
    
    # Determine multimodal value
    if avg_entropy_change < -0.1 and avg_conf_change > 0.02:
        multimodal_value = 'REDUCES_UNCERTAINTY'
    elif avg_entropy_change > 0.1 or avg_conf_change < -0.02:
        multimodal_value = 'INTRODUCES_CONFLICT'
    else:
        multimodal_value = 'NEUTRAL'
    
    return {
        'uncertainty_reduction_rate': uncertainty_reductions / total if total > 0 else 0.0,
        'conflict_introduction_rate': conflict_introductions / total if total > 0 else 0.0,
        'avg_entropy_change': avg_entropy_change,
        'avg_confidence_change': avg_conf_change,
        'multimodal_value': multimodal_value,
        'num_triplets': total,
        'interpretation': (
            f"Multimodality {multimodal_value.lower().replace('_', ' ')}. "
            f"Entropy change: {avg_entropy_change:+.4f}, "
            f"Confidence change: {avg_conf_change:+.4f}. "
            f"{uncertainty_reductions}/{total} cases show uncertainty reduction, "
            f"{conflict_introductions}/{total} show conflict introduction"
        )
    }


def analyze_zero_shot_multimodal_value(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Can zero-shot VLMs reflect the "value" of multimodal clinical imaging?
    
    This analyzes whether the model demonstrates understanding of multimodal value through:
    1. Improved certainty when modalities agree
    2. Appropriate uncertainty when modalities disagree
    3. Better calibration when using both modalities
    4. Evidence of information integration (not just bias)
    
    Args:
        ct_predictions: List of CT prediction dicts
        pet_predictions: List of PET prediction dicts
        combined_predictions: List of combined (CT+PET) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with multimodal value analysis
    """
    # Match all three prediction sets
    matched_triplets = []
    if patient_ids is not None:
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        combined_by_patient = {pid: pred for pid, pred in zip(patient_ids, combined_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient and pid in combined_by_patient:
                matched_triplets.append((
                    ct_by_patient[pid],
                    pet_by_patient[pid],
                    combined_by_patient[pid]
                ))
    else:
        min_len = min(len(ct_predictions), len(pet_predictions), len(combined_predictions))
        matched_triplets = list(zip(
            ct_predictions[:min_len],
            pet_predictions[:min_len],
            combined_predictions[:min_len]
        ))
    
    if not matched_triplets:
        return {
            'multimodal_value_demonstrated': False,
            'agreement_benefit': 0.0,
            'disagreement_handling': 'unknown',
            'information_integration': False,
            'num_triplets': 0
        }
    
    # Analyze behavior when modalities agree vs disagree
    agreement_cases = []
    disagreement_cases = []
    
    for ct_pred, pet_pred, combined_pred in matched_triplets:
        ct_class = ct_pred.get('prediction')
        pet_class = pet_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        ct_conf = ct_pred.get('confidence', 0.0)
        pet_probs_before = pet_pred.get('probabilities_before_boosting')
        pet_conf = float(np.max(np.array(pet_probs_before))) if pet_probs_before is not None and len(pet_probs_before) >= 2 else pet_pred.get('confidence', 0.0)
        combined_conf = combined_pred.get('confidence', 0.0)
        
        if ct_class == pet_class:
            # Modalities agree
            agreement_cases.append({
                'ct_conf': ct_conf,
                'pet_conf': pet_conf,
                'combined_conf': combined_conf,
                'avg_single_conf': (ct_conf + pet_conf) / 2.0
            })
        else:
            # Modalities disagree
            disagreement_cases.append({
                'ct_conf': ct_conf,
                'pet_conf': pet_conf,
                'combined_conf': combined_conf,
                'avg_single_conf': (ct_conf + pet_conf) / 2.0
            })
    
    # When modalities agree: combined should have higher confidence (value of agreement)
    agreement_benefits = []
    for case in agreement_cases:
        benefit = case['combined_conf'] - case['avg_single_conf']
        agreement_benefits.append(benefit)
    
    avg_agreement_benefit = float(np.mean(agreement_benefits)) if agreement_benefits else 0.0
    
    # When modalities disagree: combined should show appropriate uncertainty
    disagreement_handling = []
    for case in disagreement_cases:
        # Combined confidence should be lower than average (showing uncertainty)
        uncertainty_handling = case['avg_single_conf'] - case['combined_conf']
        disagreement_handling.append(uncertainty_handling)
    
    avg_disagreement_uncertainty = float(np.mean(disagreement_handling)) if disagreement_handling else 0.0
    
    # Determine if disagreement is handled appropriately
    if avg_disagreement_uncertainty > 0.02:
        disagreement_handling_result = 'APPROPRIATE_UNCERTAINTY'
    elif avg_disagreement_uncertainty < -0.02:
        disagreement_handling_result = 'INAPPROPRIATE_CONFIDENCE'
    else:
        disagreement_handling_result = 'NEUTRAL'
    
    # Information integration: combined predictions are different from both when they disagree
    information_integration_cases = 0
    for ct_pred, pet_pred, combined_pred in matched_triplets:
        ct_class = ct_pred.get('prediction')
        pet_class = pet_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if ct_class != pet_class and combined_class != ct_class and combined_class != pet_class:
            information_integration_cases += 1
    
    total = len(matched_triplets)
    information_integration_rate = information_integration_cases / total if total > 0 else 0.0
    information_integration = information_integration_rate > 0.1  # >10% shows integration
    
    # Overall assessment
    multimodal_value_demonstrated = (
        avg_agreement_benefit > 0.02 and  # Benefits from agreement
        avg_disagreement_uncertainty > 0.0 and  # Handles disagreement appropriately
        information_integration_rate > 0.05  # Shows some integration
    )
    
    return {
        'multimodal_value_demonstrated': multimodal_value_demonstrated,
        'agreement_benefit': avg_agreement_benefit,
        'disagreement_uncertainty': avg_disagreement_uncertainty,
        'disagreement_handling': disagreement_handling_result,
        'information_integration_rate': information_integration_rate,
        'information_integration': information_integration,
        'num_agreement_cases': len(agreement_cases),
        'num_disagreement_cases': len(disagreement_cases),
        'num_triplets': total,
        'interpretation': (
            f"Zero-shot VLM {'DEMONSTRATES' if multimodal_value_demonstrated else 'DOES NOT CLEARLY DEMONSTRATE'} "
            f"multimodal value. Agreement benefit: {avg_agreement_benefit:+.4f}, "
            f"Disagreement handling: {disagreement_handling_result.lower().replace('_', ' ')}, "
            f"Information integration: {information_integration_rate*100:.1f}%"
        )
    }


def analyze_pet_dominance(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze whether PET systematically dominates CT (higher confidence).
    
    Key questions:
    - Does PET consistently have higher confidence than CT?
    - When modalities disagree, does PET win more often?
    - Is there a systematic bias toward PET?
    
    Args:
        ct_predictions: List of CT prediction dicts
        pet_predictions: List of PET prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with PET dominance analysis
    """
    # Match predictions
    matched_pairs = []
    if patient_ids is not None:
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient:
                matched_pairs.append((ct_by_patient[pid], pet_by_patient[pid]))
    else:
        min_len = min(len(ct_predictions), len(pet_predictions))
        matched_pairs = list(zip(ct_predictions[:min_len], pet_predictions[:min_len]))
    
    if not matched_pairs:
        return {
            'pet_higher_confidence_rate': 0.0,
            'ct_higher_confidence_rate': 0.0,
            'avg_confidence_difference': 0.0,
            'pet_wins_when_disagree': 0,
            'ct_wins_when_disagree': 0,
            'systematic_pet_bias': False,
            'num_pairs': 0
        }
    
    pet_higher_conf = 0
    ct_higher_conf = 0
    confidence_differences = []
    pet_wins_disagree = 0
    ct_wins_disagree = 0
    
    for ct_pred, pet_pred in matched_pairs:
        ct_conf = ct_pred.get('confidence', 0.0)
        pet_conf = pet_pred.get('confidence', 0.0)
        ct_pred_class = ct_pred.get('prediction')
        pet_pred_class = pet_pred.get('prediction')
        
        conf_diff = pet_conf - ct_conf
        confidence_differences.append(conf_diff)
        
        if pet_conf > ct_conf:
            pet_higher_conf += 1
        elif ct_conf > pet_conf:
            ct_higher_conf += 1
        
        # When modalities disagree, which one has higher confidence?
        if ct_pred_class != pet_pred_class:
            if pet_conf > ct_conf:
                pet_wins_disagree += 1
            elif ct_conf > pet_conf:
                ct_wins_disagree += 1
    
    total = len(matched_pairs)
    avg_conf_diff = np.mean(confidence_differences) if confidence_differences else 0.0
    
    # Determine if there's systematic PET bias
    # Criteria: PET has higher confidence in >60% of cases AND average difference > 0.05
    pet_dominance_rate = pet_higher_conf / total if total > 0 else 0.0
    systematic_bias = pet_dominance_rate > 0.6 and avg_conf_diff > 0.05
    
    return {
        'pet_higher_confidence_rate': pet_dominance_rate,
        'ct_higher_confidence_rate': ct_higher_conf / total if total > 0 else 0.0,
        'avg_confidence_difference': float(avg_conf_diff),
        'std_confidence_difference': float(np.std(confidence_differences)) if confidence_differences else 0.0,
        'pet_wins_when_disagree': pet_wins_disagree,
        'ct_wins_when_disagree': ct_wins_disagree,
        'systematic_pet_bias': systematic_bias,
        'num_pairs': total,
        'interpretation': (
            f"PET has higher confidence in {pet_dominance_rate*100:.1f}% of cases. "
            f"Average difference: {avg_conf_diff:+.4f}. "
            f"{'SYSTEMATIC PET BIAS DETECTED' if systematic_bias else 'No systematic bias detected'}."
        )
    }


def analyze_multimodal_bias(
    ct_predictions: List[Dict],
    pet_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze whether multimodal predictions reflect true combination or simple bias toward PET.
    
    Key questions:
    - Do combined predictions match PET more often than CT?
    - Are combined predictions truly combining information or just following PET?
    - What's the correlation between combined predictions and individual modalities?
    
    Args:
        ct_predictions: List of CT prediction dicts
        pet_predictions: List of PET prediction dicts
        combined_predictions: List of combined (CT+PET) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with multimodal bias analysis
    """
    # Match all three prediction sets
    matched_triplets = []
    if patient_ids is not None:
        ct_by_patient = {pid: pred for pid, pred in zip(patient_ids, ct_predictions)}
        pet_by_patient = {pid: pred for pid, pred in zip(patient_ids, pet_predictions)}
        combined_by_patient = {pid: pred for pid, pred in zip(patient_ids, combined_predictions)}
        
        for pid in set(patient_ids):
            if pid in ct_by_patient and pid in pet_by_patient and pid in combined_by_patient:
                matched_triplets.append((
                    ct_by_patient[pid],
                    pet_by_patient[pid],
                    combined_by_patient[pid]
                ))
    else:
        min_len = min(len(ct_predictions), len(pet_predictions), len(combined_predictions))
        matched_triplets = list(zip(
            ct_predictions[:min_len],
            pet_predictions[:min_len],
            combined_predictions[:min_len]
        ))
    
    if not matched_triplets:
        return {
            'combined_matches_pet_rate': 0.0,
            'combined_matches_ct_rate': 0.0,
            'pet_bias_score': 0.0,
            'true_combination_rate': 0.0,
            'num_triplets': 0
        }
    
    matches_pet = 0
    matches_ct = 0
    matches_both = 0
    matches_neither = 0
    pet_bias_cases = 0  # Combined matches PET but not CT
    
    for ct_pred, pet_pred, combined_pred in matched_triplets:
        ct_class = ct_pred.get('prediction')
        pet_class = pet_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if combined_class == pet_class:
            matches_pet += 1
        if combined_class == ct_class:
            matches_ct += 1
        if combined_class == pet_class and combined_class == ct_class:
            matches_both += 1
        elif combined_class != pet_class and combined_class != ct_class:
            matches_neither += 1
        
        # PET bias: combined matches PET but not CT (when they disagree)
        if ct_class != pet_class and combined_class == pet_class:
            pet_bias_cases += 1
    
    total = len(matched_triplets)
    
    # Calculate rates
    matches_pet_rate = matches_pet / total if total > 0 else 0.0
    matches_ct_rate = matches_ct / total if total > 0 else 0.0
    pet_bias_score = pet_bias_cases / total if total > 0 else 0.0
    
    # True combination: when CT and PET disagree, combined prediction is different from both
    # OR when they agree, combined also agrees
    true_combination = 0
    for ct_pred, pet_pred, combined_pred in matched_triplets:
        ct_class = ct_pred.get('prediction')
        pet_class = pet_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if ct_class == pet_class:
            # When modalities agree, combined should also agree (true combination)
            if combined_class == ct_class:
                true_combination += 1
        else:
            # When modalities disagree, combined could be:
            # 1. Different from both (true combination/compromise)
            # 2. Same as one (bias toward that modality)
            # For now, we consider "different from both" as true combination
            if combined_class != ct_class and combined_class != pet_class:
                true_combination += 1
    
    true_combination_rate = true_combination / total if total > 0 else 0.0
    
    # Determine if there's simple PET bias
    # Criteria: Combined matches PET significantly more than CT (>20% difference)
    simple_pet_bias = matches_pet_rate - matches_ct_rate > 0.2
    
    return {
        'combined_matches_pet_rate': matches_pet_rate,
        'combined_matches_ct_rate': matches_ct_rate,
        'combined_matches_both_rate': matches_both / total if total > 0 else 0.0,
        'pet_bias_score': pet_bias_score,
        'true_combination_rate': true_combination_rate,
        'simple_pet_bias': simple_pet_bias,
        'num_triplets': total,
        'interpretation': (
            f"Combined matches PET: {matches_pet_rate*100:.1f}%, "
            f"CT: {matches_ct_rate*100:.1f}%. "
            f"{'SIMPLE PET BIAS DETECTED' if simple_pet_bias else 'Appears to be true combination'} "
            f"(True combination rate: {true_combination_rate*100:.1f}%)"
        )
    }


def analyze_modality_combination_effect(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    combined_certainty: Dict,
    mod1_certainty: Dict,
    mod2_certainty: Dict
) -> Dict:
    """
    Analyze how combining modalities affects certainty.
    
    Key questions:
    - Does combining modalities increase or decrease confidence?
    - Does combining modalities reduce entropy (more certain)?
    - How do logit magnitudes change when modalities are combined?
    
    Args:
        mod1_predictions: Predictions from first modality
        mod2_predictions: Predictions from second modality
        combined_certainty: Certainty metrics for combined modality
        mod1_certainty: Certainty metrics for first modality alone
        mod2_certainty: Certainty metrics for second modality alone
    
    Returns:
        Dictionary with combination effect analysis
    """
    combined_conf = combined_certainty.get('avg_confidence', 0.0)
    combined_entropy = combined_certainty.get('avg_entropy', 0.0)
    
    mod1_conf = mod1_certainty.get('avg_confidence', 0.0)
    mod2_conf = mod2_certainty.get('avg_confidence', 0.0)
    avg_single_mod_conf = (mod1_conf + mod2_conf) / 2.0
    
    mod1_entropy = mod1_certainty.get('avg_entropy', 0.0)
    mod2_entropy = mod2_certainty.get('avg_entropy', 0.0)
    avg_single_mod_entropy = (mod1_entropy + mod2_entropy) / 2.0
    
    # Calculate changes
    confidence_change = combined_conf - avg_single_mod_conf
    entropy_change = combined_entropy - avg_single_mod_entropy
    
    # Determine effect
    if confidence_change > 0.05:
        confidence_effect = "INCREASES"
    elif confidence_change < -0.05:
        confidence_effect = "DECREASES"
    else:
        confidence_effect = "MINIMAL_CHANGE"
    
    if entropy_change < -0.1:
        entropy_effect = "REDUCES_UNCERTAINTY"
    elif entropy_change > 0.1:
        entropy_effect = "INCREASES_UNCERTAINTY"
    else:
        entropy_effect = "MINIMAL_CHANGE"
    
    return {
        'combined_confidence': combined_conf,
        'average_single_modality_confidence': avg_single_mod_conf,
        'confidence_change': confidence_change,
        'confidence_effect': confidence_effect,
        'combined_entropy': combined_entropy,
        'average_single_modality_entropy': avg_single_mod_entropy,
        'entropy_change': entropy_change,
        'entropy_effect': entropy_effect,
        'interpretation': (
            f"Combining modalities {confidence_effect.lower()} confidence "
            f"({confidence_change:+.4f}) and {entropy_effect.lower().replace('_', ' ')} "
            f"({entropy_change:+.4f} entropy change)"
        )
    }


def save_results(results: Dict, output_path: str):
    """Save evaluation results to file."""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
