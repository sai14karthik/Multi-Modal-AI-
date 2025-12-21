#!/usr/bin/env python3
"""
Quick test script to verify certainty metrics with mock data.
This tests the evaluation functions without requiring model or data access.
"""

import sys
import json
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from utils.evaluation import (
    calculate_entropy,
    calculate_probability_gap,
    calculate_logit_magnitude,
    analyze_certainty_metrics,
    analyze_modality_agreement,
    analyze_ct_context_influence,
    evaluate_sequential_modalities,
    print_evaluation_results
)

def create_mock_results():
    """Create mock results that simulate what the main script would produce."""
    results = {}
    
    # Simulate 3 CT predictions
    for i in range(3):
        case_id = f"high_grade_image_{i}_CT"
        results[case_id] = [{
            'modalities_used': ['CT'],
            'prediction': 0,  # high_grade
            'confidence': 0.75 + i * 0.05,
            'label': 0,
            'probabilities': {'high_grade': 0.75 + i * 0.05, 'low_grade': 0.25 - i * 0.05},
            'probabilities_array': [0.75 + i * 0.05, 0.25 - i * 0.05],
            'probabilities_before_boosting': None,  # CT doesn't have boosting
            'logits': [1.5 + i * 0.2, -1.5 - i * 0.2],
            'patient_id': f'patient_{i}'
        }]
    
    # Simulate 3 PET predictions with CT context
    for i in range(3):
        case_id = f"high_grade_image_{i}_PET"
        # PET predictions with CT context - confidence should be higher
        prob_before = [0.70 + i * 0.05, 0.30 - i * 0.05]
        prob_after = [0.85 + i * 0.05, 0.15 - i * 0.05]  # Boosted after CT context
        
        results[case_id] = [{
            'modalities_used': ['PET'],
            'prediction': 0,  # high_grade
            'confidence': prob_after[0],
            'label': 0,
            'probabilities': {'high_grade': prob_after[0], 'low_grade': prob_after[1]},
            'probabilities_array': prob_after,
            'probabilities_before_boosting': prob_before,  # Before boosting
            'logits': [2.0 + i * 0.2, -2.0 - i * 0.2],
            'patient_id': f'patient_{i}',
            'used_context': True,
            'context_from': ['CT']
        }]
    
    return results

def test_evaluation_pipeline():
    """Test the complete evaluation pipeline."""
    print("="*80)
    print("Testing Complete Evaluation Pipeline with Mock Data")
    print("="*80)
    print()
    
    # Create mock results
    print("Creating mock results (3 CT + 3 PET predictions)...")
    results = create_mock_results()
    print(f"✓ Created {len(results)} case results")
    print()
    
    # Test evaluate_sequential_modalities
    print("-"*80)
    print("Step 1: Evaluating sequential modalities...")
    print("-"*80)
    evaluation_results = evaluate_sequential_modalities(results, ['CT', 'PET'])
    
    print(f"✓ Evaluated {len(evaluation_results['step_results'])} modalities")
    print(f"✓ Agreement metrics: {'Present' if evaluation_results.get('agreement_metrics') else 'Missing'}")
    print()
    
    # Test certainty metrics
    print("-"*80)
    print("Step 2: Checking certainty metrics...")
    print("-"*80)
    for mod_name, mod_data in evaluation_results['step_results'].items():
        cert_metrics = mod_data.get('certainty_metrics', {})
        print(f"{mod_name}:")
        print(f"  - Avg confidence: {cert_metrics.get('avg_confidence', 0.0):.4f}")
        print(f"  - Avg entropy: {cert_metrics.get('avg_entropy', 0.0):.4f}")
        print(f"  - Avg prob gap: {cert_metrics.get('avg_probability_gap', 0.0):.4f}")
        print(f"  - Samples: {cert_metrics.get('num_samples', 0)}")
    print()
    
    # Test CT context influence
    print("-"*80)
    print("Step 3: Analyzing CT context influence...")
    print("-"*80)
    pet_predictions = []
    for case_id, case_results in results.items():
        for result in case_results:
            if result.get('modalities_used') == ['PET'] and result.get('used_context'):
                pet_predictions.append(result)
    
    if pet_predictions:
        context_influence = analyze_ct_context_influence(pet_predictions)
        print(f"✓ Analyzed {context_influence['num_samples']} PET predictions with CT context")
        print(f"  - Avg confidence change: {context_influence['avg_confidence_change']:+.4f}")
        print(f"  - Confidence increases: {context_influence['context_increases_confidence']}")
        print(f"  - Confidence decreases: {context_influence['context_decreases_confidence']}")
    print()
    
    # Test print function
    print("-"*80)
    print("Step 4: Testing output format...")
    print("-"*80)
    try:
        print_evaluation_results(evaluation_results)
        print("✓ Output format test passed")
    except Exception as e:
        print(f"✗ Output format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test JSON serialization
    print("-"*80)
    print("Step 5: Testing JSON serialization...")
    print("-"*80)
    try:
        # Add CT context influence to results
        if pet_predictions:
            evaluation_results['ct_context_influence'] = context_influence
        
        json_str = json.dumps(evaluation_results, indent=2)
        print(f"✓ JSON serialization successful ({len(json_str)} bytes)")
        
        # Verify it can be deserialized
        loaded = json.loads(json_str)
        print(f"✓ JSON deserialization successful")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("="*80)
    print("ALL TESTS PASSED")
    print("="*80)
    print()
    print("The code is ready for testing with actual data and models.")
    print("Run: ./test_with_samples.sh (on a system with data and model access)")
    print()
    
    return True

if __name__ == '__main__':
    try:
        success = test_evaluation_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

