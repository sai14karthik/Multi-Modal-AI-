#!/usr/bin/env python3
"""
Unit test for certainty metrics functions using mock data.
Tests the logic without requiring model or data access.
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from utils.evaluation import (
    calculate_entropy,
    calculate_probability_gap,
    calculate_logit_magnitude,
    analyze_certainty_metrics,
    analyze_modality_agreement,
    analyze_ct_context_influence
)

def test_entropy():
    """Test entropy calculation."""
    print("Testing calculate_entropy...")
    
    # High confidence (low entropy)
    probs_high_conf = np.array([0.9, 0.1])
    entropy_high = calculate_entropy(probs_high_conf)
    assert entropy_high < 0.5, f"High confidence should have low entropy, got {entropy_high}"
    print(f"  ✓ High confidence entropy: {entropy_high:.4f} (expected < 0.5)")
    
    # Low confidence (high entropy)
    probs_low_conf = np.array([0.5, 0.5])
    entropy_low = calculate_entropy(probs_low_conf)
    assert entropy_low > 0.9, f"Low confidence should have high entropy, got {entropy_low}"
    print(f"  ✓ Low confidence entropy: {entropy_low:.4f} (expected > 0.9)")
    
    print("  ✓ Entropy calculation passed\n")

def test_probability_gap():
    """Test probability gap calculation."""
    print("Testing calculate_probability_gap...")
    
    # Large gap (confident)
    probs_large_gap = np.array([0.9, 0.1])
    gap_large = calculate_probability_gap(probs_large_gap)
    assert gap_large > 0.7, f"Large gap expected, got {gap_large}"
    print(f"  ✓ Large gap: {gap_large:.4f}")
    
    # Small gap (uncertain)
    probs_small_gap = np.array([0.51, 0.49])
    gap_small = calculate_probability_gap(probs_small_gap)
    assert gap_small < 0.1, f"Small gap expected, got {gap_small}"
    print(f"  ✓ Small gap: {gap_small:.4f}")
    
    print("  ✓ Probability gap calculation passed\n")

def test_logit_magnitude():
    """Test logit magnitude calculation."""
    print("Testing calculate_logit_magnitude...")
    
    logits = np.array([2.5, -1.3])
    magnitude = calculate_logit_magnitude(logits)
    expected = np.linalg.norm(logits)
    assert abs(magnitude - expected) < 1e-6, f"Magnitude mismatch: {magnitude} vs {expected}"
    print(f"  ✓ Logit magnitude: {magnitude:.4f}")
    
    print("  ✓ Logit magnitude calculation passed\n")

def test_analyze_certainty_metrics():
    """Test certainty metrics analysis."""
    print("Testing analyze_certainty_metrics...")
    
    mock_predictions = [
        {
            'prediction': 0,
            'confidence': 0.85,
            'probabilities_array': [0.85, 0.15],
            'logits': [2.0, -1.0]
        },
        {
            'prediction': 1,
            'confidence': 0.75,
            'probabilities_array': [0.25, 0.75],
            'logits': [-1.0, 2.0]
        },
        {
            'prediction': 0,
            'confidence': 0.90,
            'probabilities_array': [0.90, 0.10],
            'logits': [2.5, -1.5]
        }
    ]
    
    metrics = analyze_certainty_metrics(mock_predictions, 'CT')
    
    assert metrics['num_samples'] == 3, f"Expected 3 samples, got {metrics['num_samples']}"
    assert metrics['avg_confidence'] > 0.7, f"Expected avg confidence > 0.7, got {metrics['avg_confidence']}"
    assert metrics['avg_entropy'] > 0, f"Expected entropy > 0, got {metrics['avg_entropy']}"
    assert metrics['avg_probability_gap'] > 0, f"Expected prob gap > 0, got {metrics['avg_probability_gap']}"
    assert metrics['avg_logit_magnitude'] > 0, f"Expected logit magnitude > 0, got {metrics['avg_logit_magnitude']}"
    
    print(f"  ✓ Analyzed {metrics['num_samples']} samples")
    print(f"  ✓ Avg confidence: {metrics['avg_confidence']:.4f}")
    print(f"  ✓ Avg entropy: {metrics['avg_entropy']:.4f}")
    print(f"  ✓ Avg prob gap: {metrics['avg_probability_gap']:.4f}")
    print(f"  ✓ Avg logit magnitude: {metrics['avg_logit_magnitude']:.4f}")
    
    print("  ✓ Certainty metrics analysis passed\n")

def test_analyze_modality_agreement():
    """Test modality agreement analysis."""
    print("Testing analyze_modality_agreement...")
    
    ct_predictions = [
        {'prediction': 0, 'confidence': 0.8},
        {'prediction': 1, 'confidence': 0.7},
        {'prediction': 0, 'confidence': 0.9}
    ]
    
    pet_predictions = [
        {'prediction': 0, 'confidence': 0.85},  # Agrees
        {'prediction': 1, 'confidence': 0.75},  # Agrees
        {'prediction': 1, 'confidence': 0.6}    # Disagrees (PET has lower conf)
    ]
    
    agreement = analyze_modality_agreement(ct_predictions, pet_predictions)
    
    assert agreement['num_pairs'] == 3, f"Expected 3 pairs, got {agreement['num_pairs']}"
    assert agreement['agreement_rate'] == 2/3, f"Expected 2/3 agreement, got {agreement['agreement_rate']}"
    assert agreement['disagreement_rate'] == 1/3, f"Expected 1/3 disagreement, got {agreement['disagreement_rate']}"
    
    print(f"  ✓ Agreement rate: {agreement['agreement_rate']:.4f}")
    print(f"  ✓ Disagreement rate: {agreement['disagreement_rate']:.4f}")
    print(f"  ✓ CT dominates: {agreement['ct_dominates']}")
    print(f"  ✓ PET dominates: {agreement['pet_dominates']}")
    
    print("  ✓ Modality agreement analysis passed\n")

def test_analyze_ct_context_influence():
    """Test CT context influence analysis."""
    print("Testing analyze_ct_context_influence...")
    
    pet_predictions = [
        {
            'probabilities_before_boosting': [0.6, 0.4],
            'probabilities_array': [0.85, 0.15],  # Confidence increased after boosting
            'confidence': 0.85
        },
        {
            'probabilities_before_boosting': [0.55, 0.45],
            'probabilities_array': [0.70, 0.30],  # Confidence increased
            'confidence': 0.70
        },
        {
            'probabilities_before_boosting': [0.7, 0.3],
            'probabilities_array': [0.65, 0.35],  # Confidence decreased (unusual but possible)
            'confidence': 0.65
        }
    ]
    
    influence = analyze_ct_context_influence(pet_predictions)
    
    assert influence['num_samples'] == 3, f"Expected 3 samples, got {influence['num_samples']}"
    assert influence['context_increases_confidence'] >= 2, f"Expected at least 2 increases, got {influence['context_increases_confidence']}"
    assert influence['avg_confidence_change'] > 0, f"Expected positive avg change, got {influence['avg_confidence_change']}"
    
    print(f"  ✓ Samples: {influence['num_samples']}")
    print(f"  ✓ Prediction changes: {influence['context_changes_prediction']}")
    print(f"  ✓ Confidence increases: {influence['context_increases_confidence']}")
    print(f"  ✓ Confidence decreases: {influence['context_decreases_confidence']}")
    print(f"  ✓ Avg confidence change: {influence['avg_confidence_change']:+.4f}")
    
    print("  ✓ CT context influence analysis passed\n")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Empty predictions
    empty_metrics = analyze_certainty_metrics([], 'CT')
    assert empty_metrics['num_samples'] == 0, "Empty predictions should return 0 samples"
    print("  ✓ Empty predictions handled")
    
    # Missing probabilities_array (fallback to probabilities dict)
    predictions_with_dict = [
        {
            'prediction': 0,
            'confidence': 0.8,
            'probabilities': {'high_grade': 0.8, 'low_grade': 0.2},
            'logits': [1.0, -1.0]
        }
    ]
    metrics = analyze_certainty_metrics(predictions_with_dict, 'CT')
    assert metrics['num_samples'] == 1, "Should handle probabilities dict"
    print("  ✓ Probabilities dict fallback works")
    
    # Missing logits
    predictions_no_logits = [
        {
            'prediction': 0,
            'confidence': 0.8,
            'probabilities_array': [0.8, 0.2]
        }
    ]
    metrics = analyze_certainty_metrics(predictions_no_logits, 'CT')
    assert metrics['avg_logit_magnitude'] == 0.0, "Missing logits should result in 0 magnitude"
    print("  ✓ Missing logits handled")
    
    print("  ✓ Edge cases passed\n")

def main():
    """Run all tests."""
    print("="*80)
    print("Testing Certainty Metrics Functions")
    print("="*80)
    print()
    
    try:
        test_entropy()
        test_probability_gap()
        test_logit_magnitude()
        test_analyze_certainty_metrics()
        test_analyze_modality_agreement()
        test_analyze_ct_context_influence()
        test_edge_cases()
        
        print("="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

