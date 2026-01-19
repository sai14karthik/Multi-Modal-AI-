#!/bin/bash
# Template for testing a new dataset
# Copy this file and modify for your dataset
# Usage: ./test_your_dataset.sh

echo "=========================================="
echo "Testing Your New Dataset"
echo "=========================================="

# ============================================
# CONFIGURE THESE FOR YOUR DATASET
# ============================================
DATA_ROOT="."                                    # Root directory (usually "." or "data")
DATASET_CONFIG="data/your_dataset_config.yaml"  # Path to your config YAML
CLASS1="class0"                                  # First class name
CLASS2="class1"                                  # Second class name
MODALITY1="CT"                                   # First modality
MODALITY2="MRI"                                  # Second modality
MAX_SAMPLES=10                                   # Number of samples for testing (use small number first)
# ============================================

echo "Dataset config: $DATASET_CONFIG"
echo "Modalities: $MODALITY1 $MODALITY2"
echo "Classes: $CLASS1 $CLASS2"
echo "Max samples: $MAX_SAMPLES"
echo ""

# Test forward order
echo "Test 1: Forward Order ($MODALITY1 → $MODALITY2)"
echo "=========================================="
python3 -m src.main \
    --data_root "$DATA_ROOT" \
    --modalities "$MODALITY1" "$MODALITY2" \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config "$DATASET_CONFIG" \
    --class_names "$CLASS1" "$CLASS2" \
    --temperature 0.8 2>&1 | tee test_forward.log

TEST1_EXIT=$?

echo ""
echo "Test 2: Reverse Order ($MODALITY2 → $MODALITY1)"
echo "=========================================="
python3 -m src.main \
    --data_root "$DATA_ROOT" \
    --modalities "$MODALITY1" "$MODALITY2" \
    --reverse_order \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config "$DATASET_CONFIG" \
    --class_names "$CLASS1" "$CLASS2" \
    --temperature 0.8 2>&1 | tee test_reverse.log

TEST2_EXIT=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""

if [ $TEST1_EXIT -eq 0 ]; then
    echo "✅ Forward Order ($MODALITY1→$MODALITY2): PASSED"
    if ls results/results_*_${MODALITY1}_${MODALITY2}.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_${MODALITY1}_${MODALITY2}.json | head -1)"
    fi
else
    echo "❌ Forward Order ($MODALITY1→$MODALITY2): FAILED (exit code: $TEST1_EXIT)"
fi

if [ $TEST2_EXIT -eq 0 ]; then
    echo "✅ Reverse Order ($MODALITY2→$MODALITY1): PASSED"
    if ls results/results_*_${MODALITY2}_${MODALITY1}.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_${MODALITY2}_${MODALITY1}.json | head -1)"
    fi
else
    echo "❌ Reverse Order ($MODALITY2→$MODALITY1): FAILED (exit code: $TEST2_EXIT)"
fi

echo ""
if [ $TEST1_EXIT -eq 0 ] && [ $TEST2_EXIT -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "Next steps:"
    echo "1. Test with more samples: Increase MAX_SAMPLES"
    echo "2. Test on Newton: Use submit_single_model.sh"
    echo ""
    echo "Check logs:"
    echo "  - Forward: test_forward.log"
    echo "  - Reverse: test_reverse.log"
    echo ""
    echo "Check results:"
    echo "  - Forward: results/results_*_${MODALITY1}_${MODALITY2}.json"
    echo "  - Reverse: results/results_*_${MODALITY2}_${MODALITY1}.json"
else
    echo "❌ Some tests failed. Check logs above."
    exit 1
fi
