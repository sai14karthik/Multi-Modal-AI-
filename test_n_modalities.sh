#!/bin/bash
# Test N-modality cascading context on sample data locally
# Usage: ./test_n_modalities.sh [MODALITY1] [MODALITY2] ...
# Example: ./test_n_modalities.sh CT PET
#          ./test_n_modalities.sh MRI XRay

# Get modalities from command line or use default
if [ $# -ge 2 ]; then
    MODALITIES=("$@")
else
    # Default: CT PET (for backward compatibility)
    MODALITIES=("CT" "PET")
fi

MOD1="${MODALITIES[0]}"
MOD2="${MODALITIES[1]}"
MOD_SUFFIX_FORWARD=$(IFS='_'; echo "${MODALITIES[*]}")
MOD_SUFFIX_REVERSE=$(IFS='_'; echo "${MODALITIES[*]}" | awk '{for(i=NF;i>0;i--) printf "%s%s", $i, (i>1?"_":"")}')

echo "=========================================="
echo "Testing N-Modality Cascading Context"
echo "=========================================="
echo "Modalities: ${MODALITIES[*]}"
echo "Forward order: ${MOD_SUFFIX_FORWARD}"
echo "Reverse order: ${MOD_SUFFIX_REVERSE}"
echo ""

# Test with small sample size
MAX_SAMPLES=3

# Test 1: Forward order (Mod1 → Mod2)
echo "Test 1: Forward Order (${MOD1} → ${MOD2})"
echo "=========================================="
python3 -m src.main \
    --data_root data \
    --modalities "${MODALITIES[@]}" \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --split test 2>&1 | tee test_forward.log

TEST1_EXIT=$?

echo ""
echo "Test 2: Reverse Order (${MOD2} → ${MOD1})"
echo "=========================================="
python3 -m src.main \
    --data_root data \
    --modalities "${MODALITIES[@]}" \
    --reverse_order \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --split test 2>&1 | tee test_reverse.log

TEST2_EXIT=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""

if [ $TEST1_EXIT -eq 0 ]; then
    echo "Forward Order (${MOD1}→${MOD2}): PASSED"
    if ls results/results_*_${MOD_SUFFIX_FORWARD}.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_${MOD_SUFFIX_FORWARD}.json | head -1)"
    fi
else
    echo "ERROR: Forward Order (${MOD1}→${MOD2}): FAILED (exit code: $TEST1_EXIT)"
fi

if [ $TEST2_EXIT -eq 0 ]; then
    echo "Reverse Order (${MOD2}→${MOD1}): PASSED"
    if ls results/results_*_${MOD_SUFFIX_REVERSE}.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_${MOD_SUFFIX_REVERSE}.json | head -1)"
    fi
else
    echo "ERROR: Reverse Order (${MOD2}→${MOD1}): FAILED (exit code: $TEST2_EXIT)"
fi

echo ""
if [ $TEST1_EXIT -eq 0 ] && [ $TEST2_EXIT -eq 0 ]; then
    echo "All tests passed!"
    echo ""
    echo "Check logs:"
    echo "  - Forward: test_forward.log"
    echo "  - Reverse: test_reverse.log"
    echo ""
    echo "Check results:"
    echo "  - Forward: results/results_*_${MOD_SUFFIX_FORWARD}.json"
    echo "  - Reverse: results/results_*_${MOD_SUFFIX_REVERSE}.json"
else
    echo "ERROR: Some tests failed. Check logs above."
    exit 1
fi
