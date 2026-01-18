#!/bin/bash
# Test N-modality cascading context on sample data locally
# Usage: ./test_n_modalities.sh

echo "=========================================="
echo "Testing N-Modality Cascading Context"
echo "=========================================="
echo ""

# Test with small sample size
MAX_SAMPLES=3

# Test 1: 2 modalities (CT → PET) - backward compatibility
echo "Test 1: 2 Modalities (CT → PET)"
echo "=========================================="
python3 -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --split test 2>&1 | tee test_2mod.log

TEST1_EXIT=$?

echo ""
echo "Test 2: 2 Modalities Reversed (PET → CT)"
echo "=========================================="
python3 -m src.main \
    --data_root data \
    --modalities CT PET \
    --reverse_order \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --split test 2>&1 | tee test_2mod_reversed.log

TEST2_EXIT=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""

if [ $TEST1_EXIT -eq 0 ]; then
    echo "2 Modalities (CT→PET): PASSED"
    if ls results/results_*_CT_PET.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_CT_PET.json | head -1)"
    fi
else
    echo "ERROR: 2 Modalities (CT→PET): FAILED (exit code: $TEST1_EXIT)"
fi

if [ $TEST2_EXIT -eq 0 ]; then
    echo "2 Modalities Reversed (PET→CT): PASSED"
    if ls results/results_*_PET_CT.json 1> /dev/null 2>&1; then
        echo "   Results: $(ls results/results_*_PET_CT.json | head -1)"
    fi
else
    echo "ERROR: 2 Modalities Reversed (PET→CT): FAILED (exit code: $TEST2_EXIT)"
fi

echo ""
if [ $TEST1_EXIT -eq 0 ] && [ $TEST2_EXIT -eq 0 ]; then
    echo "All tests passed!"
    echo ""
    echo "Check logs:"
    echo "  - CT→PET: test_2mod.log"
    echo "  - PET→CT: test_2mod_reversed.log"
    echo ""
    echo "Check results:"
    echo "  - CT→PET: results/results_*_CT_PET.json"
    echo "  - PET→CT: results/results_*_PET_CT.json"
else
    echo "ERROR: Some tests failed. Check logs above."
    exit 1
fi
