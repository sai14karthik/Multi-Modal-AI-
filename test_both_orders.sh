#!/bin/bash
# Test both CT→PET and PET→CT orders locally
# Usage: ./test_both_orders.sh

echo "=========================================="
echo "Testing Both Modality Orders Locally"
echo "=========================================="
echo ""

# Test with small sample size
MAX_SAMPLES=5

echo "Test 1: CT → PET Order (Original)"
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
    --split test \
    --no_progress 2>&1 | tee test_ct_pt.log

CT_PT_EXIT=$?

echo ""
echo "Test 2: PET → CT Order (Reversed)"
echo "=========================================="
python3 -m src.main \
    --data_root data \
    --modalities PET CT \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
    --output_dir results \
    --batch_size 4 \
    --max_samples $MAX_SAMPLES \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --split test \
    --no_progress 2>&1 | tee test_pt_ct.log

PT_CT_EXIT=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""

if [ $CT_PT_EXIT -eq 0 ]; then
    echo "CT → PET order: PASSED"
    # Check if results file exists
    if ls results/results_*_CT_PET.json 1> /dev/null 2>&1; then
        echo "   Results file created: $(ls results/results_*_CT_PET.json | head -1)"
    else
        echo "   WARNING: Results file not found"
    fi
else
    echo "ERROR: CT → PET order: FAILED (exit code: $CT_PT_EXIT)"
fi

if [ $PT_CT_EXIT -eq 0 ]; then
    echo "PET → CT order: PASSED"
    # Check if results file exists
    if ls results/results_*_PET_CT.json 1> /dev/null 2>&1; then
        echo "   Results file created: $(ls results/results_*_PET_CT.json | head -1)"
    else
        echo "   WARNING: Results file not found"
    fi
else
    echo "ERROR: PET → CT order: FAILED (exit code: $PT_CT_EXIT)"
fi

echo ""
if [ $CT_PT_EXIT -eq 0 ] && [ $PT_CT_EXIT -eq 0 ]; then
    echo "Both orders work correctly!"
    echo ""
    echo "Check logs:"
    echo "  - CT→PET: test_ct_pt.log"
    echo "  - PET→CT: test_pt_ct.log"
    echo ""
    echo "Check results:"
    echo "  - CT→PET: results/results_*_CT_PET.json"
    echo "  - PET→CT: results/results_*_PET_CT.json"
else
    echo "ERROR: Some tests failed. Check logs above."
    exit 1
fi
