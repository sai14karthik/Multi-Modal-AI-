#!/bin/bash
# Test BiomedCLIP with 3 samples locally
# Usage: ./test_biomedclip_samples.sh

echo "=================================================================================="
echo "Testing BiomedCLIP with 3 Samples"
echo "=================================================================================="
echo ""
echo "This will test:"
echo "  - BiomedCLIP model loading"
echo "  - Model inference"
echo "  - Certainty metrics calculation"
echo "  - Modality agreement analysis"
echo "  - CT context influence analysis"
echo ""
echo "=================================================================================="
echo ""

# Set thread limits
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Check if venv exists (for local testing)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run test with 3 samples using BiomedCLIP
echo "Running test with microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224..."
echo ""

python3 -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir test_results \
    --batch_size 1 \
    --max_samples 3 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8

EXIT_CODE=$?

echo ""
echo "=================================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST COMPLETED SUCCESSFULLY"
    echo "=================================================================================="
    echo ""
    echo "Checking results..."
    
    if [ -d "test_results" ]; then
        RESULT_FILES=$(ls test_results/results_*.json 2>/dev/null | wc -l)
        if [ $RESULT_FILES -gt 0 ]; then
            echo "Results file(s) created in test_results/"
            echo ""
            echo "Result file contents:"
            ls -lh test_results/results_*.json
            echo ""
            echo "To view results:"
            echo "  cat test_results/results_*.json | python3 -m json.tool"
            echo ""
            echo "Check for BiomedCLIP warnings:"
            echo "  grep -i 'warning.*biomedclip\|successfully loaded biomedclip\|fallback' <(python3 -m src.main --help 2>&1 || echo '')"
        else
            echo "WARNING: No result files found in test_results/"
        fi
    else
        echo "WARNING: test_results directory not created"
    fi
else
    echo "TEST FAILED with exit code $EXIT_CODE"
    echo "=================================================================================="
    echo ""
    echo "Please check:"
    echo "  1. Data directory exists: data/Lung-PET-CT-Dx/"
    echo "  2. Dataset config exists: data/dataset_config.yaml"
    echo "  3. Python dependencies are installed"
    echo "  4. Model can be downloaded (requires internet)"
    echo ""
    echo "If BiomedCLIP fails to load, check the output for warnings about:"
    echo "  - Missing config.json"
    echo "  - Fallback to default model"
fi

echo ""
exit $EXIT_CODE

