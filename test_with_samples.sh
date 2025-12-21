#!/bin/bash
# Test script to run with a few samples
# Usage: ./test_with_samples.sh

echo "=================================================================================="
echo "Testing Certainty Metrics with 3 Samples"
echo "=================================================================================="
echo ""
echo "This will test:"
echo "  - Model loading and inference"
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

# Run test with 3 samples using a small, fast model
echo "Running test with openai/clip-vit-base-patch32 (small, fast model)..."
echo ""

python3 -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name openai/clip-vit-base-patch32 \
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
fi

echo ""
exit $EXIT_CODE

