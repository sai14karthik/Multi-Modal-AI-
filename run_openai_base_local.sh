#!/bin/bash
# Run OpenAI CLIP Base locally (for testing before submitting to Newton)
# Usage: ./run_openai_base_local.sh [test]
#   - Without arguments: Full dataset run
#   - With "test": Quick test with 10 samples

TEST_MODE=false
if [ "$1" == "test" ]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with 10 samples locally..."
else
    echo "🚀 FULL MODE: Running on full dataset locally..."
fi
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Make sure dependencies are installed."
fi

echo ""

if [ "$TEST_MODE" = true ]; then
    # Test mode: 10 samples
    echo "Starting OpenAI CLIP Base TEST evaluation (10 samples)..."
    python3 -u -m src.main \
        --data_root data \
        --modalities CT PET \
        --model_arch clip \
        --model_name openai/clip-vit-base-patch32 \
        --output_dir results \
        --batch_size 8 \
        --max_samples 10 \
        --dataset_config data/dataset_config.yaml \
        --class_names high_grade low_grade \
        --temperature 0.8
else
    # Full mode: full dataset
    echo "Starting OpenAI CLIP Base evaluation (full dataset)..."
    python3 -u -m src.main \
        --data_root data \
        --modalities CT PET \
        --model_arch clip \
        --model_name openai/clip-vit-base-patch32 \
        --output_dir results \
        --batch_size 16 \
        --dataset_config data/dataset_config.yaml \
        --class_names high_grade low_grade \
        --temperature 0.8 \
        --aggressive_preprocess
fi

echo ""
echo "✅ Evaluation complete!"
