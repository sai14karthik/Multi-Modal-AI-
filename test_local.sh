#!/bin/bash
# Local test script for sample data
# Usage: ./test_local.sh [model_name] [model_arch] [max_samples]
# Example: ./test_local.sh openai/clip-vit-base-patch32 clip 10

# Default values
MODEL_NAME="${1:-openai/clip-vit-base-patch32}"
MODEL_ARCH="${2:-clip}"
MAX_SAMPLES="${3:-10}"

# Default paths (adjust if needed)
DATA_ROOT="${DATA_ROOT:-data}"
DATASET_CONFIG="${DATASET_CONFIG:-data/dataset_config.yaml}"
CLASS1="${CLASS1:-high_grade}"
CLASS2="${CLASS2:-low_grade}"
MODALITIES=("CT" "PET")

echo "=========================================="
echo "LOCAL TEST - Sample Data"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Data root: $DATA_ROOT"
echo "Dataset config: $DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Max samples per patient per modality: $MAX_SAMPLES"
echo "Modalities: ${MODALITIES[*]}"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Error: Data root directory not found: $DATA_ROOT"
    echo "   Please set DATA_ROOT environment variable or create the directory"
    exit 1
fi

# Check if dataset config exists
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "⚠️  Warning: Dataset config not found: $DATASET_CONFIG"
    echo "   Will use default config"
fi

echo "=========================================="
echo "Running FORWARD order: ${MODALITIES[*]}"
echo "=========================================="
python3 -u -m src.main \
    --data_root "$DATA_ROOT" \
    --modalities ${MODALITIES[*]} \
    --model_arch "$MODEL_ARCH" \
    --model_name "$MODEL_NAME" \
    --output_dir results \
    --batch_size 4 \
    --dataset_config "$DATASET_CONFIG" \
    --class_names "$CLASS1" "$CLASS2" \
    --temperature 0.8 \
    --max_samples "$MAX_SAMPLES"

FORWARD_EXIT=$?

echo ""
echo "=========================================="
echo "Running REVERSE order: PET CT"
echo "=========================================="
python3 -u -m src.main \
    --data_root "$DATA_ROOT" \
    --modalities PET CT \
    --model_arch "$MODEL_ARCH" \
    --model_name "$MODEL_NAME" \
    --output_dir results \
    --batch_size 4 \
    --dataset_config "$DATASET_CONFIG" \
    --class_names "$CLASS1" "$CLASS2" \
    --temperature 0.8 \
    --max_samples "$MAX_SAMPLES"

REVERSE_EXIT=$?

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Forward order (CT→PET): Exit code $FORWARD_EXIT"
echo "Reverse order (PET→CT): Exit code $REVERSE_EXIT"

if [ $FORWARD_EXIT -eq 0 ] && [ $REVERSE_EXIT -eq 0 ]; then
    echo "✅ Both orders completed successfully!"
    echo ""
    echo "Results saved in: results/"
    echo "Check: ls -lh results/*${MODALITIES[0]}_${MODALITIES[1]}*.json"
    exit 0
else
    echo "❌ One or both orders failed"
    exit 1
fi
