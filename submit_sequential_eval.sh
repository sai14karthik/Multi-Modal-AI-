#!/bin/bash
# ============================================================================
# Submit Sequential Modality Evaluation Job to Newton Cluster
# Supports both CT→PET and PET→CT orders
# ============================================================================
# Usage:
#   ./submit_sequential_eval.sh [test] FIRST_MOD SECOND_MOD [MODEL_NAME] [MODEL_ARCH]
#
# Examples:
#   # CT → PET order (original)
#   ./submit_sequential_eval.sh test CT PET
#   ./submit_sequential_eval.sh CT PET
#   ./submit_sequential_eval.sh CT PET openai/clip-vit-large-patch14
#   ./submit_sequential_eval.sh CT PET microsoft/llava-med-v1.5-mistral-7b llava_med
#
#   # PET → CT order (reversed - what prof wants)
#   ./submit_sequential_eval.sh test PET CT
#   ./submit_sequential_eval.sh PET CT
#   ./submit_sequential_eval.sh PET CT openai/clip-vit-large-patch14
#   ./submit_sequential_eval.sh PET CT microsoft/llava-med-v1.5-mistral-7b llava_med
#
# NOTE: Use 'PET' not 'PT' - the dataset config maps 'PET' to the 'PT' folder
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
TEST_MODE=false
FIRST_MOD=""
SECOND_MOD=""
MODEL_NAME="openai/clip-vit-large-patch14"  # Default
MODEL_ARCH="clip"  # Default

# Check for test mode
if [ "$1" == "test" ]; then
    TEST_MODE=true
    shift
fi

# Get modalities
if [ $# -lt 2 ]; then
    echo "Error: Please provide at least two modalities"
    echo ""
    echo "Usage:"
    echo "  ./submit_sequential_eval.sh [test] FIRST_MOD SECOND_MOD [MODEL_NAME] [MODEL_ARCH]"
    echo ""
    echo "Examples:"
    echo "  ./submit_sequential_eval.sh test CT PT"
    echo "  ./submit_sequential_eval.sh PT CT openai/clip-vit-large-patch14"
    echo "  ./submit_sequential_eval.sh CT PT microsoft/llava-med-v1.5-mistral-7b llava_med"
    exit 1
fi

FIRST_MOD="$1"
SECOND_MOD="$2"

# Optional model name and arch
if [ $# -ge 3 ]; then
    MODEL_NAME="$3"
fi
if [ $# -ge 4 ]; then
    MODEL_ARCH="$4"
fi

# Validate model arch
if [ "$MODEL_ARCH" != "clip" ] && [ "$MODEL_ARCH" != "llava_med" ]; then
    echo "Error: MODEL_ARCH must be 'clip' or 'llava_med'"
    exit 1
fi

# Create safe job name
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
JOB_NAME="seq_${FIRST_MOD}_${SECOND_MOD}_${MODEL_NAME_SAFE}"

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: Running ${FIRST_MOD} → ${SECOND_MOD} with 10 samples..."
    JOB_NAME="${JOB_NAME}_test"
else
    echo "FULL MODE: Running ${FIRST_MOD} → ${SECOND_MOD} on full dataset..."
fi
echo "   Model: $MODEL_NAME ($MODEL_ARCH)"
echo ""

# Create temporary job script
JOB_SCRIPT="submit_seq_tmp_$$.sh"

if [ "$TEST_MODE" = true ]; then
    # Test mode: shorter time, 10 samples
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=output_${JOB_NAME}_%j.log
#SBATCH --error=error_${JOB_NAME}_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

set -e

echo "=========================================="
echo "Sequential Evaluation TEST Job"
echo "Order: ${FIRST_MOD} → ${SECOND_MOD}"
echo "Model: $MODEL_NAME ($MODEL_ARCH)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "=========================================="

cd ~/Multi-Modal-AI || cd "\$(dirname "\$0")"
PROJECT_DIR=\$(pwd)
echo "Project directory: \$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=\$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=\$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EOF

    # Add LLaVA-Med specific setup if needed
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        cat >> "$JOB_SCRIPT" <<'LLAVA_EOF'
# Install LLaVA-Med dependencies
echo "Installing LLaVA-Med dependencies..."
pip install --quiet --upgrade "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99 || {
    echo "✗ Failed to install dependencies"
    exit 1
}
echo "✓ Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print('✓ Dependencies verified')" || {
    echo "✗ Dependency verification failed!"
    exit 1
}
LLAVA_EOF
    fi

    cat >> "$JOB_SCRIPT" <<EOF

mkdir -p results

# Run evaluation TEST (10 samples)
echo ""
echo "Starting sequential evaluation TEST (${FIRST_MOD} → ${SECOND_MOD}, 10 samples)..."
echo ""

python3 -u -m src.main \\
    --data_root data \\
    --modalities ${FIRST_MOD} ${SECOND_MOD} \\
    --model_arch ${MODEL_ARCH} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results \\
    --batch_size 1 \\
    --max_samples 10 \\
    --dataset_config data/dataset_config.yaml \\
    --class_names high_grade low_grade \\
    --temperature 0.8 \\
    --aggressive_preprocess

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Test job completed with exit code: \$EXIT_CODE"
echo "End time: \$(date)"
echo "=========================================="

exit \$EXIT_CODE
EOF

else
    # Full mode: full dataset run
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=output_${JOB_NAME}_%j.log
#SBATCH --error=error_${JOB_NAME}_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

set -e

echo "=========================================="
echo "Sequential Evaluation FULL Job"
echo "Order: ${FIRST_MOD} → ${SECOND_MOD}"
echo "Model: $MODEL_NAME ($MODEL_ARCH)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "=========================================="

cd ~/Multi-Modal-AI || cd "\$(dirname "\$0")"
PROJECT_DIR=\$(pwd)
echo "Project directory: \$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=\$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=\$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EOF

    # Add LLaVA-Med specific setup if needed
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        cat >> "$JOB_SCRIPT" <<'LLAVA_EOF'
# Install LLaVA-Med dependencies
echo "Installing LLaVA-Med dependencies..."
pip install --quiet --upgrade "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99 || {
    echo "✗ Failed to install dependencies"
    exit 1
}
echo "✓ Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print('✓ Dependencies verified')" || {
    echo "✗ Dependency verification failed!"
    exit 1
}
LLAVA_EOF
    fi

    # Determine batch size based on model arch
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        BATCH_SIZE=1
    else
        BATCH_SIZE=16
    fi

    cat >> "$JOB_SCRIPT" <<EOF

mkdir -p results

# Run evaluation FULL (all data)
echo ""
echo "Starting sequential evaluation FULL (${FIRST_MOD} → ${SECOND_MOD}, full dataset)..."
echo ""

python3 -u -m src.main \\
    --data_root data \\
    --modalities ${FIRST_MOD} ${SECOND_MOD} \\
    --model_arch ${MODEL_ARCH} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results \\
    --batch_size ${BATCH_SIZE} \\
    --dataset_config data/dataset_config.yaml \\
    --class_names high_grade low_grade \\
    --temperature 0.8 \\
    --aggressive_preprocess

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Job completed with exit code: \$EXIT_CODE"
echo "End time: \$(date)"
echo "=========================================="

if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Sequential evaluation completed successfully"
    echo "Results saved to: results/"
else
    echo "✗ Sequential evaluation failed"
    echo "Check error log: error_${JOB_NAME}_\${SLURM_JOB_ID}.log"
fi

exit \$EXIT_CODE
EOF

fi

chmod +x "$JOB_SCRIPT"

# Submit job
if [ "$TEST_MODE" = true ]; then
    echo "Submitting TEST job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✓ Test job $JOB_ID submitted"
    echo ""
    echo "   Monitor: squeue -u \$USER"
    echo "   Logs:    tail -f output_${JOB_NAME}_${JOB_ID}.log"
    echo ""
    echo "⏱️  Note: Test mode runs 10 samples (~20-40 minutes)"
else
    echo "Submitting FULL job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✓ Job $JOB_ID submitted"
    echo ""
    echo "   Monitor: squeue -u \$USER"
    echo "   Logs:    tail -f output_${JOB_NAME}_${JOB_ID}.log"
    echo ""
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        echo "⏱️  Note: Full dataset run (~8-12 hours for LLaVA-Med)"
    else
        echo "⏱️  Note: Full dataset run (~4-8 hours for CLIP)"
    fi
fi

# Clean up temp script
rm -f "$JOB_SCRIPT"

echo ""
echo "Job submission complete!"
