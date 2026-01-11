#!/bin/bash
#SBATCH --job-name=llava_med_eval
#SBATCH --output=output_llava_med_%j.log
#SBATCH --error=error_llava_med_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# ============================================================================
# LLaVA-Med Evaluation Job for Newton Cluster
# ============================================================================
# This script runs the LLaVA-Med model evaluation with the latest fixes:
# - Improved logits extraction
# - Increased max_new_tokens for better responses
# - Image-dependent logits fallback
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "LLaVA-Med Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd ~/Multi-Modal-AI || cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: venv not found, using system Python"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Install/verify LLaVA-Med dependencies
echo ""
echo "Installing LLaVA-Med dependencies..."
pip install --quiet --upgrade "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99 || {
    echo "✗ Failed to install dependencies"
    exit 1
}
echo "✓ Dependencies installed successfully!"

# Clear corrupted tokenizer cache if it exists
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
echo "Verifying dependencies..."
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print('✓ Dependencies verified')" || {
    echo "✗ Dependency verification failed!"
    exit 1
}

# Create output directory
mkdir -p results

# Run LLaVA-Med evaluation
echo ""
echo "=========================================="
echo "Starting LLaVA-Med evaluation..."
echo "=========================================="
echo "Model: microsoft/llava-med-v1.5-mistral-7b"
echo "Modalities: CT PET"
echo "Device: cuda"
echo "Batch size: 1"
echo "Temperature: 0.8"
echo ""
echo "Note: This includes fixes for:"
echo "  - Improved logits extraction"
echo "  - Increased max_new_tokens (3→10)"
echo "  - Image-dependent logits fallback"
echo ""

# Run with unbuffered output for real-time logging
python3 -u -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch llava_med \
    --model_name microsoft/llava-med-v1.5-mistral-7b \
    --output_dir results \
    --batch_size 1 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --aggressive_preprocess

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ LLaVA-Med evaluation completed successfully"
    echo "Results saved to: results/"
    echo ""
    echo "Check results:"
    echo "  ls -lh results/results_microsoft_llava-med-v1.5-mistral-7b.json"
else
    echo "✗ LLaVA-Med evaluation failed"
    echo "Check error log: error_llava_med_${SLURM_JOB_ID}.log"
    echo "Check output log: output_llava_med_${SLURM_JOB_ID}.log"
fi

exit $EXIT_CODE
