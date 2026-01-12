#!/bin/bash
# ============================================================================
# Submit LLaVA-Med Job to Newton Cluster
# ============================================================================
# Usage:
#   ./submit_llava_med_newton.sh          # Full dataset run
#   ./submit_llava_med_newton.sh test     # Test run (10 samples)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_MODE=false
if [ "$1" == "test" ]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with 10 samples..."
else
    echo "🚀 FULL MODE: Running on full dataset..."
fi
echo ""

# Create temporary job script
JOB_SCRIPT="submit_llava_med_tmp_$$.sh"

if [ "$TEST_MODE" = true ]; then
    # Test mode: shorter time, 10 samples
    cat > "$JOB_SCRIPT" <<'EOF'
#!/bin/bash
#SBATCH --job-name=llava_med_test
#SBATCH --output=output_llava_med_test_%j.log
#SBATCH --error=error_llava_med_test_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

set -e

echo "=========================================="
echo "LLaVA-Med TEST Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

cd ~/Multi-Modal-AI || cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Install dependencies
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

mkdir -p results

# Run LLaVA-Med TEST (10 samples)
echo ""
echo "Starting LLaVA-Med TEST evaluation (10 samples)..."
echo ""

python3 -u -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch llava_med \
    --model_name microsoft/llava-med-v1.5-mistral-7b \
    --output_dir results \
    --batch_size 1 \
    --max_samples 10 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --aggressive_preprocess

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
EOF
else
    # Full mode: full dataset run
    cat > "$JOB_SCRIPT" <<'EOF'
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

set -e

echo "=========================================="
echo "LLaVA-Med Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd ~/Multi-Modal-AI || cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Install dependencies
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

mkdir -p results

# Run LLaVA-Med FULL evaluation
echo ""
echo "Starting LLaVA-Med evaluation (full dataset)..."
echo "Note: This includes fixes for improved logits extraction"
echo ""

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
else
    echo "✗ LLaVA-Med evaluation failed"
    echo "Check error log: error_llava_med_${SLURM_JOB_ID}.log"
fi

exit $EXIT_CODE
EOF
fi

chmod +x "$JOB_SCRIPT"

# Submit job
if [ "$TEST_MODE" = true ]; then
    echo "📝 Submitting LLaVA-Med TEST job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✓ Test job $JOB_ID submitted"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    tail -f output_llava_med_test_${JOB_ID}.log"
    echo ""
   
else
    echo "📝 Submitting LLaVA-Med FULL job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✓ Job $JOB_ID submitted"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    tail -f output_llava_med_${JOB_ID}.log"
    echo ""
  
fi

# Clean up temp script
rm -f "$JOB_SCRIPT"

echo ""
echo "✅ Job submission complete!"