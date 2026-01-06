#!/bin/bash
# Submit OpenAI CLIP Base job to Newton
# Usage: ./submit_openai_base.sh [test]
#   - Without arguments: Full dataset run
#   - With "test": Quick test with 10 samples

TEST_MODE=false
if [ "$1" == "test" ]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with 10 samples..."
else
    echo "🚀 FULL MODE: Running on full dataset..."
fi
echo ""

# Create temporary job script
JOB_SCRIPT="submit_openai_base_tmp.sh"

if [ "$TEST_MODE" = true ]; then
    # Test mode: shorter time, smaller batch, 10 samples
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=openai-base-test
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output_openai-base-test_%j.log
#SBATCH --error=error_openai-base-test_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Run OpenAI CLIP Base test (using -u for unbuffered output)
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
EOF
else
    # Full mode: full dataset run
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=openai-base
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_openai-base_%j.log
#SBATCH --error=error_openai-base_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Run OpenAI CLIP Base (using -u for unbuffered output)
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
EOF
fi

chmod +x "$JOB_SCRIPT"

# Submit job
if [ "$TEST_MODE" = true ]; then
    echo "📝 Submitting OpenAI CLIP Base TEST job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✅ Test job $JOB_ID submitted"
    echo ""
    echo "   Monitor: squeue -u \$USER"
    echo "   Logs: tail -f output_openai-base-test_${JOB_ID}.log"
    echo ""
    echo "⏱️  Note: Test mode runs 10 samples (~20-40 minutes)"
else
    echo "📝 Submitting OpenAI CLIP Base job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   ✅ Job $JOB_ID submitted"
    echo ""
    echo "   Monitor: squeue -u \$USER"
    echo "   Logs: tail -f output_openai-base_${JOB_ID}.log"
    echo ""
    echo "⏱️  Note: Full dataset run (4-8 hours expected)"
fi

# Clean up temp script
rm -f "$JOB_SCRIPT"
