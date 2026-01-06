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

# LLaVA-Med Evaluation Job
# This script runs the LLaVA-Med model evaluation on the full dataset

echo "=========================================="
echo "LLaVA-Med Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust based on Newton's setup)
# Note: Module loading is optional - venv may already have required packages
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || echo "⚠ Warning: python/3.10 module not found, using system Python"
module load cuda/11.8 2>/dev/null || echo "⚠ Warning: cuda/11.8 module not found, CUDA may be available via venv"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: venv not found, using system Python"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface

# Create output directory
mkdir -p results

# Run LLaVA-Med evaluation
echo ""
echo "Starting LLaVA-Med evaluation..."
echo "Model: microsoft/llava-med-v1.5-mistral-7b"
echo "Modalities: CT PET"
echo ""

python3 run_llava_med.py \
    --model-name microsoft/llava-med-v1.5-mistral-7b \
    --data_root data \
    --dataset_config data/dataset_config.yaml \
    --modalities CT PET \
    --device cuda \
    --batch_size 1 \
    --temperature 0.8 \
    --class-names high_grade low_grade \
    --output_dir results \
    --aggressive-preprocess

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

