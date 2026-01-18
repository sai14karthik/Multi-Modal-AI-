#!/bin/bash
# Submit BiomedCLIP job to Newton (HPC cluster) - Full Dataset
# Usage: ./submit_biomedclip.sh
#
# This runs the same configuration as run.sh but on Newton cluster
# Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# Full dataset: All patients, all slices

echo "Submitting BiomedCLIP job to Newton (full dataset)..."
echo ""

# Create temporary job script
JOB_SCRIPT="submit_biomedclip_tmp.sh"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=biomedclip-full
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_biomedclip-full_%j.log
#SBATCH --error=error_biomedclip-full_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Run BiomedCLIP (same as run.sh)
echo "Starting BiomedCLIP evaluation (full dataset)..."
echo "Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
echo "Dataset: Full dataset (all patients, all slices)"
echo "Start time: \$(date)"
echo ""

python3 -u -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir results \
    --batch_size 8 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --aggressive_preprocess

EXIT_CODE=\$?

echo ""
echo "Job completed with exit code: \$EXIT_CODE"
echo "End time: \$(date)"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ BiomedCLIP evaluation completed successfully"
else
    echo "❌ BiomedCLIP evaluation failed"
fi
EOF

chmod +x "$JOB_SCRIPT"

# Submit job
echo "Submitting job..."
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')

# Clean up temp script
rm -f "$JOB_SCRIPT"

echo "✅ Job $JOB_ID submitted"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: tail -f output_biomedclip-full_${JOB_ID}.log"
echo ""
echo "Expected time: 12-24 hours (full dataset)"
echo "Results will be saved to: results/results_microsoft_BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.json"
