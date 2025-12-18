#!/bin/bash
# Submit top 5 best models for medical imaging evaluation
# Usage: ./submit_top5_models.sh

echo "🚀 Submitting top 5 models..."
echo ""

# Top 5 models (best to good)
MODELS=(
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224:clip:biomedclip"
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K:clip:laion-huge"
    "openai/clip-vit-large-patch14:clip:openai-large"
    "openai/clip-vit-large-patch14-336:clip:openai-336"
    "microsoft/llava-med-v1.5-mistral-7b:llava_med:llava-med"
)

JOB_IDS=()

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_arch job_name <<< "$model_config"
    
    echo "📝 Submitting: $job_name"
    
    # Create temporary job script
    JOB_SCRIPT="submit_${job_name}_tmp.sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_${job_name}_%j.log
#SBATCH --error=error_${job_name}_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

python3 -m src.main --data_root data --modalities CT PET --model_arch ${model_arch} --model_name ${model_name} --output_dir results --batch_size 8 --dataset_config data/dataset_config.yaml --class_names high_grade low_grade --temperature 0.8 --aggressive_preprocess
EOF
    
    chmod +x "$JOB_SCRIPT"
    
    # Submit job
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    
    # Clean up temp script
    rm -f "$JOB_SCRIPT"
    
    JOB_IDS+=("$JOB_ID")
    echo "   ✅ Job $JOB_ID submitted"
    echo ""
done

echo "✅ Submitted ${#JOB_IDS[@]} jobs"
echo ""
echo "📊 Monitor: squeue -u \$USER"
echo "📄 Logs: tail -f output_*_*.log"

