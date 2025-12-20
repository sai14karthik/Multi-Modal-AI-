#!/bin/bash
# Quick test script for Newton
# Usage: ./test_quick.sh

echo " Running quick test (10 samples)..."
echo ""

# Set thread limits (important for login node)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate venv
cd ~/Multi-Modal-AI
source venv/bin/activate

# Run test
python3 -m src.main --data_root data --modalities CT PET --model_arch clip --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --output_dir results --batch_size 1 --max_samples 10 --dataset_config data/dataset_config.yaml --class_names high_grade low_grade

echo ""
echo " Test complete! Check results/ directory for output."

