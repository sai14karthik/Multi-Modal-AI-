#!/bin/bash

# ============================================================================
# BEST CONFIG FOR TESTING WHOLE DATASET
# ============================================================================
# This configuration processes ALL images from ALL patients in the dataset
# Optimized for accuracy and performance on medical images
# ============================================================================

python -m src.main \
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

# ============================================================================
# CONFIGURATION EXPLANATION:
# ============================================================================
# --model_name: BiomedCLIP (best for medical images, trained on PubMed)
# --batch_size: 8 (faster GPU processing, adjust if OOM: try 4 or 16)
# --max_samples: REMOVED (None = process ALL images from all patients)
# --temperature: 0.8 (optimal calibration for medical predictions)
# --aggressive_preprocess: Enabled (enhances contrast/sharpness for difficult cases)
# --output_dir: results (consistent output location)
#
# NEWTON HPC CLUSTER NOTES:
# - batch_size=8 is safe for most Newton GPUs (V100/A100)
# - If you have A100 (40GB+), you can try batch_size=16 for 2x speedup
# - Monitor GPU memory: nvidia-smi (if OOM, reduce to 4)
# - Progress bars are configured for Slurm log files (tqdm)
# - Results will be saved to results/ directory
# ============================================================================
#
# ALTERNATIVE CONFIGS (uncomment to use):
#
# 1. For faster processing (larger batch, but may use more GPU memory):
#    --batch_size 16
#
# 2. For specific split only (e.g., test set):
#    --split test
#
# 3. For validation set only:
#    --split val
#
# 4. For more conservative preprocessing:
#    Remove --aggressive_preprocess flag
#
# 5. For different model (larger, slower, potentially more accurate):
#    --model_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K
#
# ============================================================================