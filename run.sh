#!/bin/bash


python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_name openai/clip-vit-large-patch14 \
    --output_dir results \
    --batch_size 1 \
    --max_samples 30 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade

python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir results \
    --batch_size 8 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade