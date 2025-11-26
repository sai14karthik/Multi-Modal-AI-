#!/bin/bash


python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name openai/clip-vit-large-patch14 \
    --output_dir results \
    --batch_size 1 \
    --max_samples 30

python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir results \
    --batch_size 8