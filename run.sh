#!/bin/bash
python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --output_dir results \
    --batch_size 1 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --max_samples 10



    python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_arch clip \
    --model_name openai/clip-rn50 \
    --output_dir results \
    --batch_size 1 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --temperature 0.8 \
    --max_samples 2000 \
    --split val


   python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_name  openai_clip-vit-base-patch32\
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade \
    --batch_size 1 \
    --temperature 0.5 \
    --aggressive_preprocess \
    --max_samples 15 \
    --output_dir result