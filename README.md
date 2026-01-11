# Multi-Modal-AI Project


```bash
python -m src.data.prepare_lung_pet_ct \
    --raw_root dataset/manifest-<id>/Lung-PET-CT-Dx \
    --metadata_csv dataset/manifest-<id>/metadata.csv \
    --clinical_excel dataset/statistics-clinical-20201221.xlsx \
    --data_root data \
    --dataset_name Lung-PET-CT-Dx \
    --label_strategy histology_grade \
    --ct_slice_interval 3 \
    --pet_slice_interval 1
```

This converts all CT/PET DICOM slices into PNGs under `data/Lung-PET-CT-Dx/{CT,PT}/<class>` and writes `metadata.csv` with patient IDs, modality identifiers, histology-based class labels (`high_grade`, `low_grade`), and deterministic train/val/test splits.



## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Use Medical-Specific Model
```bash
python -m src.main \
    --data_root data \
    --modalities CT PET \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir results \
    --batch_size 1 \
    --dataset_config data/dataset_config.yaml \
    --class_names high_grade low_grade
```

Or use the provided script:
```bash
bash run.sh
```

`run.sh` performs two steps:
1. Quick sanity check on 30 samples using CLIP ViT-L/14 (fast).
2. Full-dataset evaluation with BiomedCLIP (best accuracy).



### Options

- `--no_preprocess`: Disable medical image preprocessing (default: disabled)
- `--batch_size`: Increase for faster GPU processing (e.g., 8, 16, 32)
- `--temperature`: Temperature scaling (default: 1.0, try 0.5-2.0)
- `--no_weighted_ensemble`: Disable weighted prompt averaging (use simple mean)
- `--no_swap_test`: Disable dual strategy testing (use original swap only)
- `--model_arch`: Choose between `clip` (default) and `llava`
- `--model_name`: Try different CLIP/LLaVA checkpoints (e.g., `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, `liuhaotian/llava-v1.6-mistral-7b`)
- `--dataset_config`: Path to YAML describing modality folders/classes (defaults to legacy brain tumor)
- `--class_names`: Two class labels used for prompt construction (e.g., `high_grade low_grade`). These should match the folder names inside each modality.
- `--allow_single_modality`: Skip steps 2/3 and run CT-only inference when no second modality exists.
- `--split`: Filter slices by split defined in metadata (train/val/test)

## Results

Results are saved to `results/` directory in JSON format with accuracy metrics for each modality.

## Requirements

See `requirements.txt` for full dependencies.



