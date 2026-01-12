# Multi-Modal-AI Project

This project evaluates multi-modal AI models (CLIP, LLaVA, LLaVA-Med) on medical imaging datasets with sequential modality processing (CT → PET → CT+PET).

**Note:** The dataset is already prepared. PNG images and `metadata.csv` are available in `data/Lung-PET-CT-Dx/`.

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



