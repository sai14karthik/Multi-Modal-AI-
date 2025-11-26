# Multi-Modal-AI Project


## Dataset

- **CT Images**: 4,618 total (2,300 Healthy, 2,318 Tumor)
- **MRI Images**: 5,000 total (2,000 Healthy, 3,000 Tumor)
- **Total**: 9,618 images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Default Model)
```bash
python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name openai/clip-vit-large-patch14 \
    --output_dir results \
    --batch_size 1 \
    --max_samples 100
```

### Use Best Available Model (Recommended)
```bash
python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --output_dir results \
    --batch_size 1
```

### Use Medical-Specific Model
```bash
python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --output_dir results \
    --batch_size 1
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
- `--model_name`: Try different CLIP models (e.g., `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)

## Results

Results are saved to `results/` directory in JSON format with accuracy metrics for each modality.

## Requirements

See `requirements.txt` for full dependencies.



## Project Structure

```
.
├── src/
│   ├── main.py              # Main evaluation script
│   ├── models/
│   │   └── model_wrapper.py # CLIP model wrapper
│   ├── data/
│   │   └── dataloader.py    # Data loading utilities
│   └── utils/
│       └── evaluation.py    # Evaluation metrics
├── data/                    # Dataset directory
├── results/                 # Output results
├── run.sh                   # Execution script
└── requirements.txt         # Dependencies

```

