# Multi-Modal-AI

Sequential modality evaluation for brain tumor classification using CLIP-based zero-shot classification.

## Overview

This project evaluates multimodal AI models (CT, MRI, and CT+MRI) for binary brain tumor classification (Healthy vs. Tumor) using zero-shot CLIP models.

## Features

- Sequential modality evaluation: MRI → CT → CT+MRI mix
- Support for multiple CLIP models (OpenAI CLIP, BiomedCLIP, etc.)
- Batch processing support
- Comprehensive evaluation metrics

## Dataset

- **CT Images**: 4,618 total (2,300 Healthy, 2,318 Tumor)
- **MRI Images**: 5,000 total (2,000 Healthy, 3,000 Tumor)
- **Total**: 9,618 images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main \
    --data_root data \
    --modalities CT MRI \
    --model_name openai/clip-vit-large-patch14 \
    --output_dir results \
    --batch_size 1 \
    --max_samples 100
```

Or use the provided script:
```bash
bash run.sh
```

## Results

Results are saved to `results/` directory in JSON format with accuracy metrics for each modality.

## Requirements

See `requirements.txt` for full dependencies.

## Supported Models

- `openai/clip-vit-large-patch14`
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-base-patch16`
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- And other CLIP-compatible models

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

