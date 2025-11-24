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

