#!/bin/bash
# Sync code changes to Newton cluster
# Usage: ./sync_to_newton.sh

echo "Syncing code to Newton..."
echo ""

# Sync source code and config files (excluding data and results)
rsync -avz --progress \
  --exclude 'data/Lung-PET-CT-Dx' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.DS_Store' \
  --exclude 'results/' \
  --exclude 'result/' \
  --exclude 'test_results/' \
  --exclude '*.log' \
  --exclude '.pytest_cache/' \
  --exclude '.cache/' \
  --exclude '.env' \
  --exclude '.env.local' \
  src \
  requirements.txt \
  README.md \
  data/dataset_config.yaml \
  .env.example \
  submit_top5_models.sh \
  submit_5_new_models.sh \
  submit_single_model.sh \
  test_single_model_local.sh \
  sync_to_newton.sh \
  third_party \
  sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/

echo ""
echo "Code sync complete!"
echo ""
echo "To sync dataset separately, run:"
echo "  rsync -avz --progress data/Lung-PET-CT-Dx/ sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/data/Lung-PET-CT-Dx/"

