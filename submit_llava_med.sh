#!/bin/bash
# Submit LLaVA-Med job with dependency installation
# Usage: ./submit_llava_med.sh [test]
#   - Without arguments: Full dataset run
#   - With "test": Quick test with 10 samples

TEST_MODE=false
if [ "$1" == "test" ]; then
    TEST_MODE=true
    echo "TEST MODE: Running with 10 samples..."
else
    echo " Installing LLaVA-Med dependencies..."
fi
echo ""

# Create temporary job script that installs deps first
JOB_SCRIPT="submit_llava_med_tmp.sh"

if [ "$TEST_MODE" = true ]; then
    # Test mode: shorter time, smaller batch, 10 samples
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=llava-med-test
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=output_llava-med-test_%j.log
#SBATCH --error=error_llava-med-test_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Install missing dependencies
echo "Installing dependencies..."
pip install "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99
echo "Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print(' Dependencies verified')" || {
    echo " Dependency verification failed!"
    exit 1
}

# Run LLaVA-Med test (using -u for unbuffered output)
echo "Starting LLaVA-Med TEST evaluation (10 samples)..."
python3 -u -m src.main --data_root data --modalities CT PET --model_arch llava_med --model_name microsoft/llava-med-v1.5-mistral-7b --output_dir results --batch_size 1 --max_samples 10 --dataset_config data/dataset_config.yaml --class_names high_grade low_grade --temperature 0.8
EOF
else
    # Full mode: full dataset run
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=llava-med
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output_llava-med_%j.log
#SBATCH --error=error_llava-med_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Install missing dependencies
echo "Installing dependencies..."
pip install "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99
echo "Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print(' Dependencies verified')" || {
    echo " Dependency verification failed!"
    exit 1
}

# Run LLaVA-Med (using -u for unbuffered output)
echo "Starting LLaVA-Med evaluation..."
python3 -u -m src.main --data_root data --modalities CT PET --model_arch llava_med --model_name microsoft/llava-med-v1.5-mistral-7b --output_dir results --batch_size 8 --dataset_config data/dataset_config.yaml --class_names high_grade low_grade --temperature 0.8 --aggressive_preprocess
EOF
fi

chmod +x "$JOB_SCRIPT"

# Submit job
if [ "$TEST_MODE" = true ]; then
    echo "Submitting LLaVA-Med TEST job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "    Test job $JOB_ID submitted"
    echo ""
    echo " Monitor: squeue -u \$USER"
    echo " Logs: tail -f output_llava-med-test_${JOB_ID}.log"
    echo ""
    
else
    echo "  Submitting LLaVA-Med job..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "   Job $JOB_ID submitted"
    echo ""
    echo " Monitor: squeue -u \$USER"
    echo "  Logs: tail -f output_llava-med_${JOB_ID}.log"
    echo ""
    
fi

# Clean up temp script
rm -f "$JOB_SCRIPT"

