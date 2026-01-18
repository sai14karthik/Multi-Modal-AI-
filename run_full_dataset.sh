#!/bin/bash
# ============================================================================
# Run Full Dataset Evaluation on One Model
# ============================================================================
# Usage:
#   ./run_full_dataset.sh [MODEL_NAME] [MODEL_ARCH] [ORDER]
#
# Examples:
#   # Run BOTH orders (default)
#   ./run_full_dataset.sh
#   ./run_full_dataset.sh openai/clip-vit-base-patch32 clip BOTH
#
#   # Run single order
#   ./run_full_dataset.sh openai/clip-vit-large-patch14 clip CT_PET
#   ./run_full_dataset.sh openai/clip-vit-base-patch32 clip PET_CT
#
# ORDER options:
#   - BOTH: Run both CT→PET and PET→CT orders (default)
#   - CT_PET: CT first, then PET, then PET with CT context
#   - PET_CT: PET first, then CT, then CT with PET context
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL_NAME="${1:-openai/clip-vit-base-patch32}"
MODEL_ARCH="${2:-clip}"
ORDER="${3:-BOTH}"

# Parse order
if [ "$ORDER" == "CT_PET" ]; then
    FIRST_MOD="CT"
    SECOND_MOD="PET"
    RUN_BOTH=false
elif [ "$ORDER" == "PET_CT" ]; then
    FIRST_MOD="PET"
    SECOND_MOD="CT"
    RUN_BOTH=false
elif [ "$ORDER" == "BOTH" ] || [ -z "$3" ]; then
    RUN_BOTH=true
else
    echo "Error: ORDER must be 'CT_PET', 'PET_CT', or 'BOTH' (default)"
    echo "  CT_PET: CT → PET → PET with CT context"
    echo "  PET_CT: PET → CT → CT with PET context"
    echo "  BOTH: Run both CT_PET and PET_CT orders"
    exit 1
fi

# Validate model arch
if [ "$MODEL_ARCH" != "clip" ] && [ "$MODEL_ARCH" != "llava_med" ]; then
    echo "Error: MODEL_ARCH must be 'clip' or 'llava_med'"
    exit 1
fi

# Function to submit/run a single order
submit_single_order() {
    local FIRST=$1
    local SECOND=$2
    
    local MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
    local JOB_NAME="full_${FIRST}_${SECOND}_${MODEL_NAME_SAFE}"
    local JOB_SCRIPT="submit_full_tmp_${FIRST}_${SECOND}_$$.sh"
    
    # Determine batch size and time based on model
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        BATCH_SIZE=1
        TIME_LIMIT="48:00:00"
        MEM="64G"
        CPUS=8
    else
        BATCH_SIZE=16
        TIME_LIMIT="24:00:00"
        MEM="48G"
        CPUS=8
    fi
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=output_${JOB_NAME}_%j.log
#SBATCH --error=error_${JOB_NAME}_%j.log
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

set -e

echo "============================================================================"
echo "Full Dataset Evaluation Job"
echo "============================================================================"
echo "Order:      ${FIRST} → ${SECOND}"
echo "Model:      ${MODEL_NAME} (${MODEL_ARCH})"
echo "Job ID:     \$SLURM_JOB_ID"
echo "Node:       \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "============================================================================"
echo ""

cd ~/Multi-Modal-AI || cd "\$(dirname "\$0")"
PROJECT_DIR=\$(pwd)
echo "Project directory: \$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=\$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=\$HOME/.cache/huggingface
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EOF

    # Add LLaVA-Med specific setup if needed
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        cat >> "$JOB_SCRIPT" <<'LLAVA_EOF'
# Install LLaVA-Med dependencies
echo "Installing LLaVA-Med dependencies..."
pip install --quiet --upgrade "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99 || {
    echo "✗ Failed to install dependencies"
    exit 1
}
echo "✓ Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print('✓ Dependencies verified')" || {
    echo "✗ Dependency verification failed!"
    exit 1
}
LLAVA_EOF
    fi

    cat >> "$JOB_SCRIPT" <<EOF

mkdir -p results

# Run evaluation on FULL dataset (no --max_samples = all images)
echo ""
echo "Starting full dataset evaluation..."
echo "  Order: ${FIRST} → ${SECOND}"
echo "  Model: ${MODEL_NAME} (${MODEL_ARCH})"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Dataset: Full (all images, all patients)"
echo ""

python3 -u -m src.main \\
    --data_root data \\
    --modalities ${FIRST} ${SECOND} \\
    --model_arch ${MODEL_ARCH} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results \\
    --batch_size ${BATCH_SIZE} \\
    --dataset_config data/dataset_config.yaml \\
    --class_names high_grade low_grade \\
    --temperature 0.8 \\
    --aggressive_preprocess \\
    --no_progress

EXIT_CODE=\$?

echo ""
echo "============================================================================"
echo "Job completed with exit code: \$EXIT_CODE"
echo "End time: \$(date)"
echo "============================================================================"
echo ""

if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Full dataset evaluation completed successfully"
    echo ""
    echo "Results saved to: results/"
    echo "Result file: results/results_${MODEL_NAME_SAFE}_${FIRST}_${SECOND}.json"
    echo ""
    echo "To view results:"
    echo "  cat results/results_${MODEL_NAME_SAFE}_${FIRST}_${SECOND}.json | python3 -m json.tool | less"
else
    echo "✗ Full dataset evaluation failed"
    echo "Check error log: error_${JOB_NAME}_\${SLURM_JOB_ID}.log"
fi

exit \$EXIT_CODE
EOF

    chmod +x "$JOB_SCRIPT"
    
    echo "Submitting ${FIRST} → ${SECOND} job to SLURM..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "✓ Job $JOB_ID submitted for ${FIRST} → ${SECOND}"
    
    # Clean up temp script
    rm -f "$JOB_SCRIPT"
    
    echo "$JOB_ID"
}

# Function to run single order locally
run_single_order_local() {
    local FIRST=$1
    local SECOND=$2
    
    # Determine batch size
    if [ "$MODEL_ARCH" == "llava_med" ]; then
        BATCH_SIZE=1
    else
        BATCH_SIZE=8  # Smaller for local
    fi
    
    echo "Starting full dataset evaluation..."
    echo "  Order: ${FIRST} → ${SECOND}"
    echo "  Model: ${MODEL_NAME} (${MODEL_ARCH})"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Dataset: Full (all images, all patients)"
    echo ""
    
    python3 -u -m src.main \
        --data_root data \
        --modalities ${FIRST} ${SECOND} \
        --model_arch ${MODEL_ARCH} \
        --model_name ${MODEL_NAME} \
        --output_dir results \
        --batch_size ${BATCH_SIZE} \
        --dataset_config data/dataset_config.yaml \
        --class_names high_grade low_grade \
        --temperature 0.8 \
        --aggressive_preprocess \
        --no_progress
    
    return $?
}

# Main execution
echo "============================================================================"
echo "Full Dataset Evaluation"
echo "============================================================================"
echo "Model:      $MODEL_NAME"
echo "Arch:       $MODEL_ARCH"
if [ "$RUN_BOTH" = true ]; then
    echo "Orders:     CT → PET AND PET → CT (both directions)"
else
    echo "Order:      $FIRST_MOD → $SECOND_MOD"
fi
echo "Dataset:    Full dataset (all images, all patients)"
echo "============================================================================"
echo ""

# Check if running on Newton (has SLURM)
if [ -n "$SLURM_JOB_ID" ] || command -v sbatch &> /dev/null; then
    echo "Detected SLURM environment - submitting as job(s)..."
    echo ""
    
    if [ "$RUN_BOTH" = true ]; then
        # Submit both orders as separate jobs
        echo "Submitting CT → PET order..."
        JOB1_ID=$(submit_single_order "CT" "PET")
        echo ""
        
        echo "Submitting PET → CT order..."
        JOB2_ID=$(submit_single_order "PET" "CT")
        echo ""
        
        echo "============================================================================"
        echo "Both jobs submitted successfully!"
        echo "============================================================================"
        echo ""
        echo "Job IDs:"
        echo "  CT → PET: $JOB1_ID"
        echo "  PET → CT: $JOB2_ID"
        echo ""
        echo "Monitor jobs:"
        echo "  squeue -u \$USER"
        echo ""
        echo "View logs:"
        MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
        echo "  tail -f output_full_CT_PET_${MODEL_NAME_SAFE}_${JOB1_ID}.log"
        echo "  tail -f output_full_PET_CT_${MODEL_NAME_SAFE}_${JOB2_ID}.log"
        echo ""
        if [ "$MODEL_ARCH" == "llava_med" ]; then
            echo "⏱️  Estimated time: 8-12 hours per job (LLaVA-Med is slow)"
        else
            echo "⏱️  Estimated time: 4-8 hours per job (depends on dataset size)"
        fi
    else
        # Submit single order
        submit_single_order "$FIRST_MOD" "$SECOND_MOD"
        echo ""
        echo "Monitor job:"
        echo "  squeue -u \$USER"
        echo ""
        MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
        echo "View logs:"
        echo "  tail -f output_full_${FIRST_MOD}_${SECOND_MOD}_${MODEL_NAME_SAFE}_*.log"
        echo ""
        if [ "$MODEL_ARCH" == "llava_med" ]; then
            echo "⏱️  Estimated time: 8-12 hours (LLaVA-Med is slow)"
        else
            echo "⏱️  Estimated time: 4-8 hours (depends on dataset size)"
        fi
    fi
    
else
    echo "Running locally (not on Newton cluster)..."
    echo ""
    
    if [ "$RUN_BOTH" = true ]; then
        # Run both orders sequentially
        echo "Running CT → PET order..."
        echo "----------------------------------------"
        run_single_order_local "CT" "PET"
        EXIT1=$?
        echo ""
        
        if [ $EXIT1 -eq 0 ]; then
            echo "Running PET → CT order..."
            echo "----------------------------------------"
            run_single_order_local "PET" "CT"
            EXIT2=$?
            echo ""
            
            echo "============================================================================"
            if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
                MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
                echo "✓ Both orders completed successfully"
                echo ""
                echo "Results saved to:"
                echo "  - results/results_${MODEL_NAME_SAFE}_CT_PET.json"
                echo "  - results/results_${MODEL_NAME_SAFE}_PET_CT.json"
            else
                echo "✗ Some orders failed"
                [ $EXIT1 -ne 0 ] && echo "  - CT → PET failed (exit code: $EXIT1)"
                [ $EXIT2 -ne 0 ] && echo "  - PET → CT failed (exit code: $EXIT2)"
            fi
            echo "============================================================================"
            
            exit $((EXIT1 + EXIT2))
        else
            echo "✗ CT → PET failed, skipping PET → CT"
            exit $EXIT1
        fi
    else
        # Run single order
        run_single_order_local "$FIRST_MOD" "$SECOND_MOD"
        EXIT_CODE=$?
        
        echo ""
        if [ $EXIT_CODE -eq 0 ]; then
            MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/:' '_')
            echo "✓ Full dataset evaluation completed successfully"
            echo "Results saved to: results/results_${MODEL_NAME_SAFE}_${FIRST_MOD}_${SECOND_MOD}.json"
        else
            echo "✗ Full dataset evaluation failed with exit code: $EXIT_CODE"
        fi
        
        exit $EXIT_CODE
    fi
fi
