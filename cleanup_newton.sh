#!/bin/bash
# Cleanup script for Newton cluster
# Removes old/unwanted files to match local clean state
# Run this on Newton: ./cleanup_newton.sh

echo "Cleaning up Newton directory..."
echo ""

# Remove old submission scripts
echo "Removing old submission scripts..."
rm -f submit_biomedclip.sh
rm -f submit_llava_med.sh
rm -f submit_llava_med_newton.sh
rm -f submit_openai_base.sh
rm -f submit_sequential_eval.sh

# Remove old test files
echo "Removing old test files..."
rm -f test_models.py
rm -f test_performance.py
rm -f test_quick.sh
rm -f test_with_samples.sh
rm -f quick_test_models.py

# Remove other unwanted files
echo "Removing other unwanted files..."
rm -f run_llava_med.py
rm -f builder.py
rm -f verify_sync.sh

# Remove pip install artifacts (strange files)
echo "Removing pip install artifacts..."
rm -f '=0.1.99'
rm -f '=0.20.0'
rm -f '=4.21.0'

# Move cmb_aml_config.yaml to data/ if it exists at root
if [ -f cmb_aml_config.yaml ] && [ ! -f data/cmb_aml_config.yaml ]; then
    echo "Moving cmb_aml_config.yaml to data/..."
    mv cmb_aml_config.yaml data/
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining files:"
ls -1 | grep -v "^data$\|^src$\|^results$\|^third_party$\|^venv$" | head -20
