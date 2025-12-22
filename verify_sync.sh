#!/bin/bash
# Verify that all files synced correctly to Newton
# Run this on Newton after syncing: ./verify_sync.sh

echo "=================================================================================="
echo "VERIFYING SYNC - Checking Key Files"
echo "=================================================================================="
echo ""

# Check key modified files
key_files=(
    "src/models/model_wrapper.py"
    "src/utils/evaluation.py"
    "src/main.py"
)

all_good=true

for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "✓ $file ($size bytes)"
    else
        echo "✗ $file MISSING!"
        all_good=false
    fi
done

echo ""
echo "=================================================================================="
echo "Checking for Recent Fixes"
echo "=================================================================================="
echo ""

# Check for BiomedCLIP fix (should have warning messages)
if grep -q "WARNING: Failed to load BiomedCLIP" src/models/model_wrapper.py 2>/dev/null; then
    echo "✓ BiomedCLIP fallback detection present"
else
    echo "✗ BiomedCLIP fallback detection NOT found"
    all_good=false
fi

# Check for disagreement_rate in step_results
if grep -q "disagreement_rate.*step_results" src/utils/evaluation.py 2>/dev/null || \
   grep -q "step_results\[.*\]\['disagreement_rate'\]" src/utils/evaluation.py 2>/dev/null; then
    echo "✓ disagreement_rate per modality fix present"
else
    echo "✗ disagreement_rate per modality fix NOT found"
    all_good=false
fi

echo ""
echo "=================================================================================="
if [ "$all_good" = true ]; then
    echo "✅ ALL CHECKS PASSED - Sync successful!"
    echo "=================================================================================="
    exit 0
else
    echo "❌ SOME CHECKS FAILED - Review above"
    echo "=================================================================================="
    exit 1
fi

