#!/bin/bash
# Verify code correctness before syncing to Newton
# Usage: ./verify_before_sync.sh

set -e

echo "=========================================="
echo "PRE-SYNC VERIFICATION"
echo "=========================================="
echo ""

ERRORS=0

# 1. Check Python syntax
echo "1. Checking Python syntax..."
python3 << 'PYEOF'
import ast
import sys

files_to_check = [
    'src/main.py',
    'src/models/model_wrapper.py',
    'src/utils/evaluation.py',
    'src/data/dataloader.py',
    'src/models/llava_runner.py',
    'src/models/llava_med_runner.py'
]

errors = []
for file in files_to_check:
    try:
        with open(file, 'r') as f:
            ast.parse(f.read())
        print(f"  ✅ {file}")
    except SyntaxError as e:
        print(f"  ❌ {file}: Syntax Error at line {e.lineno}: {e.msg}")
        errors.append((file, e))
    except FileNotFoundError:
        print(f"  ⚠️  {file}: Not found (may be optional)")
    except Exception as e:
        print(f"  ❌ {file}: {type(e).__name__}: {e}")
        errors.append((file, e))

if errors:
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "  ❌ Python syntax check FAILED"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ Python syntax check PASSED"
fi
echo ""

# 2. Check submission scripts don't have max_samples in full mode
echo "2. Checking submission scripts (full dataset mode)..."
if grep -q "max_samples" submit_top5_models.sh 2>/dev/null; then
    echo "  ❌ submit_top5_models.sh contains max_samples (should be removed for full dataset)"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ submit_top5_models.sh configured for full dataset"
fi

if grep -q "max_samples" submit_5_new_models.sh 2>/dev/null; then
    echo "  ❌ submit_5_new_models.sh contains max_samples (should be removed for full dataset)"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✅ submit_5_new_models.sh configured for full dataset"
fi
echo ""

# 3. Check required files exist
echo "3. Checking required files exist..."
REQUIRED_FILES=(
    "src/main.py"
    "src/models/model_wrapper.py"
    "src/utils/evaluation.py"
    "src/data/dataloader.py"
    "data/dataset_config.yaml"
    "requirements.txt"
    "submit_top5_models.sh"
    "submit_5_new_models.sh"
    "sync_to_newton.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file: MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# 4. Check submission scripts are executable
echo "4. Checking script permissions..."
SCRIPTS=(
    "submit_top5_models.sh"
    "submit_5_new_models.sh"
    "sync_to_newton.sh"
    "submit_sequential_eval.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "  ✅ $script (executable)"
        else
            echo "  ⚠️  $script (not executable, will fix)"
            chmod +x "$script"
        fi
    fi
done
echo ""

# 5. Verify model configurations
echo "5. Verifying model configurations..."
echo "  Top 5 models:"
grep -E "^[[:space:]]*\"[^\"]+\":" submit_top5_models.sh | head -5 | sed 's/^/    /'
echo ""
echo "  Next 5 models:"
grep -E "^[[:space:]]*\"[^\"]+\":" submit_5_new_models.sh | head -5 | sed 's/^/    /'
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - Ready to sync!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./sync_to_newton.sh"
    echo "  2. SSH to Newton and verify files"
    echo "  3. Run: ./submit_top5_models.sh"
    echo "  4. Run: ./submit_5_new_models.sh"
    exit 0
else
    echo "❌ VERIFICATION FAILED - $ERRORS error(s) found"
    echo "=========================================="
    echo ""
    echo "Please fix the errors before syncing."
    exit 1
fi
