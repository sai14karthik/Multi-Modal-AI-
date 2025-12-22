#!/bin/bash
# Cleanup script to remove unwanted files from Stokes cluster
# Removes log files, cache, temporary files, etc.

echo "Cleaning up unwanted files on Stokes cluster..."
echo ""

# Remove all log files
echo "Removing log files..."
rm -f output_*.log error_*.log *.log 2>/dev/null
echo "  ✓ Removed log files"

# Remove temporary submit scripts
echo "Removing temporary submit scripts..."
rm -f submit_*_tmp.sh 2>/dev/null
echo "  ✓ Removed temporary scripts"

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "  ✓ Removed Python cache"

# Remove .egg-info directories
echo "Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "  ✓ Removed .egg-info directories"

# Remove test results (if you want to keep them, comment this out)
echo "Removing test_results directory..."
rm -rf test_results/ 2>/dev/null
echo "  ✓ Removed test_results"

# Remove any .DS_Store files (macOS)
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null
echo "  ✓ Removed .DS_Store files"

# Remove temporary files
echo "Removing temporary files..."
rm -f *.tmp *.temp 2>/dev/null
echo "  ✓ Removed temporary files"

# Remove builder.py if it's a temporary file (check first)
if [ -f "builder.py" ] && [ ! -f "src/models/builder.py" ]; then
    echo "Removing builder.py (appears to be temporary)..."
    rm -f builder.py 2>/dev/null
    echo "  ✓ Removed builder.py"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining files:"
ls -la | grep -v "^d" | head -20

