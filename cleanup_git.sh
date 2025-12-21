#!/bin/bash
# Script to remove files from git tracking that should be ignored
# Run this after updating .gitignore

echo "Cleaning up git tracking for files that should be ignored..."
echo ""

# Remove test_results directory from tracking
if git ls-files --error-unmatch test_results/ >/dev/null 2>&1; then
    echo "Removing test_results/ from git tracking..."
    git rm -r --cached test_results/ 2>/dev/null || true
fi

# Remove .egg-info directories
echo "Removing .egg-info directories from git tracking..."
find . -name "*.egg-info" -type d -exec git rm -r --cached {} \; 2>/dev/null || true

# Remove __pycache__ directories
echo "Removing __pycache__ directories from git tracking..."
find . -name "__pycache__" -type d -exec git rm -r --cached {} \; 2>/dev/null || true

# Remove .pyc files
echo "Removing .pyc files from git tracking..."
find . -name "*.pyc" -exec git rm --cached {} \; 2>/dev/null || true

# Remove log files
echo "Removing log files from git tracking..."
git rm --cached output_*.log error_*.log *.log 2>/dev/null || true

# Remove temporary submit scripts
echo "Removing temporary submit scripts from git tracking..."
git rm --cached submit_*_tmp.sh 2>/dev/null || true

# Remove third_party/llava (large regular directory, not a submodule)
echo "Removing third_party/llava from git tracking (large directory)..."
git rm -r --cached third_party/llava/ 2>/dev/null || true

# Note: third_party/llava-med is a git submodule, so it's tracked separately
# Don't remove submodules - they're managed by git submodule commands

echo ""
echo "Done! Review changes with: git status"
echo "If everything looks good, commit with: git commit -m 'Update .gitignore and remove unnecessary files'"

