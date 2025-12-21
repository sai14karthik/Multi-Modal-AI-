#!/usr/bin/env python3
"""
Comprehensive codebase verification script.
Checks for:
1. Syntax errors
2. Code structure correctness
3. Alignment with project goals
4. Common issues
"""

import os
import sys
import ast
import subprocess
from pathlib import Path

def check_syntax(filepath):
    """Check if a Python file has syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

def check_project_structure():
    """Verify project structure is correct."""
    required_files = [
        'src/main.py',
        'src/utils/evaluation.py',
        'src/models/model_wrapper.py',
        'src/data/dataloader.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return missing

def check_three_step_execution():
    """Verify three-step execution flow is present."""
    with open('src/main.py', 'r') as f:
        content = f.read()
    
    checks = {
        'Step 1 (CT)': 'Step 1:' in content and 'Processing {first_modality}' in content,
        'Step 2 (PET without context)': 'Step 2:' in content and 'without {first_modality} context' in content,
        'Step 3 (PET with context)': 'Step 3:' in content and 'with {first_modality} context' in content,
    }
    
    return checks

def check_certainty_focus():
    """Verify code focuses on certainty dynamics, not accuracy."""
    with open('src/utils/evaluation.py', 'r') as f:
        content = f.read()
    
    checks = {
        'Certainty metrics mentioned': 'certainty' in content.lower() or 'confidence' in content.lower(),
        'Project goal documented': 'certainty dynamics' in content.lower() or 'model behavior' in content.lower(),
        'Accuracy de-emphasized': 'accuracy near chance' in content.lower() or 'not optimize accuracy' in content.lower(),
    }
    
    return checks

def main():
    print("="*80)
    print("COMPREHENSIVE CODEBASE VERIFICATION")
    print("="*80)
    print()
    
    errors = []
    warnings = []
    
    # 1. Check syntax
    print("1. Checking syntax...")
    python_files = [
        'src/main.py',
        'src/utils/evaluation.py',
        'src/utils/__init__.py',
        'src/models/model_wrapper.py',
        'src/models/llava_runner.py',
        'src/models/llava_med_runner.py',
        'src/data/dataloader.py',
        'src/data/config.py',
    ]
    
    for file in python_files:
        if os.path.exists(file):
            is_valid, error = check_syntax(file)
            if not is_valid:
                errors.append(f"Syntax error in {file}: {error}")
                print(f"  ❌ {file}: {error}")
            else:
                print(f"  ✓ {file}")
        else:
            warnings.append(f"File not found: {file}")
            print(f"  ⚠ {file}: Not found")
    
    print()
    
    # 2. Check project structure
    print("2. Checking project structure...")
    missing = check_project_structure()
    if missing:
        errors.extend([f"Missing required file: {f}" for f in missing])
        for f in missing:
            print(f"  ❌ Missing: {f}")
    else:
        print("  ✓ All required files present")
    print()
    
    # 3. Check three-step execution
    print("3. Checking three-step execution flow...")
    steps = check_three_step_execution()
    all_present = all(steps.values())
    for step, present in steps.items():
        if present:
            print(f"  ✓ {step}")
        else:
            errors.append(f"Missing: {step}")
            print(f"  ❌ Missing: {step}")
    print()
    
    # 4. Check certainty focus
    print("4. Checking alignment with project goals...")
    certainty_checks = check_certainty_focus()
    for check, present in certainty_checks.items():
        if present:
            print(f"  ✓ {check}")
        else:
            warnings.append(f"Potential issue: {check}")
            print(f"  ⚠ {check}")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if errors:
        print(f"\n❌ ERRORS FOUND: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ No errors found!")
    
    if warnings:
        print(f"\n⚠ WARNINGS: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✓ No warnings!")
    
    print()
    
    if errors:
        print("❌ CODEBASE HAS ERRORS - PLEASE FIX BEFORE PROCEEDING")
        return 1
    elif warnings:
        print("⚠ CODEBASE HAS WARNINGS - REVIEW RECOMMENDED")
        return 0
    else:
        print("✓ CODEBASE IS CLEAN AND READY")
        return 0

if __name__ == '__main__':
    sys.exit(main())

