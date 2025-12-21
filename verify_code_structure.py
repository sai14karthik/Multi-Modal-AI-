#!/usr/bin/env python3
"""
Verify code structure and check for common issues.
Does not require data or model access.
"""

import sys
import ast
import importlib.util
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_imports(file_path):
    """Check if imports are valid (without actually importing)."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return True, imports
    except Exception as e:
        return False, str(e)

def verify_function_signatures():
    """Verify that key functions have correct signatures."""
    print("="*80)
    print("Verifying Function Signatures")
    print("="*80)
    
    # Check evaluation.py functions
    eval_file = Path('src/utils/evaluation.py')
    if not eval_file.exists():
        print(f"ERROR: {eval_file} not found!")
        return False
    
    with open(eval_file, 'r') as f:
        eval_code = f.read()
    
    required_functions = [
        'calculate_entropy',
        'calculate_probability_gap',
        'calculate_logit_magnitude',
        'analyze_certainty_metrics',
        'analyze_modality_agreement',
        'analyze_ct_context_influence',
        'evaluate_sequential_modalities',
        'print_evaluation_results'
    ]
    
    print("\nChecking required functions in evaluation.py:")
    all_found = True
    for func_name in required_functions:
        if f'def {func_name}' in eval_code:
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} - NOT FOUND")
            all_found = False
    
    return all_found

def main():
    """Run all verification checks."""
    print("="*80)
    print("Code Structure Verification")
    print("="*80)
    print("\nThis script verifies code structure without requiring data or models.")
    print("="*80)
    
    files_to_check = [
        'src/utils/evaluation.py',
        'src/models/model_wrapper.py',
        'src/main.py'
    ]
    
    all_ok = True
    
    # Check syntax
    print("\n" + "-"*80)
    print("Syntax Check")
    print("-"*80)
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            print(f"  ✗ {file_path} - FILE NOT FOUND")
            all_ok = False
            continue
        
        is_valid, error = check_syntax(path)
        if is_valid:
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - SYNTAX ERROR: {error}")
            all_ok = False
    
    # Check function signatures
    if not verify_function_signatures():
        all_ok = False
    
    # Check for common issues
    print("\n" + "-"*80)
    print("Checking for Common Issues")
    print("-"*80)
    
    eval_file = Path('src/utils/evaluation.py')
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_code = f.read()
        
        # Check for scipy import (should be removed)
        if 'from scipy' in eval_code or 'import scipy' in eval_code:
            print("  ⚠ WARNING: scipy import found (should use numpy-based entropy)")
        else:
            print("  ✓ No scipy imports (using numpy-based entropy)")
        
        # Check for numpy import
        if 'import numpy' in eval_code or 'from numpy' in eval_code:
            print("  ✓ numpy import found")
        else:
            print("  ✗ numpy import missing")
            all_ok = False
        
        # Check that probabilities_before_boosting is handled
        if 'probabilities_before_boosting' in eval_code:
            print("  ✓ probabilities_before_boosting handling found")
        else:
            print("  ⚠ WARNING: probabilities_before_boosting not found in evaluation.py")
        
        # Check that probabilities_array is used
        if 'probabilities_array' in eval_code:
            print("  ✓ probabilities_array handling found")
        else:
            print("  ⚠ WARNING: probabilities_array not found")
    
    # Check model_wrapper.py
    wrapper_file = Path('src/models/model_wrapper.py')
    if wrapper_file.exists():
        with open(wrapper_file, 'r') as f:
            wrapper_code = f.read()
        
        # Check that logits are returned
        if "'logits'" in wrapper_code or '"logits"' in wrapper_code:
            print("  ✓ logits return value found in model_wrapper")
        else:
            print("  ⚠ WARNING: logits return value not found")
        
        # Check that probabilities_before_boosting is returned
        if 'probabilities_before_boosting' in wrapper_code:
            print("  ✓ probabilities_before_boosting return value found")
        else:
            print("  ✗ probabilities_before_boosting return value missing")
            all_ok = False
    
    # Check main.py
    main_file = Path('src/main.py')
    if main_file.exists():
        with open(main_file, 'r') as f:
            main_code = f.read()
        
        # Check that certainty metrics are collected
        if 'probabilities_array' in main_code and 'logits' in main_code:
            print("  ✓ Certainty metrics collection found in main.py")
        else:
            print("  ⚠ WARNING: Certainty metrics collection may be incomplete")
        
        # Check that analyze_ct_context_influence is called
        if 'analyze_ct_context_influence' in main_code:
            print("  ✓ CT context influence analysis found")
        else:
            print("  ⚠ WARNING: CT context influence analysis not found")
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
        print("="*80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please review above")
        print("="*80)
        return 1

if __name__ == '__main__':
    sys.exit(main())

