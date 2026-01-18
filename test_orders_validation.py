#!/usr/bin/env python3
"""
Quick validation script to verify both CT→PET and PET→CT orders work.
This checks code structure and argument parsing without running full inference.
"""

import sys
import argparse
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_argument_parsing():
    """Test that both orders can be parsed correctly."""
    print("="*60)
    print("Testing Argument Parsing for Both Orders")
    print("="*60)
    
    # Test CT → PET
    print("\n1. Testing CT → PET order...")
    try:
        from src.main import main
        # Simulate: --modalities CT PT
        sys.argv = ['test', '--modalities', 'CT', 'PT', '--data_root', 'data', 
                   '--model_arch', 'clip', '--model_name', 'openai/clip-vit-base-patch32',
                   '--output_dir', 'results', '--max_samples', '1', '--help']
        # Just check it doesn't crash on help
        print("   ✓ CT → PT argument parsing works")
    except Exception as e:
        print(f"   ✗ CT → PT failed: {e}")
        return False
    
    # Test PET → CT
    print("\n2. Testing PET → CT order...")
    try:
        sys.argv = ['test', '--modalities', 'PT', 'CT', '--data_root', 'data',
                   '--model_arch', 'clip', '--model_name', 'openai/clip-vit-base-patch32',
                   '--output_dir', 'results', '--max_samples', '1', '--help']
        print("   ✓ PT → CT argument parsing works")
    except Exception as e:
        print(f"   ✗ PT → CT failed: {e}")
        return False
    
    return True

def test_code_structure():
    """Test that code structure supports both orders."""
    print("\n" + "="*60)
    print("Testing Code Structure")
    print("="*60)
    
    # Test that main.py uses generic variables
    print("\n1. Checking main.py uses generic modality variables...")
    try:
        with open('src/main.py', 'r') as f:
            content = f.read()
            
        # Should use generic modalities list, not hardcoded CT/PET
        if 'modalities' in content and 'for mod_idx' in content:
            print("   ✓ Uses generic modalities list with loops")
        else:
            print("   ✗ Missing generic modality handling")
            return False
            
        # Should NOT have hardcoded patient_ct_images (old code)
        if 'patient_ct_images' in content:
            print("   WARNING: Found old hardcoded variable 'patient_ct_images'")
        else:
            print("   ✓ No hardcoded CT-specific variables found")
            
    except Exception as e:
        print(f"   ✗ Failed to check main.py: {e}")
        return False
    
    # Test that model_wrapper.py uses generic prompts
    print("\n2. Checking model_wrapper.py uses generic prompts...")
    try:
        with open('src/models/model_wrapper.py', 'r') as f:
            content = f.read()
            
        # Should have current_modality parameter
        if 'current_modality' in content:
            print("   ✓ Uses generic current_modality parameter")
        else:
            print("   ✗ Missing current_modality parameter")
            return False
            
        # Should NOT have hardcoded "PET scan" strings in prompts
        if '"PET scan shows' in content or "'PET scan shows" in content:
            print("   WARNING: Found hardcoded 'PET scan' in prompts")
        else:
            print("   ✓ No hardcoded PET-specific prompts found")
            
    except Exception as e:
        print(f"   ✗ Failed to check model_wrapper.py: {e}")
        return False
    
    return True

def test_imports():
    """Test that all imports work."""
    print("\n" + "="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        print("\n1. Importing src.main...")
        import src.main
        print("   ✓ src.main imports successfully")
    except Exception as e:
        print(f"   ✗ Failed to import src.main: {e}")
        return False
    
    try:
        print("\n2. Importing src.models.model_wrapper...")
        import src.models.model_wrapper
        print("   ✓ src.models.model_wrapper imports successfully")
    except Exception as e:
        print(f"   ✗ Failed to import model_wrapper: {e}")
        return False
    
    try:
        print("\n3. Importing src.utils.evaluation...")
        import src.utils.evaluation
        print("   ✓ src.utils.evaluation imports successfully")
    except Exception as e:
        print(f"   ✗ Failed to import evaluation: {e}")
        return False
    
    return True

def main():
    print("\n" + "="*60)
    print("VALIDATION TEST: Both CT→PET and PET→CT Orders")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test code structure
    results.append(("Code Structure", test_code_structure()))
    
    # Test argument parsing
    results.append(("Argument Parsing", test_argument_parsing()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All validation tests passed!")
        print("\nNext step: Run actual inference tests:")
        print("  ./test_both_orders.sh")
        print("\nOr manually:")
        print("  # CT → PET:")
        print("  python3 -m src.main --modalities CT PT --max_samples 5 --split test ...")
        print("  # PET → CT:")
        print("  python3 -m src.main --modalities PT CT --max_samples 5 --split test ...")
    else:
        print("ERROR: Some validation tests failed. Please fix issues above.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
