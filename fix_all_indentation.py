#!/usr/bin/env python3
"""Comprehensive fix for all indentation issues in main.py"""

with open('src/main.py', 'r') as f:
    lines = f.readlines()

# Fix lines 466-469 (else block)
lines[465] = '        random.seed(42)\n'
lines[466] = '        random.shuffle(first_mod_images)\n'
lines[467] = '        if second_modality:\n'
lines[468] = '            random.shuffle(second_mod_images)\n'

# Fix line 476
lines[475] = '            second_mod_images = second_mod_images[:args.max_samples]\n'

# Fix line 972
lines[971] = '        evaluation_results = evaluate_sequential_modalities(results, args.modalities)\n'

# Remove the problematic continue statement if patient_id check is outside loop
# Check lines 665-669 and fix them to be inside the for loop
for i in range(665, 670):
    if i < len(lines):
        line = lines[i]
        if line.strip():
            # These should all be indented 12 spaces (inside for loop)
            content = line.lstrip()
            if 'if case_id not in results:' in content:
                lines[i] = '            ' + content
            elif 'results[case_id] = []' in content:
                lines[i] = '                ' + content
            elif 'try:' in content:
                lines[i] = '            ' + content

with open('src/main.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed all indentation issues")

