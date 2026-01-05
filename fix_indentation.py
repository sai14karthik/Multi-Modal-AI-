#!/usr/bin/env python3
"""Fix all indentation issues in main.py"""

with open('src/main.py', 'r') as f:
    lines = f.readlines()

# Fix lines 464-476 (else block)
fixes = {
    465: ('        ', ['# Original behavior']),
    466: ('        ', ['random.seed(42)']),
    467: ('        ', ['random.shuffle(first_mod_images)']),
    468: ('        ', ['if second_modality:']),
    469: ('            ', ['random.shuffle(second_mod_images)']),
    475: ('            ', ['second_mod_images = second_mod_images[:args.max_samples]']),
}

for line_num, (indent, keywords) in fixes.items():
    if line_num < len(lines):
        line = lines[line_num]
        if any(kw in line for kw in keywords):
            content = line.lstrip()
            lines[line_num] = indent + content

with open('src/main.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed all indentation issues")

