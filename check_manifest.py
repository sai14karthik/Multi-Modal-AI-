#!/usr/bin/env python3
"""Check what's in the NBIA manifest file."""

import pandas as pd
import csv
import os

manifest_path = 'data/Lung-PET-CT-Dx-NBIA-Manifest-122220-nbia-digest.xlsx'
metadata_path = 'data/Lung-PET-CT-Dx/metadata.csv'

print("="*70)
print("ANALYZING NBIA MANIFEST FILE")
print("="*70)

try:
    # Read the manifest Excel file
    print(f"\nReading: {manifest_path}")
    df = pd.read_excel(manifest_path)
    
    print(f"\nManifest file shape: {df.shape} (rows x columns)")
    print(f"Columns: {list(df.columns)}")
    
    # Find Subject ID column
    subject_col = None
    for col in df.columns:
        if 'subject' in col.lower() and 'id' in col.lower():
            subject_col = col
            break
    
    if not subject_col:
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        # Try common names
        for col_name in ['Subject ID', 'Subject', 'Patient ID', 'Patient']:
            if col_name in df.columns:
                subject_col = col_name
                break
    
    if subject_col:
        print(f"\nUsing column: {subject_col}")
        manifest_patients = set(str(p).strip().upper() for p in df[subject_col].unique() if pd.notna(p))
        print(f"Total unique patients in manifest: {len(manifest_patients)}")
        print(f"Sample patient IDs: {sorted(list(manifest_patients))[:10]}")
        
        # Check modality distribution
        if 'Modality' in df.columns or 'Series Modality' in df.columns:
            mod_col = 'Modality' if 'Modality' in df.columns else 'Series Modality'
            print(f"\nModality distribution:")
            print(df[mod_col].value_counts())
            
            # Check patients by modality
            ct_patients = set()
            pet_patients = set()
            for _, row in df.iterrows():
                patient_id = str(row[subject_col]).strip().upper() if pd.notna(row[subject_col]) else None
                modality = str(row[mod_col]).strip().upper() if pd.notna(row[mod_col]) else None
                if patient_id:
                    if modality == 'CT':
                        ct_patients.add(patient_id)
                    elif modality in ['PT', 'PET']:
                        pet_patients.add(patient_id)
            
            print(f"\nPatients with CT: {len(ct_patients)}")
            print(f"Patients with PET: {len(pet_patients)}")
            print(f"Patients with both: {len(ct_patients & pet_patients)}")
        
        # Compare with your dataset
        print(f"\n" + "="*70)
        print("COMPARING WITH YOUR DATASET")
        print("="*70)
        
        your_patients = set()
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get('patient_id', '').strip()
                    if pid:
                        your_patients.add(pid)
        
        print(f"\nYour dataset: {len(your_patients)} patients")
        print(f"NBIA manifest: {len(manifest_patients)} patients")
        
        missing_in_yours = manifest_patients - your_patients
        extra_in_yours = your_patients - manifest_patients
        
        print(f"\nPatients in manifest but NOT in your dataset: {len(missing_in_yours)}")
        if missing_in_yours:
            missing_sorted = sorted(list(missing_in_yours))
            print(f"  First 20: {missing_sorted[:20]}")
            if len(missing_sorted) > 20:
                print(f"  ... and {len(missing_sorted) - 20} more")
        
        print(f"\nPatients in your dataset but NOT in manifest: {len(extra_in_yours)}")
        if extra_in_yours:
            print(f"  {sorted(list(extra_in_yours))}")
        
        # Show first few rows
        print(f"\n" + "="*70)
        print("SAMPLE DATA FROM MANIFEST")
        print("="*70)
        print(df.head(10).to_string())
        
        print(f"\n" + "="*70)
        print("IMPORTANT NOTE")
        print("="*70)
        print(f"This manifest file contains:")
        print(f"  ✓ Patient IDs")
        print(f"  ✓ Modality information (CT/PT)")
        print(f"  ✓ Series information")
        print(f"  ✗ Histology grades (NOT in this file)")
        print(f"\nHistology grades come from a SEPARATE file:")
        print(f"  statistics-clinical-20201221.xlsx")
        print(f"\nThe preprocessing script needs BOTH files:")
        print(f"  1. This manifest (for DICOM file locations)")
        print(f"  2. Clinical Excel (for histology grade labels)")
        
    else:
        print("\nCould not find Subject ID column")
        print("\nFirst few rows:")
        print(df.head())
        
except ImportError as e:
    print(f"\nERROR: Missing library - {e}")
    print("Install with: pip install pandas openpyxl")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
