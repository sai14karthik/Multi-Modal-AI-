"""
Prepare Lung-PET-CT-Dx dataset for the sequential modality pipeline.

Converts CT and PET DICOM series into PNG slices, generates metadata with patient-level
splits, and derives binary labels from clinical spreadsheets (histology grade or smoking).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm


def assign_split(patient_id: str) -> str:
    """Deterministic train/val/test split from patient hash."""
    digest = hashlib.md5(patient_id.encode("utf-8")).hexdigest()
    value = int(digest[:4], 16) / 0xFFFF
    if value < 0.7:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def window_image(array: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - (width / 2)
    upper = center + (width / 2)
    clipped = np.clip(array, lower, upper)
    scaled = (clipped - lower) / max(width, 1e-6)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * 255).astype(np.uint8)


def normalize_pet_image(array: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.nan_to_num(array.astype(np.float32))
    upper = np.percentile(arr, percentile)
    if upper <= 0:
        upper = arr.max()
    if upper <= 0:
        upper = 1.0
    arr = np.clip(arr, 0, upper) / upper
    return (arr * 255).astype(np.uint8)


def read_dicom_array(dicom_path: str) -> Tuple[np.ndarray, pydicom.dataset.Dataset]:
    ds = pydicom.dcmread(dicom_path)
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    array = array * slope + intercept
    return array, ds


def sanitize_patient_id(subject_id: str) -> str:
    if not subject_id:
        return ""
    subject_id = subject_id.strip()
    match = re.search(r"([A-Z]\d{4})", subject_id.upper())
    if match:
        return match.group(1)
    if "-" in subject_id:
        return subject_id.split("-")[-1].upper()
    return subject_id.upper()


def map_histology_label(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text or text in {"NAN", "NA"}:
        return None
    if any(ch in text for ch in ["3", "4"]):
        return "high_grade"
    if any(ch in text for ch in ["1", "2"]):
        return "low_grade"
    return None


def map_smoking_label(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(int(value)).strip()
    if text == "1":
        return "smoker"
    if text == "0":
        return "non_smoker"
    return None


def build_patient_label_map(
    clinical_df: pd.DataFrame,
    strategy: str,
) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for _, row in clinical_df.iterrows():
        patient_raw = str(row.get("NewPatientID", "")).strip()
        if not patient_raw or patient_raw.lower() == "nan":
            continue
        patient_id = patient_raw.upper()
        if strategy == "histology_grade":
            label = map_histology_label(row.get("Histopathological grading"))
        else:
            label = map_smoking_label(row.get("Smoking History"))
        if label:
            label_map[patient_id] = label
    return label_map


def gather_dicom_files(series_path: str) -> List[str]:
    pattern = os.path.join(series_path, "*.dcm")
    files = glob.glob(pattern)
    if not files:
        files = [
            f for f in glob.glob(os.path.join(series_path, "*"))
            if f.lower().endswith(".dcm")
        ]
    ordered: List[Tuple[int, str]] = []
    for path in files:
        order = len(ordered)
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            order = int(getattr(header, "InstanceNumber", order))
        except Exception:
            pass
        ordered.append((order, path))
    ordered.sort(key=lambda item: item[0])
    return [fp for _, fp in ordered]


def resolve_series_path(raw_root: str, relative_path: str) -> Optional[str]:
    rel_clean = relative_path.lstrip("./")
    candidates = [
        os.path.normpath(os.path.join(raw_root, rel_clean)),
        os.path.normpath(os.path.join(os.path.dirname(raw_root), rel_clean)),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def convert_series(
    series_path: str,
    modality: str,
    output_dir: str,
    patient_id: str,
    class_name: str,
    split: str,
    series_uid: str,
    slice_interval: int,
    ct_window: Tuple[float, float],
    pet_percentile: float,
    data_root: str,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    dicom_files = gather_dicom_files(series_path)
    rows: List[Dict] = []
    series_suffix = re.sub(r"[^0-9a-zA-Z]+", "", series_uid.split(".")[-1])
    for idx, dicom_path in enumerate(dicom_files):
        if slice_interval > 1 and idx % slice_interval != 0:
            continue
        try:
            array, _ = read_dicom_array(dicom_path)
        except Exception as exc:
            print(f"Warning: failed to read {dicom_path}: {exc}")
            continue
        if modality == "CT":
            processed = window_image(array, ct_window[0], ct_window[1])
        else:
            processed = normalize_pet_image(array, pet_percentile)
        image = Image.fromarray(processed).convert("L")
        file_stub = f"{patient_id}_{modality.lower()}_{series_suffix}_{idx:04d}"
        filename = f"{file_stub}.png"
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)
        rows.append({
            "image_path": os.path.normpath(os.path.relpath(save_path, data_root)),
            "patient_id": patient_id,
            "modality": modality,
            "class_name": class_name,
            "split": split,
            "series_uid": series_uid,
            "slice_index": idx,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Lung-PET-CT-Dx dataset.")
    parser.add_argument("--raw_root", required=True, help="Path to Lung-PET-CT-Dx directory inside manifest.")
    parser.add_argument("--metadata_csv", required=True, help="Path to NBIA metadata CSV.")
    parser.add_argument("--clinical_excel", required=True, help="Path to statistics-clinical-20201221.xlsx.")
    parser.add_argument("--data_root", required=True, help="Base data directory.")
    parser.add_argument("--dataset_name", default="Lung-PET-CT-Dx", help="Output dataset folder name.")
    parser.add_argument("--label_strategy", choices=["histology_grade", "smoking"], default="histology_grade")
    parser.add_argument("--ct_slice_interval", type=int, default=3, help="Keep every Nth CT slice.")
    parser.add_argument("--pet_slice_interval", type=int, default=1, help="Keep every Nth PET slice.")
    parser.add_argument("--ct_window", nargs=2, type=float, default=[-600.0, 1500.0], metavar=("CENTER", "WIDTH"))
    parser.add_argument("--pet_percentile", type=float, default=99.5, help="Upper percentile for PET normalization.")
    parser.add_argument("--limit_patients", type=int, default=None, help="Optional cap on patients for quick tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = os.path.abspath(args.raw_root)
    data_root = os.path.abspath(args.data_root)
    output_root = os.path.join(data_root, args.dataset_name)
    os.makedirs(output_root, exist_ok=True)

    clinical_df = pd.read_excel(args.clinical_excel)
    label_map = build_patient_label_map(clinical_df, args.label_strategy)
    class_names = sorted(set(label_map.values()))
    if len(class_names) < 2:
        raise RuntimeError("Label mapping did not produce two classes. Check clinical spreadsheet or strategy.")
    print(f"Discovered classes: {class_names} ({len(label_map)} labeled patients)")

    meta_df = pd.read_csv(args.metadata_csv)
    meta_df = meta_df[meta_df["Modality"].isin(["CT", "PT"])].copy()
    if meta_df.empty:
        raise RuntimeError("No CT/PT series found in metadata CSV.")

    metadata_rows: List[Dict] = []
    ct_count = 0
    pet_count = 0
    patient_counter = 0

    subject_groups = meta_df.groupby("Subject ID")
    for subject_id, group in tqdm(subject_groups, desc="Patients"):
        patient_code = sanitize_patient_id(subject_id)
        label = label_map.get(patient_code)
        if not label:
            continue
        split = assign_split(patient_code)
        for _, row in group.iterrows():
            modality = row["Modality"]
            rel_path = row["File Location"]
            series_path = resolve_series_path(raw_root, rel_path)
            if series_path is None:
                continue
            modality_dir = os.path.join(output_root, modality, label)
            slice_interval = args.ct_slice_interval if modality == "CT" else args.pet_slice_interval
            series_uid = str(row["Series UID"])
            rows = convert_series(
                series_path=series_path,
                modality=modality,
                output_dir=modality_dir,
                patient_id=patient_code,
                class_name=label,
                split=split,
                series_uid=series_uid,
                slice_interval=max(1, slice_interval),
                ct_window=tuple(args.ct_window),
                pet_percentile=args.pet_percentile,
                data_root=data_root,
            )
            metadata_rows.extend(rows)
            if modality == "CT":
                ct_count += len(rows)
            else:
                pet_count += len(rows)
        patient_counter += 1
        if args.limit_patients and patient_counter >= args.limit_patients:
            break

    if not metadata_rows:
        raise RuntimeError("No slices were converted. Verify labels and paths.")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(output_root, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved {len(metadata_rows)} slices ({ct_count} CT, {pet_count} PET) to {metadata_path}")


if __name__ == "__main__":
    main()

