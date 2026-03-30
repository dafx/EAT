#!/usr/bin/env python3
"""Download and prepare FSD50K manifests.

This script downloads the official FSD50K archives from Zenodo, reconstructs
the multipart zip archives when needed, extracts the audio and ground-truth
files, and writes fairseq-compatible manifests for pre-training and
fine-tuning.

Outputs:
  - train.tsv / valid.tsv / eval.tsv
  - train.lbl / valid.lbl / eval.lbl
  - label_descriptors.csv

The `.lbl` and `label_descriptors.csv` files are useful for classification
fine-tuning. Pre-training only needs the `.tsv` manifests.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ZENODO_RECORD_ID = "4060432"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

ARCHIVE_GROUPS = (
    "FSD50K.dev_audio",
    "FSD50K.eval_audio",
    "FSD50K.ground_truth",
)


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def fetch_record_files() -> List[dict]:
    with urllib.request.urlopen(ZENODO_API_URL) as response:
        payload = response.read().decode("utf-8")

    import json

    record = json.loads(payload)
    return record["files"]


def sort_part_key(name: str) -> Tuple[int, int, str]:
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "zip":
        return (1, 0, name)
    if suffix.startswith("z") and len(suffix) == 3 and suffix[1:].isdigit():
        return (0, int(suffix[1:]), name)
    return (2, 0, name)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    eprint(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(dest)


def group_assets(files: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for f in files:
        key = f["key"]
        for group in ARCHIVE_GROUPS:
            if key.startswith(group):
                grouped[group].append(f)
                break
    return grouped


def reconstruct_archive(parts: Sequence[Path], dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(".partial.zip")
    with open(tmp, "wb") as out:
        for part in parts:
            eprint(f"  appending {part.name}")
            with open(part, "rb") as inp:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
    tmp.replace(dest_zip)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    eprint(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def sanitize_label(label: str) -> str:
    return " ".join(label.replace(",", " ").split())


def read_vocabulary(vocab_path: Path, expected_labels: Sequence[str]) -> List[str]:
    with open(vocab_path, "r", newline="") as f:
        rows = [row for row in csv.reader(f) if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"No labels found in {vocab_path}")

    if len(rows[0]) == 1:
        candidate = [row[0].strip() for row in rows if row[0].strip()]
        return candidate

    expected = {label.strip() for label in expected_labels if label.strip()}
    best_idx = None
    best_score = -1
    for col_idx in range(len(rows[0])):
        values = []
        for row in rows:
            if col_idx < len(row):
                value = row[col_idx].strip()
                if value:
                    values.append(value)
        score = sum(1 for value in values if value in expected)
        if score > best_score:
            best_score = score
            best_idx = col_idx

    if best_idx is None:
        raise ValueError(f"Could not infer label column from {vocab_path}")

    labels = []
    for row in rows:
        if best_idx < len(row):
            value = row[best_idx].strip()
            if value and value not in {"label", "labels", "class", "classes", "mid", "mids"}:
                labels.append(value)
    if not labels:
        raise ValueError(f"No labels found in {vocab_path}")
    return labels


def write_label_descriptors(labels: Sequence[str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for idx, label in enumerate(labels):
            writer.writerow([idx, sanitize_label(label)])


def load_labels_map(labels: Sequence[str]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def parse_ground_truth(csv_path: Path, labels_map: Dict[str, int]) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["fname"].strip()
            label_field = row.get("labels", "").strip()
            split = row.get("split", "").strip()
            label_names = [x.strip() for x in label_field.split(",") if x.strip()]
            label_ids = [str(labels_map[name]) for name in label_names if name in labels_map]
            rows.append(
                {
                    "fname": fname,
                    "split": split,
                    "label_names": label_names,
                    "label_ids": label_ids,
                }
            )
    return rows


def audio_sample_count(path: Path) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.frames)


def write_manifest(
    out_path: Path,
    root_dir: Path,
    rows: Iterable[Tuple[str, int]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(str(root_dir.resolve()) + "\n")
        for rel_path, n_samples in rows:
            f.write(f"{rel_path} {n_samples}\n")


def write_labels(out_path: Path, rows: Iterable[Tuple[str, Sequence[str]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        for rel_path, label_ids in rows:
            f.write(f"{rel_path} {','.join(label_ids)}\n")


def prepare_manifests(
    dataset_root: Path,
    output_dir: Path,
    with_labels: bool,
) -> None:
    gt_dir = dataset_root / "FSD50K.ground_truth"
    dev_csv = gt_dir / "dev.csv"
    eval_csv = gt_dir / "eval.csv"
    vocab_csv = gt_dir / "vocabulary.csv"

    dev_rows_raw = parse_ground_truth(dev_csv, {})
    eval_rows_raw = parse_ground_truth(eval_csv, {})
    expected_label_names = sorted(
        {
            label
            for row in dev_rows_raw + eval_rows_raw
            for label in row["label_names"]
        }
    )

    labels = read_vocabulary(vocab_csv, expected_label_names)
    labels_map = load_labels_map(labels)

    write_label_descriptors(labels, output_dir / "label_descriptors.csv")

    dev_rows = parse_ground_truth(dev_csv, labels_map)
    eval_rows = parse_ground_truth(eval_csv, labels_map)

    splits = {
        "train": [r for r in dev_rows if r["split"] == "train"],
        "valid": [r for r in dev_rows if r["split"] == "val"],
        "eval": eval_rows,
    }

    for split, rows in splits.items():
        if split == "train" or split == "valid":
            audio_root = dataset_root / "FSD50K.dev_audio"
            audio_prefix = "FSD50K.dev_audio"
        else:
            audio_root = dataset_root / "FSD50K.eval_audio"
            audio_prefix = "FSD50K.eval_audio"

        manifest_rows: List[Tuple[str, int]] = []
        label_rows: List[Tuple[str, Sequence[str]]] = []

        for row in rows:
            rel_path = f"{audio_prefix}/{row['fname']}.wav"
            abs_path = dataset_root / rel_path
            manifest_rows.append((rel_path, audio_sample_count(abs_path)))
            label_rows.append((rel_path, row["label_ids"]))

        write_manifest(output_dir / f"{split}.tsv", dataset_root, manifest_rows)
        if with_labels:
            write_labels(output_dir / f"{split}.lbl", label_rows)


def download_and_extract(output_dir: Path, keep_archives: bool) -> None:
    files = fetch_record_files()
    grouped = group_assets(files)

    download_dir = output_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    for group in ARCHIVE_GROUPS:
        assets = grouped.get(group, [])
        if not assets:
            raise RuntimeError(f"Could not find Zenodo assets for {group}")

        assets = sorted(assets, key=lambda item: sort_part_key(item["key"]))
        part_paths: List[Path] = []
        for asset in assets:
            dest = download_dir / asset["key"]
            if not dest.exists():
                download_file(asset["links"]["self"], dest)
            part_paths.append(dest)

        assembled_zip = download_dir / f"{group}.zip"
        reconstruct_archive(part_paths, assembled_zip)
        extract_zip(assembled_zip, output_dir)

        if not keep_archives:
            for path in part_paths:
                path.unlink(missing_ok=True)
            assembled_zip.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FSD50K from Zenodo and create fairseq manifests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the dataset will be downloaded and prepared.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip parts after extraction.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip writing .lbl files (useful for pre-training only).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eprint(f"Preparing FSD50K in {output_dir}")
    download_and_extract(output_dir, keep_archives=args.keep_archives)
    prepare_manifests(output_dir, output_dir, with_labels=not args.no_labels)
    eprint("Done.")


if __name__ == "__main__":
    main()
