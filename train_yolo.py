"""
train_yolo.py
=============
Production-grade YOLOv8 training and EDA pipeline for the Smart Donation Box.

Features:
  • Comprehensive Exploratory Data Analysis (EDA) module.
  • POSIX-safe dataset validation & dynamic data.yaml generation.
  • Currency-specific augmentation & hyperparameter tuning.
  • Automatic alignment with detector.py (models/yolov8n_currency_best.pt).
  • Optional ONNX export & automatic config.json metadata synchronization.

Usage:
  # Run full training (100 epochs, default)
  python train_yolo.py

  # Run EDA and save visual analysis plots to runs/eda/
  python train_yolo.py --eda-only

  # Custom training with EDA and ONNX export
  python train_yolo.py --eda --epochs 120 --batch 16 --export-onnx
"""

import os
import sys
import glob
import json
import shutil
import argparse
import datetime
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Class Mapping aligned with Indian Currency Dataset
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAMES = {
    0: "10",
    1: "100",
    2: "20",
    3: "200",
    4: "50",
    5: "500",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 EDA & Training Pipeline for Indian Currency Recognition"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of total epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (default: yolov8n.pt)")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for GPU, 'cpu' for CPU (default: auto)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows safety)")
    parser.add_argument("--eda", action="store_true", help="Perform EDA and save summary plots before training")
    parser.add_argument("--eda-only", action="store_true", help="Perform EDA only and exit without training")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument(
        "--dest-name",
        type=str,
        default="yolov8n_currency_best.pt",
        help="Target filename inside models/ directory (default: yolov8n_currency_best.pt)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Exploratory Data Analysis (EDA) Module
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(dataset_dir: str, output_dir: str):
    """Parses dataset labels and generates statistical summaries and plots."""
    print("\n" + "=" * 65)
    print("🔍 Running Exploratory Data Analysis (EDA)...")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)
    records = []
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")

        img_files = glob.glob(os.path.join(img_dir, "*.*"))
        for img_p in img_files:
            stem = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, stem + ".txt")

            try:
                with Image.open(img_p) as im:
                    w, h = im.size
            except Exception:
                w, h = 640, 640

            if os.path.exists(lbl_p):
                with open(lbl_p, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        records.append({
                            "split": split, "img_name": os.path.basename(img_p),
                            "class_id": -1, "class_name": "Background",
                            "width": w, "height": h, "box_area": 0.0, "aspect_ratio": 0.0
                        })
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            records.append({
                                "split": split,
                                "img_name": os.path.basename(img_p),
                                "class_id": cid,
                                "class_name": DATASET_NAMES.get(cid, str(cid)),
                                "width": w,
                                "height": h,
                                "box_area": round(bw * bh, 4),
                                "aspect_ratio": round((bw * w) / (bh * h), 2) if (bh * h) > 0 else 0.0,
                            })

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ No annotations found for EDA.")
        return

    # 1. Dataset Breakdown Table
    print(f"📊 Total Bounding Boxes: {len(df)} across {df['img_name'].nunique()} unique images")
    split_counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    print("\n--- Class Counts by Split ---")
    print(split_counts)

    # 2. Plot Class Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df[df["class_name"] != "Background"],
        x="class_name",
        hue="split",
        palette="Blues_d",
        order=["10", "20", "50", "100", "200", "500"]
    )
    plt.title("Denomination Distribution across Splits", fontsize=13, fontweight="bold")
    plt.xlabel("Denomination (₹)", fontsize=11)
    plt.ylabel("Annotations Count", fontsize=11)
    plt.legend(title="Split")
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()

    # 3. Plot Bounding Box Geometry
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(
        data=df[df["class_name"] != "Background"],
        x="box_area",
        hue="class_name",
        multiple="stack",
        ax=axes[0],
        palette="turbo"
    )
    axes[0].set_title("Normalized Box Area Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Relative Area (w * h)")

    sns.boxplot(
        data=df[df["class_name"] != "Background"],
        x="class_name",
        y="aspect_ratio",
        hue="class_name",
        legend=False,
        ax=axes[1],
        palette="Set2",
        order=["10", "20", "50", "100", "200", "500"]
    )
    axes[1].set_title("Aspect Ratio (Width / Height)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Denomination (₹)")

    plt.tight_layout()
    geom_plot_path = os.path.join(output_dir, "bounding_box_geometry.png")
    plt.savefig(geom_plot_path, dpi=200)
    plt.close()

    print(f"✅ Saved EDA plots to: {output_dir}")
    print("   • class_distribution.png")
    print("   • bounding_box_geometry.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML Generator
# ─────────────────────────────────────────────────────────────────────────────
def prepare_yaml(workspace_dir: str) -> str:
    """Validate dataset structure and generate a fresh, absolute-path data.yaml."""
    dataset_dir = os.path.join(workspace_dir, "data", "Indian currency")
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    valid_img_dir = os.path.join(dataset_dir, "valid", "images")

    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(
            f"Dataset train images not found at: {train_img_dir}\n"
            f"Please ensure the dataset is placed at data/Indian currency/"
        )

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    posix_path = dataset_dir.replace(os.sep, "/")

    yaml_lines = [
        f'path: "{posix_path}"',
        "train: train/images",
        "val: valid/images",
    ]

    test_img_dir = os.path.join(dataset_dir, "test", "images")
    if os.path.exists(test_img_dir):
        yaml_lines.append("test: test/images")

    yaml_lines.append("\nnames:")
    for k, v in DATASET_NAMES.items():
        yaml_lines.append(f'  {k}: "{v}"')

    yaml_content = "\n".join(yaml_lines) + "\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ Generated dataset YAML at: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Update Helper
# ─────────────────────────────────────────────────────────────────────────────
def update_model_config(
    workspace_dir: str,
    map50: float,
    map50_95: float,
    model_name: str,
    img_size: int,
    epochs: int,
):
    """Keep models/config.json synchronized with latest training metrics."""
    config_path = os.path.join(workspace_dir, "models", "config.json")
    config = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}

    config["yolo_map50"] = round(float(map50), 4)
    config["yolo_map50_95"] = round(float(map50_95), 4)
    config["yolo_img_size"] = img_size
    config["last_trained"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["trained_epochs"] = epochs
    config["base_model"] = model_name

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Updated metadata in: {config_path}")
    except Exception as e:
        print(f"⚠️ Could not update config.json: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(workspace_dir, "data", "Indian currency")
    models_dir = os.path.join(workspace_dir, "models")
    runs_dir = os.path.join(workspace_dir, "runs")
    eda_dir = os.path.join(runs_dir, "eda")

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("🪙  Smart Donation Box — YOLOv8 Currency Trainer")
    print("=" * 65)

    # 1. Run EDA if requested
    if args.eda or args.eda_only:
        run_eda(dataset_dir, eda_dir)
        if args.eda_only:
            print("\n✨ EDA completed! Exiting (--eda-only specified).")
            return

    # 2. Device Setup
    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    device_label = "NVIDIA CUDA GPU" if device == 0 or device == "0" else "CPU"
    print(f"🖥️  Compute Device: {device} ({device_label})")

    # 3. YAML Preparation
    try:
        yaml_path = prepare_yaml(workspace_dir)
    except Exception as e:
        print(f"❌ Error setting up dataset YAML: {e}")
        sys.exit(1)

    # 4. Model Initialization
    print(f"📦 Loading base model: {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Failed to load model '{args.model}': {e}")
        sys.exit(1)

    # 5. Training
    train_runs_dir = os.path.join(runs_dir, "train")
    exp_name = "indian_currency_yolov8"

    print("\n🚀 Commencing YOLOv8 training with tailored currency augmentations...")
    print(f"   • Epochs: {args.epochs} (Patience: {args.patience})")
    print(f"   • Batch size: {args.batch} | Resolution: {args.imgsz}px")
    print(f"   • Optimizer: AdamW | Cosine LR Scheduler: True")

    try:
        train_results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            # Optimizer & Learning Rate
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=3.0,
            # Banknote Augmentations
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.4,
            shear=2.0,
            perspective=0.0005,
            fliplr=0.5,
            mosaic=0.7,
            mixup=0.1,
            close_mosaic=10,
            # Outputs
            save=True,
            project=train_runs_dir,
            name=exp_name,
            exist_ok=True,
            verbose=True,
        )
        print("\n🎉 Training finished successfully!")

    except Exception as e:
        print(f"\n❌ Training interrupted or failed: {e}")
        sys.exit(1)

    # 6. Validation Assessment
    print("\n📊 Evaluating best weights on validation set...")
    best_weights_path = os.path.join(train_runs_dir, exp_name, "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        print(f"⚠️ Warning: Best weights not found at {best_weights_path}")
        return

    best_model = YOLO(best_weights_path)
    val_results = best_model.val(data=yaml_path, imgsz=args.imgsz, device=device, verbose=False)

    map50 = getattr(val_results.box, "map50", 0.0)
    map50_95 = getattr(val_results.box, "map", 0.0)
    print(f"   • Validation mAP@0.50:      {map50:.4f} ({map50*100:.2f}%)")
    print(f"   • Validation mAP@0.50:0.95: {map50_95:.4f} ({map50_95*100:.2f}%)")

    # 7. Synchronize weights with detector.py
    dest_path = os.path.join(models_dir, args.dest_name)
    shutil.copy(best_weights_path, dest_path)
    print(f"✅ Best weights saved and aligned with detector.py:")
    print(f"   -> {dest_path}")

    # 8. Optional ONNX export
    if args.export_onnx:
        print("\n⚡ Exporting model to ONNX format for accelerated runtime...")
        try:
            onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
            dest_onnx_name = args.dest_name.replace(".pt", ".onnx")
            dest_onnx_path = os.path.join(models_dir, dest_onnx_name)
            if os.path.exists(onnx_path):
                shutil.copy(onnx_path, dest_onnx_path)
                print(f"✅ ONNX model exported: {dest_onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")

    # 9. Update config metadata
    update_model_config(
        workspace_dir=workspace_dir,
        map50=map50,
        map50_95=map50_95,
        model_name=args.model,
        img_size=args.imgsz,
        epochs=args.epochs,
    )

    print("\n" + "=" * 65)
    print("✨ All steps completed! You can now launch or refresh Streamlit:")
    print("   streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
