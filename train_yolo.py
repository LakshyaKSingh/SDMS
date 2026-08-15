import os
import shutil
import torch
from ultralytics import YOLO

def main():
    # 1. Define paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(workspace_dir, "data", "Indian currency")
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    print("--- Smart Donation Box YOLOv8 Trainer ---")
    print(f"Dataset location: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}!")
        return
        
    # 2. Write data.yaml file for YOLOv8
    yaml_content = f"""path: "{dataset_dir.replace(os.sep, '/')}"
train: train/images
val: valid/images
test: test/images

names:
  0: "10"
  1: "100"
  2: "20"
  3: "200"
  4: "50"
  5: "500"
"""
    try:
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        print(f"Generated dataset YAML configuration at: {yaml_path}")
    except Exception as e:
        print(f"Error generating YAML file: {e}")
        return
        
    # 3. Choose computing device (GPU if available, else CPU)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using training device: {device} " + ("(NVIDIA GPU)" if device == 0 else "(CPU - Training might take some time)"))
    
    # 4. Load base pretrained model
    print("Loading base YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # 5. Start training
    epochs = 5  # Reduced epochs for fast CPU training demonstration
    print(f"Starting training for {epochs} epochs...")
    
    try:
        results = model.train(
    data=yaml_path,

    # Training
    epochs=150,
    patience=30,

    # Images
    imgsz=640,

    # Hardware
    batch=16,
    device=device,
    workers=0,          # Windows

    # Performance
    cache=True,

    # Optimization
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,

    # Augmentation
    augment=True,
    mosaic=0.5,
    mixup=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.4,

    # Saving
    save=True,
    project=os.path.join(workspace_dir, "runs", "train"),
    name="indian_currency_yolo",
    exist_ok=True
)
        print("Training completed successfully!")
        
        # 6. Copy best weights to models/currency_yolo.pt
        best_weights_path = os.path.join(workspace_dir, "runs", "train", "indian_currency_yolo", "weights", "best.pt")
        dest_weights_path = os.path.join(workspace_dir, "models", "currency_yolo.pt")
        
        os.makedirs(os.path.join(workspace_dir, "models"), exist_ok=True)
        
        if os.path.exists(best_weights_path):
            shutil.copy(best_weights_path, dest_weights_path)
            print(f"Copied custom-trained weights directly to app folder: {dest_weights_path}")
            print("Restart/reload your Streamlit app to activate automatic currency detection!")
        else:
            print(f"Warning: Trained weights file not found at {best_weights_path}")
            
    except Exception as e:
        print(f"Training encountered an error: {e}")

if __name__ == "__main__":
    main()
