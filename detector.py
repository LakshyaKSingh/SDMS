"""
detector.py
-----------
Two-stage currency detection pipeline:
  Stage 1: YOLOv8-nano  → detect & crop the banknote
  Stage 2: MobileNetV3-Small → classify the cropped note denomination
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from ultralytics import YOLO
from PIL import Image

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CLASS_NAMES  = ["10", "100", "20", "200", "2000", "50", "500"]
NUM_CLASSES  = len(CLASS_NAMES)
CNN_IMG_SIZE = 224  # MobileNetV3 expects 224×224

DENOM_COLORS = {          # BGR palette for bounding-box overlays
    10:   (45,  82,  160),
    20:   (50,  205,  50),
    50:   (255, 191,   0),
    100:  (211,  85, 186),
    200:  (0,   140, 255),
    500:  (128, 128, 128),
    2000: (0,    69, 255),
}

# Pre-processing transform for MobileNetV3
_mobilenet_transform = transforms.Compose([
    transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Model Loaders
# ──────────────────────────────────────────────
def _build_mobilenet(num_classes: int) -> nn.Module:
    """Build MobileNetV3-Small with custom classification head."""
    model = mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(256, num_classes),
    )
    return model


def _load_mobilenet(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load saved MobileNetV3 checkpoint (strips 'backbone.' prefix if present)."""
    model = _build_mobilenet(num_classes)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip 'backbone.' prefix added during training
    clean_sd = {
        (k[len("backbone."):] if k.startswith("backbone.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_sd, strict=True)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Main Detector Class
# ──────────────────────────────────────────────
class CurrencyDetector:
    """
    Two-stage pipeline:
      1. YOLOv8 detects & crops currency notes from the frame.
      2. MobileNetV3 classifies each crop into a denomination.
    """

    def __init__(
        self,
        yolo_path: str = "models/yolov8n_currency_best.pt",
        mobilenet_path: str = "models/mobilenetv3_currency.pth",
    ):
        self.yolo_path       = yolo_path
        self.mobilenet_path  = mobilenet_path
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model      = None
        self.cnn_model       = None
        self.is_ready        = False
        self._load_models()

    # ── loaders ──────────────────────────────
    def _load_models(self):
        """Load YOLO + MobileNetV3; set is_ready if both succeed."""
        yolo_ok = self._load_yolo()
        cnn_ok  = self._load_cnn()
        self.is_ready = yolo_ok and cnn_ok

    def _load_yolo(self) -> bool:
        try:
            if not os.path.exists(self.yolo_path):
                print(f"[YOLO] Model not found at {self.yolo_path}")
                return False
            self.yolo_model = YOLO(self.yolo_path)
            print(f"[YOLO] Loaded: {self.yolo_path}")
            return True
        except Exception as e:
            print(f"[YOLO] Load error: {e}")
            return False

    def _load_cnn(self) -> bool:
        try:
            if not os.path.exists(self.mobilenet_path):
                print(f"[CNN] Model not found at {self.mobilenet_path}")
                return False
            self.cnn_model = _load_mobilenet(self.mobilenet_path, NUM_CLASSES, self.device)
            print(f"[CNN] Loaded: {self.mobilenet_path}  (device={self.device})")
            return True
        except Exception as e:
            print(f"[CNN] Load error: {e}")
            return False

    # ── helpers ──────────────────────────────
    def _preprocess_frame(self, rgb: np.ndarray) -> np.ndarray:
        """CLAHE on Value channel to handle folds & lighting variation."""
        try:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
        except Exception:
            return rgb

    def _classify_crop(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        """Run MobileNetV3 on a cropped note, return (denomination_str, confidence)."""
        pil_img = Image.fromarray(crop_rgb)
        tensor  = _mobilenet_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = CLASS_NAMES[idx.item()]
        return label, conf.item()

    def _draw_detection(self, img: np.ndarray, box: list, label: str, amount: int, conf: float) -> None:
        """Draw bounding box + denomination label on img (in-place)."""
        xmin, ymin, xmax, ymax = box
        color = DENOM_COLORS.get(amount, (0, 0, 255))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
        text   = f"\u20b9{label} ({conf:.1%})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (xmin, ymin - th - 12), (xmin + tw + 10, ymin), color, -1)
        cv2.putText(img, text, (xmin + 5, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── public API ───────────────────────────
    def detect(
        self,
        image_rgb: np.ndarray,
        yolo_conf: float = 0.50,
        cnn_conf: float  = 0.50,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detect + classify currency notes in image_rgb.

        Returns
        -------
        annotated_image : np.ndarray (RGB)
        detections      : list of dicts with keys:
            box, label, amount, yolo_conf, cnn_conf, confidence (=cnn_conf)
        """
        if not self.is_ready:
            return image_rgb, []

        # 1. Convert RGB to BGR for YOLOv8 (ultralytics expects BGR for numpy arrays)
        bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 2. Run YOLOv8 inference at correct resolution (416) with class-agnostic NMS to eliminate overlapping boxes
        results = self.yolo_model(bgr_img, imgsz=416, conf=yolo_conf, iou=0.45, agnostic_nms=True, verbose=False)
        result  = results[0]

        annotated  = image_rgb.copy()
        detections = []

        for box in result.boxes:
            y_conf = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            # ── guard against degenerate boxes ──
            h, w = image_rgb.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            # ── Stage 2: MobileNetV3 classification on raw crop ──
            crop  = image_rgb[ymin:ymax, xmin:xmax]
            label_cnn, c_conf = self._classify_crop(crop)

            # YOLOv8's own prediction
            cls_id = int(box.cls[0])
            label_yolo = self.yolo_model.names[cls_id]

            # Ensemble: default to YOLOv8's highly accurate prediction (mAP 98.28%)
            # if MobileNet has low confidence or they disagree.
            if label_cnn == label_yolo:
                label = label_cnn
                confidence = c_conf
                is_validated = True
            else:
                # If they disagree, fallback to YOLOv8's prediction directly
                label = label_yolo
                confidence = y_conf
                is_validated = False

            # Filter based on cnn_conf only if validated, otherwise we trust YOLOv8
            if is_validated and confidence < cnn_conf:
                continue

            try:
                amount = int(label)
            except ValueError:
                amount = 0

            self._draw_detection(annotated, [xmin, ymin, xmax, ymax], label, amount, confidence)
            detections.append({
                "box":       [xmin, ymin, xmax, ymax],
                "label":     f"\u20b9{label}",
                "amount":    amount,
                "yolo_conf": y_conf,
                "cnn_conf":  c_conf,
                "confidence": confidence,
            })

        return annotated, detections

    def get_denomination_color(self, amount: int) -> tuple:
        return DENOM_COLORS.get(amount, (0, 0, 255))
