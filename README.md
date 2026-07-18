# 🪙 Smart Donation Management System

An AI-powered Smart Donation Box that **automatically detects and classifies Indian currency notes** using a two-stage deep-learning pipeline, records donations to MongoDB, and displays a real-time dashboard.

---

## 🧠 AI Pipeline

| Stage | Model | Task |
|-------|-------|------|
| 1 | **YOLOv8-nano** (`yolov8n_currency_best.pt`) | Detect & crop currency notes |
| 2 | **MobileNetV3-Small** (`mobilenetv3_currency.pth`) | Classify denomination + confidence |

**Supported denominations:** ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000

---

## 📁 Project Structure

```
imp_project-Avishkaar/
├── app.py                  # Main Streamlit application
├── detector.py             # Two-stage YOLO + MobileNetV3 pipeline
├── database.py             # MongoDB integration
├── requirements.txt        # Python dependencies
│
├── models/
│   ├── yolov8n_currency_best.pt   # YOLOv8-nano weights
│   ├── mobilenetv3_currency.pth   # MobileNetV3-Small weights
│   └── config.json                # Model metadata
│
└── data/
    └── audit_images/              # Auto-saved receipt audit images
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10 – 3.13**
- **MongoDB Community** running on `localhost:27017`
  - Download: https://www.mongodb.com/try/download/community
  - Start: `mongod` (or via Windows Services)

### 2. Set Up Environment

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell
# OR
.venv\Scripts\activate.bat    # CMD

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📱 App Features

### 🏠 Home Page
- **Upload Image** — Drop a JPG/PNG of a currency note
- **Live Webcam** — Capture directly from your laptop camera
- Runs the two-stage AI pipeline automatically
- Shows YOLO detection box + MobileNetV3 denomination + confidence bar
- **"Confirm & Record Donation"** button → generates a unique Receipt ID and saves to MongoDB

### 📊 Dashboard Page
- **KPI cards** — Total collected, total donations, average donation
- **Donut chart** — Denomination mix
- **Area chart** — Daily donation trend
- **Bar chart** — Note count by denomination
- **Audit Log Table** — Browse all receipts with timestamps
- **Audit Image Viewer** — Click any receipt to see the saved camera frame

---

## ⚙️ Configuration

| Setting | Default | How to Change |
|---------|---------|---------------|
| MongoDB URI | `mongodb://localhost:27017/` | Set `MONGO_URI` env var |
| Database name | `smart_donation_db` | Set `MONGO_DB` env var |
| YOLO confidence | 0.50 | Sidebar slider in app |
| CNN confidence  | 0.50 | Sidebar slider in app |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ModuleNotFoundError: torch` | `pip install torch torchvision` |
| `MongoDB offline` warning in sidebar | Start MongoDB: `mongod` or enable the Windows service |
| Models not loaded | Ensure `models/yolov8n_currency_best.pt` and `models/mobilenetv3_currency.pth` exist |
| No detections | Lower YOLO/CNN confidence sliders; ensure good lighting |

---

## 📊 Model Performance (from training)

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv8-nano | mAP@0.5 | **98.28%** |
| YOLOv8-nano | mAP@0.5:0.95 | **68.95%** |
| MobileNetV3-Small | Test Accuracy | **100.0%** |
| Combined params | — | ~4.1M |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit 1.30+
- **Object Detection:** Ultralytics YOLOv8-nano
- **Classification:** PyTorch MobileNetV3-Small
- **Image Processing:** OpenCV, Pillow
- **Database:** MongoDB (via PyMongo)
- **Visualisation:** Plotly, Pandas
