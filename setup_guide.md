# 🖥️ Setup and Migration Guide for Other PCs/Laptops

This guide outlines the step-by-step instructions, commands, and potential changes required to set up and run the Smart Donation Management System on a new PC or laptop.

---

## 📋 Prerequisites

Before setting up the project, ensure the target PC has the following:
1. **Python**: Installed version **3.8 to 3.11** (recommended). Make sure to check the option **"Add Python to PATH"** during installation on Windows.
2. **Webcam / Camera**: A connected webcam is required for the Live Webcam Capture feature (optional; you can use Image File Upload or the simulator if a camera is not available).
3. **Internet Connection**: Required for the first run to download the base YOLOv8 model weights (`yolov8n.pt`).

---

## 🚀 Step-by-Step Setup

### Step 1: Copy/Clone Project Folder
Copy the entire `imp_project-Avishkaar` folder to the target machine (e.g., to the Documents directory).

### Step 2: Open Terminal / Command Prompt
Open PowerShell, Command Prompt, or terminal and navigate (`cd`) to the project directory:
```bash
cd "path/to/imp_project-Avishkaar"
```

### Step 3: Set Up a Python Virtual Environment (Recommended)
Creating a virtual environment ensures that the project dependencies do not conflict with other Python projects on the system.

**On Windows:**
```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1   # If using PowerShell
# OR
.venv\Scripts\activate.bat   # If using Command Prompt
```

**On macOS/Linux:**
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

*(You will know activation was successful when you see `(.venv)` prepend your terminal prompt).*

### Step 4: Install Dependencies
Install all the required Python packages specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration Changes

### 1. Training Path Configuration (Automated)
If you move the project to a new location or a new PC, the absolute dataset path inside `data/Indian currency/data.yaml` will become invalid. 

**Solution**: You do not need to edit it manually. The training script `train_yolo.py` automatically detects its current directory and re-generates `data.yaml` with the correct absolute path before starting training. Simply run:
```bash
python train_yolo.py
```

### 2. Switching Database (Optional)
By default, the application runs on a local SQLite database located at `data/donation_system.db`. If you need to scale to a production PostgreSQL database on the new machine:
- Export the `DATABASE_URL` environment variable before running the application.

**On Windows (PowerShell):**
```powershell
$env:DATABASE_URL="postgresql://username:password@localhost:5432/dbname"
```
**On macOS/Linux (Terminal):**
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/dbname"
```

---

## 🖥️ Running the Application

To launch the Streamlit web app:
```bash
python -m streamlit run app.py
```
Or simply:
```bash
streamlit run app.py
```

Streamlit will start a local web server and print the URLs:
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://<your-network-ip>:8501`

Open `http://localhost:8501` in your web browser.

---

## 🧠 Model Modes

1. **Custom AI Inference Mode**:
   - If the trained model file `models/currency_yolo.pt` is present in the `models/` directory, the system automatically uses your custom model for banknote detection.
2. **COCO Fallback Simulator Mode**:
   - If the `models/currency_yolo.pt` file is missing, the application downloads the general-purpose `yolov8n.pt` model and maps common objects to currencies for demo purposes (e.g., Cell Phone = ₹500, Book = ₹100, Cup = ₹50).
