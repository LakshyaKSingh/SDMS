"""
app.py  –  Smart Donation Management System
==========================================
Two-stage AI pipeline:
  • YOLOv8-nano   → detect & crop banknotes
  • MobileNetV3-Small → classify denomination

Pages
-----
  Home       – Upload Image  OR  Live Webcam capture
  Dashboard  – All donations, charts, audit images
"""

import os
import datetime
import uuid
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from database import (
    is_connected, save_donation, get_all_donations,
    get_donation_stats, generate_receipt_id, clear_all_donations,
)
from detector import CurrencyDetector

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Donation Box  |  AI Currency Detector",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS – Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #060b14 0%, #0d1525 50%, #060b14 100%);
    color: #e2e8f0;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #0a1220 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(14, 165, 233, 0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin: 0;
    font-weight: 400;
}

/* ── Nav Tabs ── */
.nav-container {
    display: flex;
    gap: 12px;
    margin-bottom: 28px;
}
.nav-btn {
    flex: 1;
    padding: 16px 24px;
    border-radius: 14px;
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    border: 2px solid transparent;
    text-decoration: none;
}
.nav-btn.active {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    box-shadow: 0 8px 25px rgba(14,165,233,0.4);
}
.nav-btn.inactive {
    background: rgba(255,255,255,0.04);
    border-color: #1e3a5f;
    color: #94a3b8;
}
.nav-btn.inactive:hover {
    border-color: #0ea5e9;
    color: #e2e8f0;
}

/* ── Card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 28px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: border-color 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(14,165,233,0.35);
}

/* ── Metric Cards ── */
.metrics-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 160px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: #0ea5e9;
    box-shadow: 0 12px 35px rgba(14,165,233,0.2);
}
.metric-icon { font-size: 1.6rem; margin-bottom: 10px; }
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Input Mode Selector ── */
.mode-selector {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}
.mode-card {
    flex: 1;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 2px solid #1e3a5f;
    background: rgba(255,255,255,0.03);
}
.mode-card.selected {
    border-color: #0ea5e9;
    background: rgba(14,165,233,0.08);
    box-shadow: 0 0 0 4px rgba(14,165,233,0.12);
}
.mode-icon { font-size: 2rem; margin-bottom: 8px; }
.mode-label { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }

/* ── Detection Result ── */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.15));
    border: 1px solid rgba(14,165,233,0.4);
    border-radius: 100px;
    padding: 10px 20px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 8px 4px;
}
.detection-amount {
    font-size: 1.4rem;
    font-weight: 800;
    color: #a78bfa;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar {
    height: 8px;
    border-radius: 100px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}

/* ── Receipt ── */
.receipt-box {
    background: #040810;
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
    padding: 28px 24px;
    font-family: 'Courier New', monospace;
    color: #e2e8f0;
    position: relative;
    overflow: hidden;
}
.receipt-box::before {
    content: 'OFFICIAL RECEIPT';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-35deg);
    font-size: 3rem;
    font-weight: 900;
    color: rgba(14,165,233,0.04);
    white-space: nowrap;
    pointer-events: none;
}
.receipt-header {
    text-align: center;
    font-weight: 900;
    font-size: 1.1rem;
    color: #38bdf8;
    letter-spacing: 4px;
    margin-bottom: 20px;
    text-transform: uppercase;
}
.receipt-row {
    display: flex;
    justify-content: space-between;
    margin: 8px 0;
    font-size: 0.9rem;
}
.receipt-divider {
    border-top: 1px dashed #1e3a5f;
    margin: 12px 0;
}
.receipt-total {
    font-size: 1.2rem;
    font-weight: 900;
    color: #38bdf8;
}
.verified-stamp {
    text-align: center;
    color: #10b981;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 2px;
    margin-top: 12px;
}

/* ── Status badges ── */
.status-ok  { color: #10b981; font-weight: 700; }
.status-err { color: #f87171; font-weight: 700; }
.status-warn { color: #fbbf24; font-weight: 700; }

/* ── Section Header ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Streamlit overrides ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(14,165,233,0.5) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
}
.stSlider > div { color: #94a3b8; }
div[data-testid="stRadio"] label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Init – Models & Session State
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖  Loading AI models…")
def load_detector():
    return CurrencyDetector(
        yolo_path="models/yolov8n_currency_best.pt",
        mobilenet_path="models/mobilenetv3_currency.pth",
    )

detector = load_detector()

AUDIT_DIR = "data/audit_images"
os.makedirs(AUDIT_DIR, exist_ok=True)

for key, val in [
    ("page", "home"),
    ("input_mode", "Upload Image"),
    ("last_detection", None),
    ("last_receipt", None),
    ("uploader_key", 0),
    ("camera_key", 100),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.8rem;'>🪙</div>
        <div style='font-size:1.1rem; font-weight:800; color:#38bdf8; margin-top:4px;'>Smart Donation Box</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>AI Currency Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model status ──
    st.markdown("**🤖 Model Status**")
    if detector.is_ready:
        st.markdown('<span class="status-ok">✅ Both models loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">⚠️ Model not ready</span>', unsafe_allow_html=True)
        if not detector.yolo_model:
            st.caption("❌ YOLOv8 model not found")
        if not detector.cnn_model:
            st.caption("❌ MobileNetV3 model not found")

    # ── DB status ──
    st.markdown("**🗄️ Database Status**")
    if is_connected():
        st.markdown('<span class="status-ok">✅ MongoDB connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">⚠️ MongoDB offline — donations won\'t save</span>',
                    unsafe_allow_html=True)
        st.caption("Start MongoDB and restart the app.")

    st.divider()

    # ── Confidence sliders ──
    st.markdown("**⚙️ Detection Settings**")
    yolo_conf = st.slider("YOLO confidence", 0.10, 0.95, 0.25, 0.05,
                          help="Minimum YOLO detection confidence (0.25 is recommended for cropping)")
    cnn_conf  = st.slider("CNN confidence",  0.20, 0.95, 0.50, 0.05,
                          help="Minimum MobileNetV3 classification confidence")

    st.divider()

    # ── Quick stats ──
    st.markdown("**📊 Quick Stats**")
    stats = get_donation_stats()
    st.metric("Total Collected",  f"₹{stats['total_amount']:,}")
    st.metric("Donations Logged", f"{stats['total_count']:,}")

    st.divider()

    # ── Maintenance ──
    st.markdown("**🧹 Maintenance**")
    if st.button("Clear All Records", type="secondary"):
        clear_all_donations()
        st.success("All records cleared!")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🪙 Smart Donation Management System</div>
    <div class="hero-subtitle">
        YOLOv8-nano + MobileNetV3-Small · Real-time Currency Verification · MongoDB Receipt Logging
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Page navigation
# ─────────────────────────────────────────────────────────────────────
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🏠  Home — Detect Donation", type="primary" if st.session_state.page == "home" else "secondary",
                 use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
with col_nav2:
    if st.button("📊  Dashboard — All Donations", type="primary" if st.session_state.page == "dashboard" else "secondary",
                 use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()


# ═════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ═════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: Input panel ─────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📥 Select Input Method</div>', unsafe_allow_html=True)

        # Mode selector (radio-based)
        input_mode = st.radio(
            "Input method",
            ["📁  Upload Image", "📷  Live Webcam"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.input_mode = input_mode

        raw_image = None

        # ── Upload Mode ───────────────────────────────────────────────
        if "Upload" in input_mode:
            st.info("📌 Upload a clear photo of an Indian currency note (JPG / PNG).")
            uploaded = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.uploader_key}",
            )
            if uploaded is not None:
                pil_img   = Image.open(uploaded).convert("RGB")
                raw_image = np.array(pil_img)
                st.image(raw_image, caption="📷 Uploaded Image", use_column_width=True)

        # ── Webcam Mode ───────────────────────────────────────────────
        else:
            st.info("📸 Use your webcam to capture a note. Ensure good lighting!")
            cam_snap = st.camera_input("Take a photo", label_visibility="collapsed", key=f"camera_{st.session_state.camera_key}")
            if cam_snap is not None:
                pil_img   = Image.open(cam_snap).convert("RGB")
                raw_image = np.array(pil_img)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Run Detection ─────────────────────────────────────────────
        if raw_image is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if not detector.is_ready:
                st.error("❌ Models not loaded. Check sidebar for details.")
            else:
                with st.spinner("🔍 Running YOLO + MobileNetV3 pipeline…"):
                    annotated, detections = detector.detect(
                        raw_image,
                        yolo_conf=yolo_conf,
                        cnn_conf=cnn_conf,
                    )
                st.session_state.last_detection = {
                    "raw":         raw_image,
                    "annotated":   annotated,
                    "detections":  detections,
                }
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: Result + Receipt panel ─────────────────────────────────
    with col_right:
        detection_data = st.session_state.last_detection

        if detection_data is None:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:60px 28px;">
                <div style="font-size:4rem; margin-bottom:16px;">🔍</div>
                <div style="font-size:1.1rem; font-weight:600; color:#64748b;">
                    Upload or capture an image on the left to detect currency notes.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            annotated  = detection_data["annotated"]
            detections = detection_data["detections"]

            # ── Annotated image ───────────────────────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Detection Result</div>', unsafe_allow_html=True)
            st.image(annotated, caption="YOLOv8 + MobileNetV3 Output", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if len(detections) == 0:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:30px;">
                    <div style="font-size:2rem; margin-bottom:8px;">🤔</div>
                    <div style="color:#64748b; font-weight:600;">
                        No currency notes detected. Try a clearer image or lower the confidence threshold.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                total_detected = sum(d["amount"] for d in detections)

                # ── Detection badges ──────────────────────────────────
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">✅ {len(detections)} Note(s) Detected</div>',
                            unsafe_allow_html=True)

                for d in detections:
                    cnn_pct  = d["cnn_conf"]
                    yolo_pct = d["yolo_conf"]
                    st.markdown(f"""
                    <div style="
                        background: rgba(14,165,233,0.07);
                        border: 1px solid rgba(14,165,233,0.2);
                        border-radius: 12px;
                        padding: 14px 18px;
                        margin-bottom: 10px;
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <div style="font-size:1.3rem; font-weight:800; color:#38bdf8;">{d['label']}</div>
                            <div style="font-size:0.85rem; color:#64748b;">
                                YOLO: {yolo_pct:.0%} &nbsp;|&nbsp; CNN: {cnn_pct:.0%}
                            </div>
                        </div>
                        <div style="font-size:0.72rem; color:#64748b; margin-bottom:4px;">
                            CLASSIFICATION CONFIDENCE
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar" style="width:{cnn_pct*100:.1f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(99,102,241,0.12));
                    border: 1px solid rgba(14,165,233,0.3);
                    border-radius: 14px;
                    padding: 18px;
                    text-align: center;
                    margin-top: 12px;
                ">
                    <div style="font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:2px;">Total Value</div>
                    <div style="font-size:2.5rem; font-weight:900; color:#38bdf8;">₹{total_detected:,}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── Record Donation Button ────────────────────────────
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧾 Record This Donation</div>',
                            unsafe_allow_html=True)

                if not is_connected():
                    st.warning("⚠️ MongoDB is offline. Receipts won't be saved.")

                record_clicked = st.button(
                    "✅  Confirm & Record Donation",
                    type="primary",
                    use_container_width=True,
                    key="record_btn",
                )

                if record_clicked:
                    receipt_id = generate_receipt_id()
                    denom_str  = ", ".join(d["label"] for d in detections)
                    # receipt uses ensemble/validated confidence consistently
                    avg_conf   = sum(float(d.get("confidence", 0.0)) for d in detections) / len(detections)


                    # Save audit image
                    img_path = None
                    try:
                        img_filename = f"{receipt_id}.jpg"
                        img_path     = os.path.join(AUDIT_DIR, img_filename)
                        Image.fromarray(annotated).save(img_path)
                    except Exception as e:
                        st.warning(f"Could not save audit image: {e}")

                    # Save to MongoDB
                    result = save_donation(
                        receipt_id=receipt_id,
                        denomination=denom_str,
                        amount=total_detected,
                        confidence=avg_conf,
                        image_path=img_path,
                    )

                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.last_receipt = {
                        "receipt_id": receipt_id,
                        "timestamp":  ts,
                        "detections": detections,
                        "total":      total_detected,
                        "avg_conf":   avg_conf,
                        "saved":      result is not None,
                    }
                    st.session_state.last_detection = None
                    st.session_state.uploader_key += 1
                    st.session_state.camera_key += 1
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        # ── Receipt Display ───────────────────────────────────────────
        receipt = st.session_state.last_receipt
        if receipt:
            saved_icon = "✔ SAVED TO DATABASE" if receipt.get("saved") else "⚠ DATABASE OFFLINE"
            saved_color = "#10b981" if receipt.get("saved") else "#fbbf24"

            rows_html = ""
            for det in receipt["detections"]:
                try:
                    conf_val = float(det.get("confidence", 0.0))
                except Exception:
                    conf_val = 0.0

                label = det.get("label", "Unknown")
                rows_html += f'<div class="receipt-row"><span>{label}</span><span>{conf_val:.1%} confidence</span></div>'

            receipt_html = (
                f'<div class="receipt-box">'
                f'  <div class="receipt-header">🪙 SMART DONATION RECEIPT 🪙</div>'
                f'  <div class="receipt-row"><span>RECEIPT ID</span><span style="color:#38bdf8; font-weight:700;">{receipt["receipt_id"]}</span></div>'
                f'  <div class="receipt-row"><span>DATE & TIME</span><span>{receipt["timestamp"]}</span></div>'
                f'  <div class="receipt-divider"></div>'
                f'  <div class="receipt-row" style="font-weight:700; color:#94a3b8;"><span>DENOMINATION</span><span>CONFIDENCE</span></div>'
                f'  {rows_html}'
                f'  <div class="receipt-divider"></div>'
                f'  <div class="receipt-row receipt-total"><span>TOTAL DEPOSITED</span><span>₹{receipt["total"]:,}</span></div>'
                f'  <div class="receipt-row" style="font-size:0.8rem; color:#64748b;"><span>AVG CNN CONFIDENCE</span><span>{receipt["avg_conf"]:.2%}</span></div>'
                f'  <div class="receipt-divider"></div>'
                f'  <div class="verified-stamp" style="color:{saved_color};">{saved_icon}</div>'
                f'</div>'
            )
            st.markdown(receipt_html, unsafe_allow_html=True)

            if st.button("🗑  Dismiss Receipt", type="secondary"):
                st.session_state.last_receipt = None
                st.rerun()


# ═════════════════════════════════════════════════════════════════════
#  PAGE 2 – DASHBOARD
# ═════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    stats = get_donation_stats()

    # ── KPI Metrics ───────────────────────────────────────────────────
    avg_donation = (
        int(stats["total_amount"] / stats["total_count"])
        if stats["total_count"] > 0 else 0
    )

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Total Collected</div>
            <div class="metric-value">₹{stats['total_amount']:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🧾</div>
            <div class="metric-label">Total Donations</div>
            <div class="metric-value">{stats['total_count']:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-label">Avg. Donation</div>
            <div class="metric-value">₹{avg_donation:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_connected():
        st.warning("⚠️ MongoDB is offline. No donation data to display.")
    else:
        all_donations = get_all_donations()

        if len(all_donations) == 0:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:60px 28px;">
                <div style="font-size:4rem; margin-bottom:16px;">📭</div>
                <div style="font-size:1.1rem; font-weight:600; color:#64748b;">
                    No donations recorded yet. Head to Home to record your first donation!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = pd.DataFrame(all_donations)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # ── Charts ───────────────────────────────────────────────
            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🍩 Denomination Mix</div>', unsafe_allow_html=True)
                denom_data = df.groupby("denomination")["amount"].sum().reset_index()
                fig_pie = px.pie(
                    denom_data,
                    values="amount",
                    names="denomination",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                fig_pie.update_traces(textfont_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_ch2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📅 Daily Donation Trend</div>', unsafe_allow_html=True)
                df["date"] = df["timestamp"].dt.date
                trend = df.groupby("date")["amount"].sum().reset_index()
                fig_line = px.area(
                    trend, x="date", y="amount",
                    color_discrete_sequence=["#0ea5e9"],
                    labels={"amount": "Total ₹", "date": "Date"},
                )
                fig_line.update_traces(
                    line_color="#0ea5e9",
                    fillcolor="rgba(14,165,233,0.12)",
                )
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_line, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Bar chart ─────────────────────────────────────────────
            if len(stats.get("denominations", [])) > 0:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 Notes Count by Denomination</div>',
                            unsafe_allow_html=True)
                denom_df = pd.DataFrame(stats["denominations"])
                denom_df = denom_df.sort_values("total_value", ascending=False)
                fig_bar = px.bar(
                    denom_df, x="denomination", y="count",
                    color="total_value",
                    color_continuous_scale="Blues",
                    labels={"denomination": "Denomination", "count": "# Notes", "total_value": "Total ₹"},
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#1e3a5f"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Audit Log Table + Image Viewer ────────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🕵️ Audit Log</div>', unsafe_allow_html=True)

            col_tbl, col_img = st.columns([3, 2])

            with col_tbl:
                df_display = df.copy()
                df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x:.1%}")

                selected_receipt = st.selectbox(
                    "Select receipt to view audit image",
                    df_display["receipt_id"].tolist(),
                )
                st.dataframe(
                    df_display[["receipt_id", "timestamp", "denomination", "amount", "confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )

            with col_img:
                if selected_receipt:
                    row = df[df["receipt_id"] == selected_receipt]
                    if not row.empty:
                        img_path = row.iloc[0].get("image_path", None)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"Audit: {selected_receipt}",
                                     use_column_width=True)
                        else:
                            st.info("No audit image found for this receipt.")

            st.markdown('</div>', unsafe_allow_html=True)
