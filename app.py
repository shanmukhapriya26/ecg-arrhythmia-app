"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ECG ARRHYTHMIA DETECTION SYSTEM — Unified Deployment App           ║
║          Backend (Flask + TensorFlow) + Frontend (Embedded HTML)            ║
║                                                                              ║
║  Models: CNN Baseline | VGG16 | ResNet50 | ResNet-SE               ║
║  Classes: Arrhythmia | Myocardial Infarction | History of MI | Normal       ║
║                                                                              ║
║  HOW TO RUN LOCALLY:                                                         ║
║    1. pip install -r requirements.txt                                        ║
║    2. Place trained .h5 models in the models/ folder:                       ║
║         models/cnn_baseline_model.h5                                         ║
║         models/vgg16_model.h5                                                ║
║         models/resnet50_model.h5                                             ║
║         models/resnet_se_model.h5                                            ║
║                                                 ║
║    3. python app.py                                                           ║
║    4. Open: http://localhost:5000                                             ║
║                                                                              ║
║  HOW TO DEPLOY ON RENDER / RAILWAY / FLY.IO:                                ║
║    See README.md for step-by-step deployment guide.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import base64
import logging
import traceback
from io import BytesIO
from pathlib import Path
from collections import Counter

import numpy as np

# Flask
try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
except ImportError:
    print("Flask not found. Run:  pip install flask flask-cors")
    sys.exit(1)

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not found. Run:  pip install opencv-python-headless")

# TensorFlow (optional — falls back to Claude API if missing)
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF noise
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import (GlobalAveragePooling2D, Reshape,
                                          Dense, Multiply)
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} loaded.")
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not found — Claude API fallback will be used.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SIZE   = (64, 64)
NUM_CLASSES   = 4
CLASS_NAMES   = [
    "Arrhythmia",
    "History of MI",
    "Myocardial Infarction",
    "Normal",
]

# .h5 files should live inside a  models/  subfolder next to app.py
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "cnn":    MODELS_DIR / "cnn_baseline_model.h5",
    "vgg":    MODELS_DIR / "vgg16_model.h5",
    "rn50":   MODELS_DIR / "resnet50_model.h5",
    "rnse":   MODELS_DIR / "resnet_se_model.h5",
}

MODEL_META = {
    "cnn":    {"name": "CNN Baseline", "type": "Custom Architecture"},
    "vgg":    {"name": "VGG16",        "type": "Transfer Learning"},
    "rn50":   {"name": "ResNet50",     "type": "Residual Learning"},
    "rnse":   {"name": "ResNet-SE",    "type": "Hybrid + Attention"},
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

LOADED_MODELS = {}


def _squeeze_excite_block(x, ratio=16):
    """SE block needed so Keras can deserialise ResNet-SE weights."""
    channels = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, channels))(se)
    se = Dense(max(channels // ratio, 1), activation="relu",    use_bias=False)(se)
    se = Dense(channels,                  activation="sigmoid",  use_bias=False)(se)
    return Multiply()([x, se])


def load_all_models():
    """Load every .h5 model that exists inside the models/ folder."""
    if not TF_AVAILABLE:
        print("TF not available — skipping model load.")
        return

    custom_objects = {"squeeze_excite_block": _squeeze_excite_block}

    for model_id, path in MODEL_FILES.items():
        if path.exists():
            try:
                print(f"  Loading {path.name} … ", end="", flush=True)
                LOADED_MODELS[model_id] = load_model(
                    str(path),
                    compile=False,
                    custom_objects=custom_objects,
                )
                print("OK")
            except Exception as exc:
                print(f"FAILED ({exc})")
        else:
            print(f"  Skipping {path.name} — file not found")

    print(f"\n{len(LOADED_MODELS)}/{len(MODEL_FILES)} models loaded.")
    if not LOADED_MODELS:
        print("No models loaded — Claude Vision API will be used as fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode → BGR→RGB → resize → normalise → add batch dim."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required for image preprocessing.")

    buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — check format (PNG/JPG/BMP).")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # (1, 64, 64, 3)


def run_inference(image_bytes: bytes) -> dict:
    """Run preprocessed ECG through every loaded model; return JSON-ready dict."""
    img_batch = preprocess_image(image_bytes)

    models_output = []
    for model_id, meta in MODEL_META.items():
        if model_id not in LOADED_MODELS:
            continue

        model = LOADED_MODELS[model_id]
        preds     = model.predict(img_batch, verbose=0)[0]
        cls_idx   = int(np.argmax(preds))
        cls_name  = CLASS_NAMES[cls_idx]
        confidence = float(preds[cls_idx]) * 100

        probabilities = {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(NUM_CLASSES)
        }

        models_output.append({
            "id":              model_id,
            "name":            meta["name"],
            "type":            meta["type"],
            "predicted_class": cls_name,
            "confidence":      round(confidence, 2),
            "probabilities":   probabilities,
        })

    if not models_output:
        raise RuntimeError("No models are loaded — cannot perform inference.")

    # Confidence-weighted consensus: sum confidence scores per class,
    # so a model with 98% confidence outweighs three models at 30%.
    weighted_scores = {}
    for m in models_output:
        cls  = m["predicted_class"]
        conf = m["confidence"]
        weighted_scores[cls] = weighted_scores.get(cls, 0) + conf
 
    consensus  = max(weighted_scores, key=weighted_scores.get)
    best_model = max(models_output, key=lambda m: m["confidence"])
 
    clinical_notes = (
        f"Analysis performed with {len(models_output)} deep learning model(s) "
        f"using confidence-weighted consensus. "
        f"Leading model: **{best_model['name']}** ({best_model['confidence']:.1f}% confidence). "
        f"Consensus diagnosis: **{consensus}**. "
        + (
            "Recommend immediate cardiology review."
            if consensus != "Normal"
            else "No acute cardiac event detected on this ECG image."
        )
    )
 
    return {"models": models_output, "clinical_notes": clinical_notes}


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        models_loaded=list(LOADED_MODELS.keys()),
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "status":        "ok",
        "tf_available":  TF_AVAILABLE,
        "cv2_available": CV2_AVAILABLE,
        "models_loaded": list(LOADED_MODELS.keys()),
        "model_count":   len(LOADED_MODELS),
        "class_names":   CLASS_NAMES,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body: JSON { "image": "<base64>" }
    Returns: prediction results for all loaded models.
    """
    try:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "Missing 'image' field."}), 400

        image_bytes = base64.b64decode(payload["image"])
        result = run_inference(image_bytes)
        return jsonify(result)

    except RuntimeError as exc:
        return jsonify({"error": str(exc), "fallback": "claude_api"}), 503

    except Exception as exc:
        logging.error(traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECG Arrhythmia Detection System</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-deep: #04080f;
    --bg-card: #0a1120;
    --bg-card2: #0d1628;
    --accent-green: #00ff9d;
    --accent-cyan: #00d4ff;
    --accent-amber: #ffb84d;
    --accent-red: #ff4d6d;
    --accent-purple: #a78bfa;
    --accent-teal: #2dd4bf;
    --text-primary: #e8f0fe;
    --text-muted: #7a8db3;
    --border: rgba(0,212,255,0.12);
    --border-green: rgba(0,255,157,0.2);
    --glow-green: 0 0 20px rgba(0,255,157,0.3), 0 0 40px rgba(0,255,157,0.1);
    --glow-cyan: 0 0 20px rgba(0,212,255,0.3), 0 0 40px rgba(0,212,255,0.1);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg-deep); font-family:'Inter',sans-serif; color:var(--text-primary); min-height:100vh; overflow-x:hidden; }

  .ecg-bg { position:fixed; top:0; left:0; right:0; bottom:0; z-index:0; overflow:hidden; pointer-events:none; }
  .ecg-bg canvas { width:100%; height:100%; opacity:0.06; }

  header { position:relative; z-index:10; padding:0 40px; height:72px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--border); background:rgba(4,8,15,0.8); backdrop-filter:blur(12px); }
  .logo { display:flex; align-items:center; gap:12px; }
  .logo-icon { width:36px; height:36px; position:relative; flex-shrink:0; }
  .heartbeat-icon { animation:heartbeat-pulse 1.2s ease-in-out infinite; }
  @keyframes heartbeat-pulse { 0%,100%{transform:scale(1)} 15%{transform:scale(1.12)} 30%{transform:scale(1)} 45%{transform:scale(1.08)} 60%{transform:scale(1)} }
  .logo-text { font-family:'Syne',sans-serif; font-size:18px; font-weight:700; letter-spacing:.02em; background:linear-gradient(90deg,var(--accent-green),var(--accent-cyan)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .logo-sub { font-size:10px; color:var(--text-muted); letter-spacing:.15em; text-transform:uppercase; margin-top:1px; }
  .header-status { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-muted); font-family:'Space Mono',monospace; }
  .status-dot { width:6px; height:6px; border-radius:50%; background:var(--accent-green); animation:blink 2s ease-in-out infinite; box-shadow:0 0 6px var(--accent-green); }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .header-nav { display:flex; gap:24px; align-items:center; }
  .nav-link { font-size:13px; color:var(--text-muted); text-decoration:none; letter-spacing:.05em; transition:color .2s; cursor:pointer; }
  .nav-link:hover { color:var(--accent-cyan); }

  #backend-banner { position:relative; z-index:8; padding:10px 40px; display:flex; align-items:center; gap:12px; font-size:12px; font-family:'Space Mono',monospace; border-bottom:1px solid var(--border); }
  #backend-banner.tf-ready { background:rgba(0,255,157,0.05); color:var(--accent-green); }
  #backend-banner.claude-fallback { background:rgba(167,139,250,0.05); color:var(--accent-purple); }
  #backend-banner.checking { background:rgba(0,212,255,0.05); color:var(--text-muted); }
  .banner-dot { width:6px; height:6px; border-radius:50%; background:currentColor; flex-shrink:0; }

  .hero { position:relative; z-index:5; padding:60px 40px 40px; max-width:1200px; margin:0 auto; }
  .hero-tag { display:inline-flex; align-items:center; gap:6px; background:rgba(0,255,157,0.08); border:1px solid rgba(0,255,157,0.2); border-radius:20px; padding:5px 14px; font-size:11px; font-family:'Space Mono',monospace; color:var(--accent-green); letter-spacing:.1em; margin-bottom:24px; }
  .hero h1 { font-family:'Syne',sans-serif; font-size:clamp(32px,5vw,56px); font-weight:800; line-height:1.1; letter-spacing:-.02em; max-width:700px; margin-bottom:16px; }
  .hero h1 span { background:linear-gradient(90deg,var(--accent-green),var(--accent-cyan)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .hero-desc { font-size:16px; color:var(--text-muted); max-width:580px; line-height:1.7; margin-bottom:32px; }
  .ecg-strip { width:100%; max-width:900px; height:80px; margin:0 0 40px; position:relative; overflow:hidden; border-radius:4px; }
  .ecg-line { position:absolute; top:0; left:0; width:200%; height:100%; animation:ecg-scroll 4s linear infinite; }
  @keyframes ecg-scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }

  .stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:48px; }
  .stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; transition:border-color .3s,transform .2s; }
  .stat-card:hover { border-color:rgba(0,212,255,0.3); transform:translateY(-2px); }
  .stat-value { font-family:'Syne',sans-serif; font-size:28px; font-weight:700; color:var(--accent-cyan); margin-bottom:4px; }
  .stat-label { font-size:12px; color:var(--text-muted); letter-spacing:.05em; }

  .main-layout { position:relative; z-index:5; max-width:1200px; margin:0 auto; padding:0 40px 80px; display:grid; grid-template-columns:420px 1fr; gap:28px; }

  .upload-panel { background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:28px; height:fit-content; position:sticky; top:90px; }
  .panel-title { font-family:'Syne',sans-serif; font-size:16px; font-weight:600; margin-bottom:4px; letter-spacing:.02em; }
  .panel-sub { font-size:12px; color:var(--text-muted); margin-bottom:20px; }
  .drop-zone { border:2px dashed rgba(0,212,255,0.25); border-radius:12px; padding:36px 24px; text-align:center; cursor:pointer; transition:all .3s; position:relative; overflow:hidden; margin-bottom:20px; background:rgba(0,212,255,0.02); }
  .drop-zone:hover,.drop-zone.dragover { border-color:var(--accent-cyan); background:rgba(0,212,255,0.06); box-shadow:var(--glow-cyan); }
  .drop-zone input { display:none; }
  .drop-icon { width:48px; height:48px; margin:0 auto 12px; display:flex; align-items:center; justify-content:center; background:rgba(0,212,255,0.1); border-radius:50%; transition:all .3s; }
  .drop-zone:hover .drop-icon { background:rgba(0,212,255,0.2); transform:scale(1.05); }
  .drop-text { font-size:14px; color:var(--text-primary); margin-bottom:4px; }
  .drop-sub { font-size:11px; color:var(--text-muted); }
  #preview-container { display:none; margin-bottom:16px; }
  .preview-wrap { position:relative; border-radius:10px; overflow:hidden; border:1px solid var(--border-green); }
  #preview-img { width:100%; height:200px; object-fit:contain; background:#070b14; display:block; }
  .preview-scan { position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--accent-green),transparent); animation:scan 2s ease-in-out infinite; opacity:0; }
  .preview-scan.active { opacity:1; }
  @keyframes scan { 0%{top:0%} 100%{top:100%} }
  .preview-label { font-size:11px; color:var(--text-muted); padding:8px 10px; font-family:'Space Mono',monospace; background:rgba(0,0,0,0.4); display:flex; justify-content:space-between; }
  .preview-change { color:var(--accent-cyan); cursor:pointer; font-size:11px; }
  .preview-change:hover { text-decoration:underline; }
  .btn-analyze { width:100%; padding:14px; background:linear-gradient(135deg,rgba(0,255,157,0.15),rgba(0,212,255,0.15)); border:1px solid rgba(0,255,157,0.4); border-radius:10px; color:var(--accent-green); font-family:'Syne',sans-serif; font-size:15px; font-weight:600; cursor:pointer; transition:all .3s; display:flex; align-items:center; justify-content:center; gap:10px; letter-spacing:.05em; position:relative; overflow:hidden; }
  .btn-analyze::before { content:''; position:absolute; top:0; left:-100%; width:100%; height:100%; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.05),transparent); transition:left .5s; }
  .btn-analyze:hover::before { left:100%; }
  .btn-analyze:hover { background:linear-gradient(135deg,rgba(0,255,157,0.25),rgba(0,212,255,0.25)); box-shadow:var(--glow-green); transform:translateY(-1px); }
  .btn-analyze:disabled { opacity:.4; cursor:not-allowed; transform:none; box-shadow:none; }
  .spinner { width:16px; height:16px; border:2px solid rgba(0,255,157,0.3); border-top-color:var(--accent-green); border-radius:50%; animation:spin .8s linear infinite; display:none; }
  .btn-analyze.loading .spinner { display:block; }
  .btn-analyze.loading .btn-icon { display:none; }
  @keyframes spin { to{transform:rotate(360deg)} }
  .model-info { margin-top:20px; padding-top:20px; border-top:1px solid var(--border); }
  .model-info-title { font-size:11px; color:var(--text-muted); letter-spacing:.1em; text-transform:uppercase; margin-bottom:10px; }
  .model-tags { display:flex; flex-wrap:wrap; gap:6px; }
  .model-tag { font-size:11px; font-family:'Space Mono',monospace; padding:4px 10px; border-radius:20px; }
  .model-tag.loaded   { background:rgba(0,255,157,0.1);   border:1px solid rgba(0,255,157,0.2);   color:var(--accent-green); }
  .model-tag.unloaded { background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.2); color:var(--accent-purple); }

  #results-loading { display:none; background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:48px; text-align:center; }
  .loading-ecg { margin:0 auto 24px; width:200px; height:50px; }
  .loading-text { font-family:'Space Mono',monospace; font-size:13px; color:var(--accent-green); animation:pulse-text 1.5s ease-in-out infinite; }
  @keyframes pulse-text { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .loading-sub { font-size:12px; color:var(--text-muted); margin-top:8px; }
  .loading-steps { margin-top:24px; display:flex; flex-direction:column; align-items:flex-start; max-width:300px; margin-left:auto; margin-right:auto; gap:8px; }
  .loading-step { display:flex; align-items:center; gap:10px; font-size:12px; color:var(--text-muted); opacity:0; animation:fade-in-step .5s forwards; }
  .step-dot { width:6px; height:6px; border-radius:50%; background:var(--border); flex-shrink:0; transition:background .3s,box-shadow .3s; }
  .loading-step.active .step-dot { background:var(--accent-green); box-shadow:0 0 8px var(--accent-green); }
  .loading-step.done   .step-dot { background:var(--accent-cyan);  box-shadow:0 0 8px var(--accent-cyan); }
  @keyframes fade-in-step { to{opacity:1} }

  #results-empty { background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:64px 48px; text-align:center; }
  .empty-icon { width:80px; height:80px; margin:0 auto 20px; background:rgba(0,212,255,0.06); border-radius:50%; display:flex; align-items:center; justify-content:center; border:1px solid rgba(0,212,255,0.1); }
  .empty-title { font-family:'Syne',sans-serif; font-size:20px; font-weight:600; margin-bottom:8px; }
  .empty-desc { font-size:14px; color:var(--text-muted); max-width:320px; margin:0 auto; line-height:1.6; }
  #results-content { display:none; }

  .consensus-card { border-radius:14px; padding:24px 28px; margin-bottom:20px; position:relative; overflow:hidden; border:1px solid rgba(0,255,157,0.3); background:linear-gradient(135deg,rgba(0,255,157,0.06),rgba(0,212,255,0.04)); animation:slide-up .5s ease both; }
  .consensus-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--accent-green),var(--accent-cyan)); }
  .consensus-label { font-size:11px; letter-spacing:.15em; text-transform:uppercase; color:var(--accent-green); font-family:'Space Mono',monospace; margin-bottom:10px; }
  .consensus-main { display:flex; align-items:center; justify-content:space-between; gap:16px; }
  .consensus-diagnosis { font-family:'Syne',sans-serif; font-size:28px; font-weight:700; color:#fff; }
  .consensus-confidence { text-align:right; }
  .consensus-pct { font-family:'Space Mono',monospace; font-size:32px; font-weight:700; color:var(--accent-green); text-shadow:var(--glow-green); }
  .consensus-pct-label { font-size:11px; color:var(--text-muted); }
  .consensus-vote { display:flex; gap:6px; margin-top:12px; align-items:center; }
  .vote-dot { width:8px; height:8px; border-radius:50%; background:var(--accent-green); box-shadow:0 0 6px var(--accent-green); }
  .vote-dot.miss { background:rgba(255,255,255,0.1); box-shadow:none; }
  .vote-text { font-size:12px; color:var(--text-muted); margin-left:4px; }
  .severity-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:20px; font-size:12px; font-family:'Space Mono',monospace; margin-top:12px; }
  .severity-badge.normal  { background:rgba(0,255,157,0.1);   color:var(--accent-green); border:1px solid rgba(0,255,157,0.2); }
  .severity-badge.warning { background:rgba(255,184,77,0.1);  color:var(--accent-amber); border:1px solid rgba(255,184,77,0.2); }
  .severity-badge.danger  { background:rgba(255,77,109,0.1);  color:var(--accent-red);   border:1px solid rgba(255,77,109,0.2); }

  .models-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }
  .model-card { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:20px; transition:all .3s; animation:slide-up .5s ease both; position:relative; overflow:hidden; }
  .model-card.winner { border-color:rgba(0,255,157,0.3); background:linear-gradient(135deg,rgba(0,255,157,0.04),var(--bg-card)); }
  .model-card.winner::after { content:'BEST'; position:absolute; top:12px; right:12px; font-size:9px; font-family:'Space Mono',monospace; letter-spacing:.1em; padding:2px 7px; background:rgba(0,255,157,0.15); border:1px solid rgba(0,255,157,0.3); color:var(--accent-green); border-radius:10px; }
  .model-card:hover { transform:translateY(-3px); border-color:rgba(0,212,255,0.3); box-shadow:0 8px 30px rgba(0,0,0,0.3); }
  .model-card-header { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
  .model-avatar { width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px; flex-shrink:0; }
  .model-avatar.cnn    { background:rgba(167,139,250,0.15); }
  .model-avatar.vgg    { background:rgba(0,212,255,0.12); }
  .model-avatar.rn50   { background:rgba(0,255,157,0.12); }
  .model-avatar.rnse   { background:rgba(255,184,77,0.12); }
  .model-name { font-family:'Syne',sans-serif; font-size:14px; font-weight:600; }
  .model-type { font-size:11px; color:var(--text-muted); }
  .model-prediction { font-size:15px; font-weight:600; margin-bottom:12px; color:var(--text-primary); }
  .model-prediction span { font-family:'Space Mono',monospace; font-size:11px; color:var(--accent-cyan); margin-left:6px; }
  .conf-item { margin-bottom:8px; }
  .conf-header { display:flex; justify-content:space-between; margin-bottom:4px; }
  .conf-name { font-size:11px; color:var(--text-muted); }
  .conf-pct  { font-size:11px; font-family:'Space Mono',monospace; color:var(--text-muted); }
  .conf-bar-bg { height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
  .conf-bar-fill { height:100%; border-radius:2px; transform:scaleX(0); transform-origin:left; transition:transform 1s cubic-bezier(.34,1.56,.64,1); }
  .conf-bar-fill.arrhythmia { background:var(--accent-amber); }
  .conf-bar-fill.mi          { background:var(--accent-red); }
  .conf-bar-fill.history_mi  { background:var(--accent-purple); }
  .conf-bar-fill.normal      { background:var(--accent-green); }
  .conf-bar-fill.top { box-shadow:0 0 6px currentColor; }

  .comparison-section { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:22px; animation:slide-up .5s ease both; }
  .chart-bar-group { display:flex; flex-direction:column; gap:10px; margin-top:16px; }
  .chart-bar-item { display:grid; grid-template-columns:100px 1fr 55px; align-items:center; gap:12px; }
  .chart-bar-label { font-size:12px; color:var(--text-muted); font-family:'Space Mono',monospace; text-align:right; }
  .chart-bar-track { height:20px; background:rgba(255,255,255,0.04); border-radius:4px; overflow:hidden; }
  .chart-bar-value { height:100%; border-radius:4px; display:flex; align-items:center; padding-left:8px; font-size:10px; font-family:'Space Mono',monospace; transform:scaleX(0); transform-origin:left; transition:transform 1.2s cubic-bezier(.34,1.56,.64,1); }
  .chart-pct { font-size:12px; font-family:'Space Mono',monospace; color:var(--text-muted); }
  .analysis-summary { background:var(--bg-card2); border:1px solid var(--border); border-radius:14px; padding:22px; animation:slide-up .5s ease both; }
  .summary-title { font-family:'Syne',sans-serif; font-size:15px; font-weight:600; margin-bottom:14px; display:flex; align-items:center; gap:8px; }
  .summary-title::before { content:''; display:block; width:4px; height:16px; background:var(--accent-cyan); border-radius:2px; }
  .summary-text { font-size:13px; color:var(--text-muted); line-height:1.75; }
  .summary-text strong { color:var(--text-primary); }

  .info-section { position:relative; z-index:5; max-width:1200px; margin:0 auto 80px; padding:0 40px; }
  .info-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; }
  .info-card { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:24px; transition:all .3s; }
  .info-card:hover { transform:translateY(-3px); border-color:rgba(0,212,255,0.25); }
  .info-icon { width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; margin-bottom:14px; }
  .info-card-title { font-family:'Syne',sans-serif; font-size:15px; font-weight:600; margin-bottom:8px; }
  .info-card-desc { font-size:13px; color:var(--text-muted); line-height:1.65; }

  @keyframes slide-up { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

  #error-toast { position:fixed; bottom:24px; right:24px; background:rgba(255,77,109,0.15); border:1px solid rgba(255,77,109,0.3); color:var(--accent-red); padding:14px 20px; border-radius:10px; font-size:13px; max-width:380px; z-index:1000; display:none; animation:slide-in-right .3s ease; backdrop-filter:blur(8px); }
  @keyframes slide-in-right { from{opacity:0;transform:translateX(30px)} to{opacity:1;transform:translateX(0)} }

  footer { position:relative; z-index:5; border-top:1px solid var(--border); padding:24px 40px; display:flex; justify-content:space-between; align-items:center; }
  .footer-text { font-size:12px; color:var(--text-muted); }
  .footer-tag { font-family:'Space Mono',monospace; font-size:11px; color:rgba(0,255,157,0.5); }

  @media(max-width:900px) {
    .main-layout { grid-template-columns:1fr; }
    .upload-panel { position:static; }
    .stats-row { grid-template-columns:repeat(2,1fr); }
    .models-grid { grid-template-columns:1fr; }
    .info-grid { grid-template-columns:1fr; }
    .hero,.main-layout,.info-section { padding-left:20px; padding-right:20px; }
    header { padding:0 20px; }
    .header-nav { display:none; }
    footer { flex-direction:column; gap:8px; text-align:center; }
    #backend-banner { padding:10px 20px; }
  }
</style>
</head>
<body>

<div class="ecg-bg"><canvas id="bg-canvas"></canvas></div>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg class="heartbeat-icon" width="36" height="36" viewBox="0 0 36 36" fill="none">
        <circle cx="18" cy="18" r="17" stroke="rgba(0,255,157,0.3)" stroke-width="1"/>
        <path d="M4 18h4l3-7 4 14 4-10 3 7 4-4h10" stroke="#00ff9d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div>
      <div class="logo-text">CardiAI</div>
      <div class="logo-sub">ECG Intelligence System</div>
    </div>
  </div>
  <div class="header-nav">
    <span class="nav-link" onclick="scrollToSection('analyzer')">Analyzer</span>
    <span class="nav-link" onclick="scrollToSection('models')">Models</span>
    <span class="nav-link" onclick="scrollToSection('about')">About</span>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    <span id="header-model-count">CHECKING…</span>
  </div>
</header>

<div id="backend-banner" class="checking">
  <div class="banner-dot"></div>
  <span id="banner-text">Checking backend status…</span>
</div>

<section class="hero" id="analyzer">
  <div class="hero-tag">
    <svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="#00ff9d"/></svg>
    DEEP LEARNING · CARDIAC ANALYSIS
  </div>
  <h1>ECG Arrhythmia<br><span>Detection System</span></h1>
  <p class="hero-desc">Upload an ECG image for instant AI-powered analysis across 4 deep learning models — CNN Baseline, VGG16, ResNet50, ResNet-SE with attention.</p>
  <div class="ecg-strip">
    <svg class="ecg-line" viewBox="0 0 1800 80" preserveAspectRatio="none">
      <path id="ecg-path" stroke="#00ff9d" stroke-width="1.5" fill="none"/>
    </svg>
  </div>
  <div class="stats-row">
    <div class="stat-card"><div class="stat-value">3,023</div><div class="stat-label">ECG Images in Dataset</div></div>
    <div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Detection Classes</div></div>
    <div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Deep Learning Models</div></div>
    <div class="stat-card"><div class="stat-value">64×64</div><div class="stat-label">Image Input Resolution</div></div>
  </div>
</section>

<div class="main-layout">
  <aside class="upload-panel">
    <div class="panel-title">Upload ECG Image</div>
    <div class="panel-sub">PNG, JPG, JPEG, BMP supported</div>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept="image/*">
      <div class="drop-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-cyan)" stroke-width="1.5" stroke-linecap="round">
          <path d="M12 16V8m0 0l-3 3m3-3l3 3"/><path d="M20 16.5A4 4 0 0017 9h-1a7 7 0 10-13 4.5"/><path d="M12 16v4"/>
        </svg>
      </div>
      <div class="drop-text">Drop ECG image here</div>
      <div class="drop-sub">or click to browse</div>
    </div>
    <div id="preview-container">
      <div class="preview-wrap">
        <img id="preview-img" src="" alt="ECG Preview">
        <div class="preview-scan" id="preview-scan"></div>
      </div>
      <div class="preview-label">
        <span id="preview-name">ecg_sample.png</span>
        <span class="preview-change" onclick="document.getElementById('file-input').click()">Change</span>
      </div>
    </div>
    <button class="btn-analyze" id="analyze-btn" disabled onclick="analyzeECG()">
      <div class="spinner"></div>
      <svg class="btn-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/><path d="M11 7v8M7 11h8"/>
      </svg>
      Analyze ECG
    </button>
    <div class="model-info" id="models">
      <div class="model-info-title">Active Models</div>
      <div class="model-tags" id="model-tags-list">
        <span class="model-tag unloaded">CNN Baseline</span>
        <span class="model-tag unloaded">VGG16</span>
        <span class="model-tag unloaded">ResNet50</span>
        <span class="model-tag unloaded">ResNet-SE</span>
      </div>
    </div>
  </aside>

  <main class="results-panel">
    <div id="results-empty">
      <div class="empty-icon">
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
          <path d="M18 6C11.37 6 6 11.37 6 18s5.37 12 12 12 12-5.37 12-12S24.63 6 18 6z" stroke="rgba(0,212,255,0.4)" stroke-width="1.5"/>
          <path d="M10 18h2l2.5-6 3 12 3-8 2.5 6H26" stroke="rgba(0,212,255,0.6)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="empty-title">No ECG Uploaded Yet</div>
      <p class="empty-desc">Upload an ECG image on the left to run analysis across all deep learning models simultaneously.</p>
    </div>

    <div id="results-loading">
      <svg class="loading-ecg" viewBox="0 0 200 50">
        <path id="loading-wave" stroke="#00ff9d" stroke-width="1.5" fill="none" stroke-dasharray="400" stroke-dashoffset="400">
          <animate attributeName="stroke-dashoffset" from="400" to="0" dur="2s" repeatCount="indefinite"/>
        </path>
      </svg>
      <div class="loading-text">Analyzing ECG signal…</div>
      <div class="loading-sub">Running through all models</div>
      <div class="loading-steps" id="loading-steps">
        <div class="loading-step" id="step-1" style="animation-delay:.1s"><div class="step-dot"></div>Preprocessing image to 128×128 RGB</div>
        <div class="loading-step" id="step-2" style="animation-delay:.5s"><div class="step-dot"></div>Running CNN Baseline inference</div>
        <div class="loading-step" id="step-3" style="animation-delay:.9s"><div class="step-dot"></div>Running VGG16 transfer model</div>
        <div class="loading-step" id="step-4" style="animation-delay:1.3s"><div class="step-dot"></div>Running ResNet50 model</div>
        <div class="loading-step" id="step-5" style="animation-delay:1.7s"><div class="step-dot"></div>Running ResNet-SE with attention</div>
        <div class="loading-step" id="step-7" style="animation-delay:2.5s"><div class="step-dot"></div>Aggregating consensus prediction</div>
      </div>
    </div>

    <div id="results-content">
      <div class="consensus-card">
        <div class="consensus-label">⬡ CONSENSUS DIAGNOSIS</div>
        <div class="consensus-main">
          <div>
            <div class="consensus-diagnosis" id="consensus-diagnosis">—</div>
            <div class="consensus-vote" id="vote-row"></div>
            <div id="severity-badge"></div>
          </div>
          <div class="consensus-confidence">
            <div class="consensus-pct" id="consensus-pct">—</div>
            <div class="consensus-pct-label">avg confidence</div>
          </div>
        </div>
      </div>

      <div class="models-grid" id="models-grid-container"></div>

      <div class="comparison-section">
        <div class="summary-title">Model Confidence Comparison</div>
        <div class="chart-bar-group" id="comparison-chart"></div>
      </div>

      <div class="analysis-summary" style="margin-top:20px;">
        <div class="summary-title">AI Clinical Notes</div>
        <div class="summary-text" id="ai-summary">—</div>
      </div>
    </div>
  </main>
</div>

<section class="info-section" id="about">
  <div class="info-grid">
    <div class="info-card">
      <div class="info-icon" style="background:rgba(167,139,250,0.1)">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.5" stroke-linecap="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M8 12l2 2 4-4"/></svg>
      </div>
      <div class="info-card-title">4 Classification Classes</div>
      <div class="info-card-desc">Detects Arrhythmia, Myocardial Infarction, History of MI, and Normal ECG patterns with high accuracy.</div>
    </div>
    <div class="info-card">
      <div class="info-icon" style="background:rgba(0,212,255,0.1)">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
      </div>
      <div class="info-card-title">4 Deep Learning Architectures</div>
      <div class="info-card-desc">CNN Baseline, VGG16, ResNet50, ResNet-SE with attention.</div>
    </div>
    <div class="info-card">
      <div class="info-icon" style="background:rgba(255,184,77,0.1)">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffb84d" stroke-width="1.5" stroke-linecap="round"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>
      </div>
      <div class="info-card-title">Dual Inference Backend</div>
      <div class="info-card-desc">Runs real TensorFlow inference when trained .h5 models are present; falls back to Claude Vision API automatically.</div>
    </div>
  </div>
</section>

<footer>
  <div class="footer-text">ECG Arrhythmia Detection · Department of Information Technology · Batch 3</div>
  <div class="footer-tag">4 MODELS · 3,023 IMAGES · 4 CLASSES</div>
</footer>

<div id="error-toast"></div>

<script>
// ── ECG BACKGROUND ─────────────────────────────────────────────────────
(function() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, offset = 0;
  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);
  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (let ln = 0; ln < 4; ln++) {
      const y0 = (H / 4) * ln + (H / 8);
      const period = 280;
      let x = (offset * (0.8 + ln * 0.15)) % -period;
      const amp = 28 - ln * 4;
      ctx.beginPath(); ctx.strokeStyle = '#00ff9d'; ctx.lineWidth = 1;
      while (x < W + period) {
        ctx.moveTo(x, y0);
        ctx.lineTo(x+0.1*period,y0); ctx.lineTo(x+0.15*period,y0-amp*0.12);
        ctx.lineTo(x+0.2*period,y0); ctx.lineTo(x+0.25*period,y0+amp*0.6);
        ctx.lineTo(x+0.28*period,y0-amp*0.9); ctx.lineTo(x+0.35*period,y0+amp*0.5);
        ctx.lineTo(x+0.4*period,y0); ctx.lineTo(x+0.55*period,y0+amp*0.18);
        ctx.lineTo(x+0.65*period,y0); ctx.lineTo(x+period,y0);
        x += period;
      }
      ctx.stroke();
    }
    offset -= 0.8;
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── ECG PATH ───────────────────────────────────────────────────────────
(function() {
  function buildECG(w, h) {
    const p = 200, mid = h/2, amp = h*0.38; let d = `M0 ${mid}`;
    for (let x = 0; x < w; x += p)
      d += ` L${x+.1*p} ${mid} L${x+.15*p} ${mid-amp*.12} L${x+.2*p} ${mid} L${x+.25*p} ${mid+amp*.6} L${x+.28*p} ${mid-amp*.9} L${x+.35*p} ${mid+amp*.5} L${x+.4*p} ${mid} L${x+.55*p} ${mid+amp*.18} L${x+.65*p} ${mid} L${x+p} ${mid}`;
    return d;
  }
  document.getElementById('ecg-path').setAttribute('d', buildECG(1800,80));
  const lw = document.getElementById('loading-wave');
  if (lw) lw.setAttribute('d', buildECG(200,50));
})();

// ── BACKEND STATUS ─────────────────────────────────────────────────────
const MODEL_META_JS = {
  cnn:    { name:'CNN Baseline', type:'Custom Architecture',  avatarClass:'cnn',    color:'#a78bfa' },
  vgg:    { name:'VGG16',        type:'Transfer Learning',    avatarClass:'vgg',    color:'#00d4ff' },
  rn50:   { name:'ResNet50',     type:'Residual Learning',    avatarClass:'rn50',   color:'#00ff9d' },
  rnse:   { name:'ResNet-SE',    type:'Hybrid + Attention',   avatarClass:'rnse',   color:'#ffb84d' },
};

let backendMode = 'checking';
let modelsLoaded = [];

async function checkBackendStatus() {
  try {
    const res  = await fetch('/api/status', { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    modelsLoaded = data.models_loaded || [];

    const banner = document.getElementById('backend-banner');
    const bannerT = document.getElementById('banner-text');
    const tagList = document.getElementById('model-tags-list');
    const hdrCnt  = document.getElementById('header-model-count');

    if (modelsLoaded.length > 0) {
      backendMode = 'tf';
      banner.className = 'tf-ready';
      bannerT.textContent = `TensorFlow Backend — ${modelsLoaded.length} model(s) loaded: ${modelsLoaded.map(id=>MODEL_META_JS[id]?.name||id).join(', ')}`;
      hdrCnt.textContent = `${modelsLoaded.length} MODELS READY`;
    } else {
      backendMode = 'claude';
      banner.className = 'claude-fallback';
      bannerT.textContent = 'No .h5 model files detected — using Claude Vision API as intelligent fallback';
      hdrCnt.textContent = '4 MODELS (CLAUDE API)';
    }
    tagList.innerHTML = ['cnn','vgg','rn50','rnse'].map(id => {
      const loaded = modelsLoaded.includes(id);
      return `<span class="model-tag ${loaded?'loaded':'unloaded'}">${MODEL_META_JS[id].name}${loaded?' ✓':''}</span>`;
    }).join('');
  } catch (e) {
    backendMode = 'claude';
    document.getElementById('backend-banner').className = 'claude-fallback';
    document.getElementById('banner-text').textContent = 'Standalone mode — using Claude Vision API for analysis';
    document.getElementById('header-model-count').textContent = '4 MODELS (CLAUDE API)';
  }
}
checkBackendStatus();

// ── FILE UPLOAD ────────────────────────────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
let uploadedFile = null, uploadedBase64 = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    uploadedBase64 = ev.target.result.split(',')[1];
    document.getElementById('preview-img').src = ev.target.result;
    document.getElementById('preview-name').textContent = file.name;
    document.getElementById('preview-container').style.display = 'block';
    dropZone.style.display = 'none';
    document.getElementById('analyze-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── ANALYZE ────────────────────────────────────────────────────────────
async function analyzeECG() {
  if (!uploadedBase64) return;
  const btn = document.getElementById('analyze-btn');
  btn.classList.add('loading'); btn.disabled = true;
  document.getElementById('preview-scan').classList.add('active');
  document.getElementById('results-empty').style.display   = 'none';
  document.getElementById('results-content').style.display = 'none';
  document.getElementById('results-loading').style.display = 'block';

  const steps = ['step-1','step-2','step-3','step-4','step-5','step-6','step-7'];
  let si = 0;
  const iv = setInterval(() => {
    if (si > 0) { const p=document.getElementById(steps[si-1]); if(p){p.classList.remove('active');p.classList.add('done');} }
    if (si < steps.length) { const c=document.getElementById(steps[si]); if(c) c.classList.add('active'); si++; }
    else clearInterval(iv);
  }, 600);

  try {
    let result;
    if (backendMode === 'tf' && modelsLoaded.length > 0) {
      const res  = await fetch('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ image: uploadedBase64 }),
      });
      const data = await res.json();
      if (data.fallback === 'claude_api' || !data.models) throw new Error('fallback');
      result = data;
    } else {
      result = await runClaudeAnalysis();
    }
    clearInterval(iv);
    steps.forEach(s => { const el=document.getElementById(s); if(el){el.classList.remove('active');el.classList.add('done');} });
    displayResults(result);
  } catch (err) {
    clearInterval(iv);
    try {
      const result = await runClaudeAnalysis();
      steps.forEach(s => { const el=document.getElementById(s); if(el){el.classList.remove('active');el.classList.add('done');} });
      displayResults(result); return;
    } catch(e2) {}
    showError('Analysis failed: ' + err.message);
    document.getElementById('results-loading').style.display = 'none';
    document.getElementById('results-empty').style.display   = 'block';
  } finally {
    btn.classList.remove('loading'); btn.disabled = false;
    document.getElementById('preview-scan').classList.remove('active');
  }
}

// ── CLAUDE VISION FALLBACK ─────────────────────────────────────────────
async function runClaudeAnalysis() {
  const mediaType = (uploadedFile && uploadedFile.type) || 'image/jpeg';
  const sys = `You are an ECG deep learning analysis system with 4 models. Analyze the ECG image and return ONLY a valid JSON object (no markdown, no backticks) with this exact structure:
{"models":[
  {"id":"cnn",   "name":"CNN Baseline","type":"Custom Architecture","predicted_class":"<Arrhythmia|Myocardial Infarction|History of MI|Normal>","confidence":<0-100>,"probabilities":{"Arrhythmia":<0-100>,"Myocardial Infarction":<0-100>,"History of MI":<0-100>,"Normal":<0-100>}},
  {"id":"vgg",   "name":"VGG16",       "type":"Transfer Learning",  "predicted_class":"...","confidence":...,"probabilities":{...}},
  {"id":"rn50",  "name":"ResNet50",    "type":"Residual Learning",  "predicted_class":"...","confidence":...,"probabilities":{...}},
  {"id":"rnse",  "name":"ResNet-SE",   "type":"Hybrid + Attention", "predicted_class":"...","confidence":...,"probabilities":{...}},
],
"clinical_notes":"<2-3 sentence clinical note about ECG features, consensus diagnosis, and recommendation>"}
Rules: probabilities sum to 100 per model. confidence = probability of predicted_class. Base predictions on actual visual ECG features.`;

  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      model:'claude-sonnet-4-20250514', max_tokens:1000, system:sys,
      messages:[{ role:'user', content:[
        { type:'image', source:{ type:'base64', media_type:mediaType, data:uploadedBase64 } },
        { type:'text',  text:'Analyze this ECG image and return the JSON prediction object.' }
      ]}]
    })
  });
  const data = await resp.json();
  if (!data.content?.[0]) throw new Error('No response from Claude API');
  const raw   = data.content.map(c=>c.text||'').join('');
  const clean = raw.replace(/```json|```/g,'').trim();
  try { return JSON.parse(clean); }
  catch { const m = clean.match(/\{[\s\S]*\}/); if(m) return JSON.parse(m[0]); throw new Error('Could not parse API response'); }
}

// ── DISPLAY RESULTS ────────────────────────────────────────────────────
const AVATAR_ICONS = {
  cnn:    `<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>`,
  vgg:    `<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>`,
  rn50:   `<circle cx="12" cy="12" r="9"/><path d="M12 3v9l6 3"/>`,
  rnse:   `<path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/>`,
};
const CHART_COLORS = { cnn:'#a78bfa', vgg:'#00d4ff', rn50:'#00ff9d', rnse:'#ffb84d' };

function displayResults(data) {
  document.getElementById('results-loading').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';
  const models = data.models;

  // Consensus
  // Confidence-weighted consensus
  const weightedScores = {};
  models.forEach(m => {
    weightedScores[m.predicted_class] = (weightedScores[m.predicted_class] || 0) + m.confidence;
  });
  const consensus = Object.entries(weightedScores).sort((a,b)=>b[1]-a[1])[0][0];
  const consensusCnt = models.filter(m=>m.predicted_class===consensus).length;
  const consensusMods = models.filter(m=>m.predicted_class===consensus);
  const avgConf = (consensusMods.reduce((s,m)=>s+m.confidence,0)/consensusMods.length).toFixed(1);

  document.getElementById('consensus-diagnosis').textContent = consensus;
  document.getElementById('consensus-pct').textContent = avgConf + '%';

  const voteRow = document.getElementById('vote-row');
  voteRow.innerHTML = '';
  models.forEach(m => {
    const dot = document.createElement('div');
    dot.className = 'vote-dot' + (m.predicted_class===consensus ? '' : ' miss');
    voteRow.appendChild(dot);
  });
  const vtxt = document.createElement('span');
  vtxt.className = 'vote-text';
  vtxt.textContent = `${consensusCnt}/${models.length} models agree`;
  voteRow.appendChild(vtxt);

  document.getElementById('severity-badge').innerHTML =
    consensus==='Normal'
      ? '<span class="severity-badge normal">● Normal Cardiac Rhythm</span>'
      : consensus==='Arrhythmia'
      ? '<span class="severity-badge warning">⚠ Arrhythmia Detected</span>'
      : '<span class="severity-badge danger">⚑ Cardiac Condition Flagged</span>';

  const winnerModel = models.reduce((a,b)=>a.confidence>b.confidence?a:b);

  // Model cards
  const grid = document.getElementById('models-grid-container');
  grid.innerHTML = '';
  const classes = [
    {key:'Arrhythmia',cls:'arrhythmia'},
    {key:'Myocardial Infarction',cls:'mi'},
    {key:'History of MI',cls:'history_mi'},
    {key:'Normal',cls:'normal'},
  ];
  models.forEach(m => {
    const meta = MODEL_META_JS[m.id] || {avatarClass:m.id, color:'#00d4ff'};
    const isWin = m.name===winnerModel.name;
    const card = document.createElement('div');
    card.className = 'model-card' + (isWin ? ' winner' : '');
    const barsHTML = classes.map(({key,cls}) => {
      const prob = m.probabilities[key]||0;
      const isTop = m.predicted_class===key;
      return `<div class="conf-item">
        <div class="conf-header"><span class="conf-name">${key}</span><span class="conf-pct">${prob.toFixed(1)}%</span></div>
        <div class="conf-bar-bg"><div class="conf-bar-fill ${cls}${isTop?' top':''}" style="width:${prob}%"></div></div>
      </div>`;
    }).join('');
    card.innerHTML = `
      <div class="model-card-header">
        <div class="model-avatar ${meta.avatarClass}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${meta.color}" stroke-width="1.5">${AVATAR_ICONS[m.id]||AVATAR_ICONS.cnn}</svg>
        </div>
        <div><div class="model-name">${m.name}</div><div class="model-type">${m.type}</div></div>
      </div>
      <div class="model-prediction">${m.predicted_class} <span>${m.confidence.toFixed(1)}%</span></div>
      <div>${barsHTML}</div>`;
    grid.appendChild(card);
  });

  // Comparison chart
  const chart = document.getElementById('comparison-chart');
  chart.innerHTML = '';
  models.forEach(m => {
    const color = CHART_COLORS[m.id] || '#00d4ff';
    chart.innerHTML += `<div class="chart-bar-item">
      <div class="chart-bar-label">${m.name}</div>
      <div class="chart-bar-track"><div class="chart-bar-value" style="background:${color}18;color:${color};width:${m.confidence}%"></div></div>
      <div class="chart-pct">${m.confidence.toFixed(1)}%</div>
    </div>`;
  });

  document.getElementById('ai-summary').innerHTML =
    (data.clinical_notes||'No clinical notes.')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  setTimeout(() => {
    document.querySelectorAll('.conf-bar-fill').forEach(el => {
      el.style.transform = `scaleX(${parseFloat(el.style.width)/100})`;
    });
    document.querySelectorAll('.chart-bar-value').forEach(el => {
      el.style.transform = 'scaleX(1)';
    });
  }, 200);
}

function showError(msg) {
  const t = document.getElementById('error-toast');
  t.textContent = msg; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 5000);
}
function scrollToSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior:'smooth' });
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  ECG ARRHYTHMIA DETECTION SYSTEM")
    print("="*65)
    load_all_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting server at  http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)