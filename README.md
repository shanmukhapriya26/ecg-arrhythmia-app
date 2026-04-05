# ECG Arrhythmia Detection System — Deployment Guide

**5 Deep Learning Models · 4 Classes · Flask + React-style Frontend**

---

## Project Structure

```
ecg_app/
├── app.py                    ← Main Flask application (backend + frontend)
├── requirements.txt          ← Python dependencies
├── Procfile                  ← For Render / Railway deployment
├── colab_export_models.py    ← Paste this into Colab to download your models
├── README.md                 ← This file
└── models/                   ← Place your trained .h5 files here
    ├── cnn_baseline_model.h5
    ├── vgg16_model.h5
    ├── resnet50_model.h5
    ├── resnet_se_model.h5
    └── lenet5_model.h5
```

---

## Step 1 — Export Models from Colab

After training your models in the Colab notebook:

1. Open `colab_export_models.py`
2. Copy the entire content
3. Paste it into a **new cell at the bottom** of your Colab notebook
4. Run it — it will save all 5 `.h5` files and trigger browser downloads
5. Check your **Downloads** folder for all 5 files

**Important:** Make sure the `CLASS_NAMES` list in `app.py` matches the output printed by the export script. The default order is:
```
Index 0 → Arrhythmia
Index 1 → History of MI
Index 2 → Myocardial Infarction
Index 3 → Normal
```
If your Colab output shows a different order, update `CLASS_NAMES` in `app.py` line ~40.

---

## Step 2 — Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create models folder and put your .h5 files there
mkdir models
# Move your downloaded .h5 files into models/

# 3. Run the app
python app.py

# 4. Open in browser
# http://localhost:5000
```

The app automatically detects which models are loaded and shows a status banner.
If no `.h5` files are found, it uses the **Claude Vision API** as a smart fallback.

---

## Step 3 — Deploy Online (Choose One)

### Option A: Render.com (Free tier available)

1. Push this folder to a **GitHub repository**
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
5. Click **Deploy**

> **Note:** Free Render instances have ~512MB RAM. TensorFlow + 5 models may exceed this.
> If you hit memory limits, either use a paid plan or rely on the Claude API fallback.

### Option B: Railway.app (Simple, generous free tier)

1. Push folder to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the `Procfile`
4. Done — Railway gives you a public URL

### Option C: Fly.io (Best performance on free tier)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Inside your project folder
fly launch    # follow prompts
fly deploy
```

### Option D: Local + ngrok (Instant public URL, no deployment needed)

```bash
# 1. Run the app locally
python app.py

# 2. In a second terminal, create a public tunnel
pip install pyngrok
python -c "from pyngrok import ngrok; t=ngrok.connect(5000); print(t.public_url)"

# Share the printed URL — it works immediately!
```

---

## Uploading Large Model Files

`.h5` model files (especially VGG16/ResNet50) can be **100–500MB each**. GitHub has a 100MB file limit.

### Option 1 — Git LFS (GitHub Large File Storage)
```bash
git lfs install
git lfs track "models/*.h5"
git add .gitattributes models/
git commit -m "add models via LFS"
git push
```

### Option 2 — Download at startup from Google Drive
Add to `app.py` before `load_all_models()`:
```python
import gdown
MODEL_GDRIVE_IDS = {
    "models/cnn_baseline_model.h5": "YOUR_GDRIVE_FILE_ID",
    # ... etc
}
for path, gid in MODEL_GDRIVE_IDS.items():
    if not Path(path).exists():
        gdown.download(id=gid, output=path, quiet=False)
```

### Option 3 — Use Claude API Fallback (No model files needed)
Deploy without any `.h5` files. The system automatically uses the Claude Vision API
for ECG analysis — fully functional, intelligent, and zero extra setup.

---

## How the Dual-Backend Works

```
User uploads ECG image
        │
        ▼
  Flask /api/predict
        │
  ┌─────┴──────┐
  │            │
  ▼            ▼
.h5 models   No models
present?     loaded?
  │            │
  ▼            ▼
TensorFlow   Claude Vision API
inference    (fallback)
  │            │
  └─────┬──────┘
        ▼
  Results shown in frontend
  (Consensus + 5 model cards + clinical notes)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow==2.16.2` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| ResNet-SE model fails to load | The custom SE block is registered in `app.py` automatically |
| Wrong class predictions | Check that `CLASS_NAMES` in `app.py` matches your Colab `label_encoder.classes_` output |
| Out of memory on cloud | Use 1 worker (`--workers 1`), or use Claude API fallback mode |
| Port already in use | `PORT=8080 python app.py` |

---

## Class Names Verification

After training, your Colab will print something like:
```
Class order:
  Index 0 → Arrhythmia
  Index 1 → History of MI
  Index 2 → Myocardial Infarction
  Index 3 → Normal
```

Make sure `app.py` line ~40 matches exactly:
```python
CLASS_NAMES = [
    "Arrhythmia",
    "History of MI",
    "Myocardial Infarction",
    "Normal",
]
```

---

*ECG Arrhythmia Detection · Department of Information Technology · Batch 3*
