"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 1 — Run this cell in your Colab notebook AFTER training   ║
║  It saves all 5 models to your Google Drive + downloads them.   ║
╚══════════════════════════════════════════════════════════════════╝

Paste this into a new Colab cell and run it.
"""

import os
from google.colab import files

# ── Save all 5 trained models ──────────────────────────────────────────
models_to_save = {
    'cnn_baseline_model.h5' : cnn_baseline,
    'vgg16_model.h5'        : vgg16_model,
    'resnet50_model.h5'     : resnet50_model,
    'resnet_se_model.h5'    : resnet_se_model,
    'lenet5_model.h5'       : lenet5_model,
}

# Also save to Google Drive for backup
drive_save_dir = '/content/drive/MyDrive/ecg_project/saved_models'
os.makedirs(drive_save_dir, exist_ok=True)

for fname, model in models_to_save.items():
    # Save locally in Colab
    model.save(fname)
    print(f'Saved locally: {fname}')

    # Backup to Drive
    drive_path = f'{drive_save_dir}/{fname}'
    model.save(drive_path)
    print(f'Saved to Drive: {drive_path}')

    # Download to your computer
    files.download(fname)
    print(f'Downloaded: {fname}\n')

# ── Print the exact class order (IMPORTANT!) ───────────────────────────
print('='*60)
print('CLASS ORDER (must match app.py CLASS_NAMES list):')
print('='*60)
for i, cls in enumerate(label_encoder.classes_):
    print(f'  Index {i} → {cls}')

print("""
╔═══════════════════════════════════════════════════════════════╗
║  NEXT STEPS:                                                  ║
║  1. Download all 5 .h5 files (check your browser downloads)  ║
║  2. Place them inside the  models/  folder of the project    ║
║  3. Run:  python app.py                                       ║
║  4. Open: http://localhost:5000                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
