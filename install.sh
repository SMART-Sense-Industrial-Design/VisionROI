#!/bin/bash
set -euo pipefail

# ===== config =====
USER_NAME="$(whoami)"
BASE_DIR="/home/$USER_NAME/SMARTSense/ninenox"
VENV_DIR="$BASE_DIR/venv"
REPO_URL="https://github.com/SMARTSenseIndustrialDesign/VisionROI.git"
REPO_NAME="VisionROI"

# ===== prepare base dir =====
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# ===== (แนะนำ) ติดตั้ง system deps สำหรับ Ubuntu/Debian ถ้ามี apt =====
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y python3-venv python3-dev build-essential ffmpeg tesseract-ocr libtesseract-dev libgl1 libglib2.0-0
fi

# ===== create & activate venv =====
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ===== clone or update repo =====
if [ -d "$REPO_NAME/.git" ]; then
  echo "[info] Repo exists, updating..."
  git -C "$REPO_NAME" pull --ff-only
else
  git clone "$REPO_URL"
fi

cd "$REPO_NAME"

# ===== install python deps =====
pip install --upgrade pip
# แพ็กเกจหลักของโปรเจ็กต์
pip install "."
# ชุด server จะติดตั้ง uvicorn + websockets ให้อัตโนมัติ
pip install ".[server]"

echo
echo "✅ Done."
echo "👉 Run: python app.py --use-uvicorn    # หรือ python app.py --port 12000"
