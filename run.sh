#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import cv2, ultralytics, google.genai, dotenv, elevenlabs, pygame; from ultralytics import YOLOE; assert tuple(map(int, ultralytics.__version__.split(".")[:2])) >= (8, 4)' >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"
fi

exec "$VENV/bin/python" "$HERE/run_camera.py" "$@"
