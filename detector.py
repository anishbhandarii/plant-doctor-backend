# detector.py — loads TFLite model and runs plant disease inference

import base64
import io
import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level startup: locate model, labels, and Python 3.12 worker
# ---------------------------------------------------------------------------
_MODEL_PATH  = "./model/plant_disease.tflite"
_LABELS_PATH = "./model/labels.txt"
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tflite_worker.py")

# Python 3.12 executable — has tensorflow-cpu installed for TFLite inference
_PY312 = r"C:\Users\anisb\AppData\Local\Python\pythoncore-3.12-64\python.exe"

# Load labels
with open(_LABELS_PATH, "r") as _f:
    LABELS = [line.strip() for line in _f if line.strip()]
print(f"Labels loaded: {len(LABELS)} classes")

# Enable REAL mode if both the model file and the Python 3.12 worker are reachable
if os.path.exists(_MODEL_PATH) and os.path.exists(_PY312):
    MOCK_MODE = False
    print("Model loaded: REAL mode")
else:
    MOCK_MODE = True
    print("WARNING: No model file found — running in MOCK mode")


# ---------------------------------------------------------------------------
# Helper: parse a raw label into human-readable crop / disease
# ---------------------------------------------------------------------------
def _parse_label(raw_label: str) -> tuple[str, str, bool]:
    """Parses raw_label into (crop, disease, is_healthy)."""
    parts = raw_label.split("___", 1)
    crop_raw = parts[0] if len(parts) > 0 else raw_label
    disease_raw = parts[1] if len(parts) > 1 else ""

    # --- Clean crop ---
    crop = crop_raw
    crop = crop.replace("_(maize)", "")          # Corn_(maize) → Corn
    crop = crop.replace("(including_sour)", "")  # Cherry_(including_sour) → Cherry_
    crop = crop.replace(",_bell", "")            # Pepper,_bell → Pepper
    crop = crop.replace("_", " ").strip().title()

    # --- Clean disease ---
    disease = disease_raw
    # Handle the two compound labels that have a space in them
    disease = disease.replace("Two-spotted_spider_mite", "").strip()  # Spider_mites Two-spotted_spider_mite → Spider_mites
    disease = disease.replace("Gray_leaf_spot", "Gray Leaf Spot")     # already has a space before it
    disease = disease.replace("_", " ").strip()
    # Collapse any double spaces left after replacements
    while "  " in disease:
        disease = disease.replace("  ", " ")
    disease = disease.title()

    is_healthy = "healthy" in raw_label.lower()
    return (crop, disease, is_healthy)


# ---------------------------------------------------------------------------
# Main function: run inference or return a mock result
# ---------------------------------------------------------------------------
def detect_disease(image_bytes: bytes) -> dict:
    """Run plant disease detection on raw image bytes.

    Returns a dict with crop, disease, confidence, health_score, top_3, and mode.
    Works in REAL mode when the TFLite model is present, MOCK mode otherwise.
    """
    if MOCK_MODE:
        return _mock_result()
    return _real_result(image_bytes)


def _real_result(image_bytes: bytes) -> dict:
    """Run TFLite inference via the Python 3.12 subprocess worker."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    result = subprocess.run(
        [_PY312, _WORKER_PATH, _MODEL_PATH, _LABELS_PATH],
        input=image_b64,          # pass base64 via stdin to avoid Windows arg length limit
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"tflite_worker failed: {result.stderr.strip()}")

    data = json.loads(result.stdout.strip())
    return _build_result(data["top_3"], mode="real")


def _mock_result() -> dict:
    """Return a realistic random result using the full label list."""
    idx = random.randrange(len(LABELS))
    confidence = round(random.uniform(70, 98), 1)

    # Build fake top-3 with descending confidences from other random labels
    remaining = round(100 - confidence, 1)
    second_conf = round(random.uniform(0, remaining), 1)
    third_conf = round(remaining - second_conf, 1)

    other_indices = random.sample(
        [i for i in range(len(LABELS)) if i != idx], 2
    )
    top_3 = [
        {"label": LABELS[idx], "confidence": confidence},
        {"label": LABELS[other_indices[0]], "confidence": second_conf},
        {"label": LABELS[other_indices[1]], "confidence": third_conf},
    ]

    return _build_result(top_3, mode="mock")


def _build_result(top_3: list[dict], mode: str) -> dict:
    """Assemble the standard result dict from a top_3 list."""
    best = top_3[0]
    raw_label = best["label"]
    confidence = best["confidence"]

    crop, disease, is_healthy = _parse_label(raw_label)

    health_score = round(confidence) if is_healthy else round(100 - confidence)

    return {
        "raw_label": raw_label,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "health_score": health_score,
        "is_healthy": is_healthy,
        "top_3": top_3,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def get_mode() -> str:
    """Return 'real' or 'mock' depending on whether the TFLite model is loaded."""
    return "mock" if MOCK_MODE else "real"
