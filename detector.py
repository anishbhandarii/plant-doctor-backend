# detector.py — loads TFLite model and runs plant disease inference

import io
import os
import random

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level startup: import TFLite interpreter, locate model and labels
# ---------------------------------------------------------------------------
_MODEL_PATH  = "./model/plant_disease.tflite"
_LABELS_PATH = "./model/labels.txt"

# Try lightweight tflite_runtime first (smaller install), fall back to full TF
try:
    from tflite_runtime.interpreter import Interpreter
    _INTERPRETER_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        _INTERPRETER_AVAILABLE = True
    except ImportError:
        _INTERPRETER_AVAILABLE = False

# Load labels
with open(_LABELS_PATH, "r") as _f:
    LABELS = [line.strip() for line in _f if line.strip()]
print(f"Labels loaded: {len(LABELS)} classes")

# REAL mode requires both the model file and a working Interpreter import
if os.path.exists(_MODEL_PATH) and _INTERPRETER_AVAILABLE:
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
    crop_raw    = parts[0] if len(parts) > 0 else raw_label
    disease_raw = parts[1] if len(parts) > 1 else ""

    # --- Clean crop ---
    crop = crop_raw
    crop = crop.replace("_(maize)", "")          # Corn_(maize) → Corn
    crop = crop.replace("(including_sour)", "")  # Cherry_(including_sour) → Cherry_
    crop = crop.replace(",_bell", "")            # Pepper,_bell → Pepper
    crop = crop.replace("_", " ").strip().title()

    # --- Clean disease ---
    disease = disease_raw
    disease = disease.replace("Two-spotted_spider_mite", "").strip()
    disease = disease.replace("Gray_leaf_spot", "Gray Leaf Spot")
    disease = disease.replace("_", " ").strip()
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
    """Run TFLite inference directly in-process."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter = Interpreter(model_path=_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"]).squeeze()

    top_indices = np.argsort(scores)[::-1][:3]
    top_3 = [
        {"label": LABELS[i], "confidence": round(float(scores[i]) * 100, 1)}
        for i in top_indices
    ]
    return _build_result(top_3, mode="real")


def _mock_result() -> dict:
    """Return a realistic random result using the full label list."""
    idx = random.randrange(len(LABELS))
    confidence = round(random.uniform(70, 98), 1)

    remaining    = round(100 - confidence, 1)
    second_conf  = round(random.uniform(0, remaining), 1)
    third_conf   = round(remaining - second_conf, 1)

    other_indices = random.sample(
        [i for i in range(len(LABELS)) if i != idx], 2
    )
    top_3 = [
        {"label": LABELS[idx],                "confidence": confidence},
        {"label": LABELS[other_indices[0]],   "confidence": second_conf},
        {"label": LABELS[other_indices[1]],   "confidence": third_conf},
    ]
    return _build_result(top_3, mode="mock")


def _build_result(top_3: list[dict], mode: str) -> dict:
    """Assemble the standard result dict from a top_3 list."""
    best       = top_3[0]
    raw_label  = best["label"]
    confidence = best["confidence"]

    crop, disease, is_healthy = _parse_label(raw_label)
    health_score = round(confidence) if is_healthy else max(round(100 - confidence), 5)

    return {
        "raw_label":    raw_label,
        "crop":         crop,
        "disease":      disease,
        "confidence":   confidence,
        "health_score": health_score,
        "is_healthy":   is_healthy,
        "top_3":        top_3,
        "mode":         mode,
    }


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def get_mode() -> str:
    """Return 'real' or 'mock' depending on whether the TFLite model is loaded."""
    return "mock" if MOCK_MODE else "real"
