# tflite_worker.py — runs TFLite inference under Python 3.12 and prints JSON result to stdout

import sys
import json
import base64
import io
import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

def run(model_path: str, labels_path: str, image_b64: str) -> dict:
    # Decode image
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Load model and run inference
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"]).squeeze()

    # Load labels
    with open(labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]

    # Top-3
    top_indices = np.argsort(scores)[::-1][:3]
    top_3 = [
        {"label": labels[i], "confidence": round(float(scores[i]) * 100, 1)}
        for i in top_indices
    ]
    return {"top_3": top_3}

if __name__ == "__main__":
    # Args: model_path labels_path
    # image_b64 is read from stdin to avoid Windows command-line length limits
    model_path  = sys.argv[1]
    labels_path = sys.argv[2]
    image_b64   = sys.stdin.read().strip()
    result = run(model_path, labels_path, image_b64)
    # Print JSON to stdout for the caller to read
    print(json.dumps(result))
