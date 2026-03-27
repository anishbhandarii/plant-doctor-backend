# storage.py — handles image compression and saving to disk

import io
import os
import uuid
from datetime import datetime, timezone

from PIL import Image

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
IMAGES_DIR = os.getenv("IMAGES_DIR", "./data/images")
os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Image storage ready: {IMAGES_DIR}")


# ---------------------------------------------------------------------------
# Save and compress an uploaded image
# ---------------------------------------------------------------------------
def save_image(image_bytes: bytes, user_id: int, scan_id: int) -> dict:
    """Compress and save an uploaded image to IMAGES_DIR.

    Resizes to max 800px on either side, saves as JPEG at quality=75.
    Returns metadata dict with path, filename, and compression stats.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize if either dimension exceeds 800px, preserving aspect ratio
    img.thumbnail((800, 800), Image.LANCZOS)

    # Compress into an in-memory buffer first to measure size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    compressed_bytes = buf.getvalue()
    compressed_size = len(compressed_bytes)

    # Build a unique filename: date_userid_scanid_randomhex.jpg
    filename = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        f"_{user_id}_{scan_id}_{uuid.uuid4().hex[:8]}.jpg"
    )
    filepath = os.path.join(IMAGES_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(compressed_bytes)

    original_size = len(image_bytes)
    return {
        "image_path":        f"data/images/{filename}",
        "image_filename":    filename,
        "original_size_kb":  round(original_size / 1024, 1),
        "compressed_size_kb": round(compressed_size / 1024, 1),
        "compression_ratio": round(original_size / compressed_size, 1),
    }


# ---------------------------------------------------------------------------
# Retrieve a saved image path
# ---------------------------------------------------------------------------
def get_image_path(filename: str) -> str | None:
    """Return the full filesystem path for filename, or None if it does not exist."""
    full_path = os.path.join(IMAGES_DIR, filename)
    return full_path if os.path.exists(full_path) else None
