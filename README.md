# PlantDoctor Backend

## What is this?

AI-powered plant disease detection API for rural farmers in South Asia. A farmer photographs a sick leaf, the system identifies the disease using a MobileNetV2 CNN model, and returns organic treatment advice in the farmer's local language — all within seconds.

---

## Features

- 🌿 **41 disease classes** — PlantVillage dataset + 3 South Asia rice diseases
- 🤖 **TFLite on-device CNN** — MobileNetV2, 97.43% validation accuracy
- 💬 **Multilingual advice** — organic treatments via Claude AI in any language
- 🔒 **JWT authentication** — secure user accounts with bcrypt password hashing
- 📸 **Image compression** — auto-resized to 800px, JPEG quality 75, ~3–5× savings
- ⚡ **Smart caching** — same disease + language never calls Claude twice
- 🛡️ **Rate limiting** — 10 requests/minute per IP via slowapi
- 🌾 **Farmer-first fallback** — farmer always gets useful advice, even if AI fails
- 🔍 **3-tier inference** — confidence-based routing between Haiku and Sonnet
- 📊 **Scan history & stats** — SQLite-backed history and aggregated analytics

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | Python 3.10+ · FastAPI · Uvicorn |
| ML inference | TensorFlow Lite (MobileNetV2) via Python 3.12 subprocess worker |
| AI advice | Anthropic Claude Haiku 4.5 (text) · Claude Sonnet 4.6 (vision fallback) |
| Database | SQLite via sqlite-utils |
| Auth | JWT (python-jose) · bcrypt (passlib) |
| Image processing | Pillow |
| Rate limiting | slowapi |

---

## Project Structure

```
backend/
├── main.py              # FastAPI app — all routes and middleware wiring
├── detector.py          # TFLite inference + MOCK mode fallback
├── analyzer.py          # Claude AI advice with 3-tier routing and caching
├── database.py          # SQLite scan history, user management, stats
├── auth.py              # JWT token creation and validation, password hashing
├── storage.py           # Image compression and disk storage
├── tflite_worker.py     # Python 3.12 subprocess worker for TFLite inference
├── treatments.json      # Organic treatment data for all 41 disease classes
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── model/
│   ├── plant_disease.tflite   # Trained MobileNetV2 model (add after training)
│   └── labels.txt             # 41 class names in model output order
└── data/
    ├── plantdoctor.db         # SQLite database (auto-created)
    └── images/                # Compressed leaf images (auto-created)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Python 3.12 with `tensorflow-cpu` installed (for TFLite inference)
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install TFLite inference dependencies under Python 3.12
py -3.12 -m pip install tensorflow-cpu numpy pillow

# Copy environment template and fill in your values
cp .env.example .env
```

### Running the server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

> **Note:** If `model/plant_disease.tflite` is absent, the server starts in **MOCK mode** and returns realistic random results — useful for development without the model file.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key for Claude AI calls |
| `PORT` | `8000` | Port the server listens on |
| `MAX_FILE_SIZE_MB` | `5` | Maximum upload size in megabytes |
| `RATE_LIMIT_PER_MINUTE` | `10` | Max `/diagnose` requests per IP per minute |
| `DB_PATH` | `./data/plantdoctor.db` | Path to the SQLite database file |
| `SECRET_KEY` | *(required)* | Secret used to sign JWT tokens — change in production |
| `IMAGES_DIR` | `./data/images` | Directory where compressed leaf images are saved |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | Health check — returns API status and model mode |
| `GET` | `/diseases` | No | All 41 diseases grouped by crop |
| `POST` | `/auth/register` | No | Create a new user account |
| `POST` | `/auth/login` | No | Log in with email + password, receive JWT token |
| `POST` | `/diagnose` | ✅ JWT | Upload leaf photo, get disease detection + treatment |
| `GET` | `/history` | ✅ JWT | Last 20 scans for the logged-in user |
| `GET` | `/me` | ✅ JWT | Current user's basic info |
| `GET` | `/images/{filename}` | ✅ JWT | Retrieve a saved leaf image by filename |
| `GET` | `/stats` | ✅ JWT | Aggregated scan stats, cache stats, model mode |

### Example: Diagnose a leaf

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Authorization: Bearer <your-jwt-token>" \
  -F "file=@leaf.jpg" \
  -F "language=hindi"
```

**Response:**
```json
{
  "scan_id": 42,
  "crop": "Tomato",
  "disease": "Late Blight",
  "confidence": 94.3,
  "health_score": 6,
  "is_healthy": false,
  "result_type": "model_plus_llm",
  "explanation": "Late blight is caused by the fungus-like organism Phytophthora infestans...",
  "treatment_steps": ["Mix 3g copper oxychloride in 1L water...", "..."],
  "urgency": "high",
  "image_filename": "20260327_5_42_a1b2c3d4.jpg",
  "compression_ratio": 4.2
}
```

---

## How It Works

The backend uses a **3-tier inference system** that balances speed, cost, and accuracy based on model confidence:

```
Farmer uploads leaf photo
         │
         ▼
  TFLite CNN model
  (MobileNetV2, 41 classes)
         │
    ┌────┴────────────────────────┐
    │                             │
 Healthy?                   Confidence?
    │                             │
    ▼                    ┌────────┼────────┐
  "Your plant          ≥90%    75–90%   60–75%
  looks great!"          │       │         │
                         ▼       ▼         ▼
                       Tier 1  Tier 2    Tier 3
```

### Tier 1 — High Confidence (≥ 90%)
- **Model:** Claude Haiku 4.5 (text only)
- **Result type:** `model_plus_llm`
- **Behavior:** Full organic treatment advice in the requested language. Result is **cached** — the same disease + language pair never calls Claude again.

### Tier 2 — Moderate Confidence (75–90%)
- **Model:** Claude Haiku 4.5 (text only)
- **Result type:** `model_plus_llm_caution`
- **Behavior:** Same as Tier 1 but includes a `caution` note advising the farmer to monitor the plant for 3 days. Also cached.

### Tier 3 — Low Confidence (60–75%)
- **Model:** Claude Sonnet 4.6 (vision — image + diagnosis sent together)
- **Result type:** `model_plus_vision_review`
- **Behavior:** The actual leaf photo is sent to Sonnet for visual verification. Sonnet can confirm or correct the TFLite diagnosis. Results are **not cached** — each image is unique.

### Fast Exits (no Claude call)
- **Healthy plant** → immediate "your plant looks healthy" response
- **Confidence < 60%** → "image unclear" response with retake tips

---

## Model

| Attribute | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Input size | 224 × 224 × 3 |
| Output classes | 41 |
| Training dataset | PlantVillage + 3 Rice diseases |
| Validation accuracy | **97.43%** |
| Format | TensorFlow Lite (`.tflite`) |
| Inference runtime | Python 3.12 subprocess (tensorflow-cpu) |

The model runs via `tflite_worker.py` — a separate Python 3.12 subprocess — because TensorFlow requires Python 3.12 while the FastAPI server runs on 3.10+. Image bytes are passed over stdin as base64; the worker returns JSON top-3 predictions over stdout.

---

## Disease Classes

All 41 classes recognized by the model, grouped by crop:

| Crop | Diseases |
|---|---|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Rice** | Bacterial Leaf Blight, Brown Spot, Leaf Smut |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Adding New Datasets

1. **Collect images** — add new disease class folders to your training directory following the `Crop___Disease_name` naming convention.
2. **Update `model/labels.txt`** — add the new class names in the exact order your retrained model outputs them.
3. **Retrain in Colab** — use MobileNetV2 transfer learning; target 224×224 input.
4. **Export to TFLite** — replace `model/plant_disease.tflite` with the new `.tflite` file.
5. **Update `treatments.json`** — add organic treatment entries for each new disease class.
6. Restart the server — it auto-detects the model and switches to REAL mode.

---

## License

MIT — see [LICENSE](LICENSE) for details.
