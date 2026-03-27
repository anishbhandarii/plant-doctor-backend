# PlantDoctor Backend

## What this project is
A FastAPI backend for a plant disease detection app targeting rural farmers
in South Asia. Farmers photograph a sick leaf, the app detects the disease
and gives organic treatment advice in their local language.

## How it works
1. Farmer uploads a leaf photo
2. TFLite CNN model (MobileNetV2, 41 classes) detects the disease on the server
3. Claude claude-haiku-4-5 is called with TEXT ONLY (never images) to generate
   organic treatment advice in the farmer's language
4. Result is saved to SQLite and returned to the farmer

## 41 Disease Classes
38 PlantVillage classes + 3 Rice diseases for South Asia:
Rice___Bacterial_leaf_blight, Rice___Brown_spot, Rice___Leaf_smut

## Stack
- Python 3.10+
- FastAPI + uvicorn
- TFLite (model at ./model/plant_disease.tflite)
- Anthropic claude-haiku-4-5 — text only, never vision
- SQLite via sqlite-utils
- slowapi for rate limiting

## Project files (build in this order)
- requirements.txt
- model/labels.txt       — 41 class names, order must match model output
- treatments.json        — organic treatments for all 41 classes
- detector.py            — TFLite inference + mock mode fallback
- analyzer.py            — Claude text advice + caching
- database.py            — SQLite scan history
- auth.py                — JWT tokens and password hashing
- main.py                — FastAPI app wiring everything together
- .env.example
- .gitignore

## Model status
- plant_disease.tflite is NOT present yet — it is being trained in Google Colab
- detector.py MUST run in MOCK mode when the model file is missing
- MOCK mode returns a realistic random result from the 41 classes
- When the real model is added later, it auto-switches to REAL mode

## Coding rules
- load_dotenv() must be the very first line in main.py before any other imports
- Always use async/await for FastAPI endpoints
- Claude API: text only, model claude-haiku-4-5, max_tokens 800
- Never send images to Claude under any circumstance
- Never hardcode API keys — always use os.getenv()
- Always handle errors gracefully — never expose stack traces to the client
- Add a one-line comment at the top of every file explaining what it does
- Organic treatments must use locally available materials with specific quantities

## Environment variables (.env)
ANTHROPIC_API_KEY=
PORT=8000
MAX_FILE_SIZE_MB=5
RATE_LIMIT_PER_MINUTE=10
DB_PATH=./data/plantdoctor.db
SECRET_KEY=

## Key design decisions
- Caching: same disease + language combo never calls Claude twice
- Fast exits: healthy plants and low confidence (<60%) skip Claude entirely
- Fallback: if Claude API fails, build response from treatments.json directly
- Farmer never sees an error — always gets a useful response
```