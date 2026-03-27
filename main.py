# main.py — FastAPI application, all routes and middleware

# load_dotenv MUST be first — before any other import reads os.getenv()
from dotenv import load_dotenv
load_dotenv()

import json
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from analyzer import get_cache_stats, get_treatment
from auth import create_token, get_current_user, hash_password, verify_password
from database import create_user, get_history, get_stats, get_user_by_email, save_scan
from detector import detect_disease, get_mode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: str
    full_name: str

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="PlantDoctor API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    with open("treatments.json") as f:
        count = len(json.load(f))
    print("-------------------------------")
    print("  PlantDoctor API started")
    print(f"  Model mode : {get_mode()}")
    print(f"  Diseases   : {count}")
    print("  Version    : 1.0.0")
    print("-------------------------------")

# ---------------------------------------------------------------------------
# Auth routes — no token required
# ---------------------------------------------------------------------------
@app.post("/auth/register")
async def register(body: RegisterRequest):
    """Create a new user account."""
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        user = create_user(
            email=body.email,
            password_hash=hash_password(body.password),
            full_name=body.full_name,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Email already registered")
    return {"message": "Account created successfully", "email": user["email"]}


@app.post("/auth/login", response_model=LoginResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Log in with email and password, receive a JWT access token."""
    user = get_user_by_email(form.username)  # OAuth2 form uses 'username' for the email field
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token({"sub": user["email"], "user_id": user["id"]})
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
    )

# ---------------------------------------------------------------------------
# Protected routes — require valid JWT
# ---------------------------------------------------------------------------
@app.post("/diagnose")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '10')}/minute")
async def diagnose(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(default="english"),
    current_user: dict = Depends(get_current_user),
):
    """Accept a leaf image, run disease detection, return organic treatment advice."""
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted")

    # Read and validate file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Max 5MB.")

    try:
        detection = detect_disease(image_bytes)
        advice    = await get_treatment(detection, language, image_bytes)
        session_id = str(current_user["user_id"])
        scan_id    = save_scan(detection, advice, language, session_id)

        return {
            "scan_id":   scan_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id":   current_user["user_id"],
            "language":  language,
            **detection,
            **advice,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[diagnose] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong analysing the image. Please try again.",
        )


@app.get("/history")
async def history(current_user: dict = Depends(get_current_user)):
    """Return the last 20 scans for the logged-in user."""
    user_id = current_user["user_id"]
    scans = get_history(session_id=str(user_id), limit=20)
    return {"user_id": user_id, "scans": scans}


@app.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the current user's basic info."""
    return {"user_id": current_user["user_id"], "email": current_user["email"]}


@app.get("/stats")
async def stats(current_user: dict = Depends(get_current_user)):
    """Return aggregated scan stats, cache stats, and model mode. Requires login."""
    return {
        **get_stats(),
        **get_cache_stats(),
        "model_mode": get_mode(),
    }

# ---------------------------------------------------------------------------
# Public routes — no token required
# ---------------------------------------------------------------------------
@app.get("/diseases")
async def diseases():
    """Return all known diseases grouped by crop name."""
    with open("treatments.json") as f:
        treatments = json.load(f)

    grouped: dict[str, list[str]] = {}
    for raw_label, entry in treatments.items():
        # Skip healthy classes
        if "healthy" in raw_label.lower():
            continue

        # Clean crop name (mirrors _parse_label logic in detector.py)
        crop_raw = raw_label.split("___")[0]
        crop = (
            crop_raw
            .replace("_(maize)", "")
            .replace("(including_sour)", "")
            .replace(",_bell", "")
            .replace("_", " ")
            .strip()
            .title()
        )

        # Clean disease name from common_name (already human-readable)
        disease = entry.get("common_name", "").replace(f"{crop} ", "").strip()

        grouped.setdefault(crop, []).append(disease)

    return grouped


@app.get("/health")
async def health():
    """Public health check — confirms the API is running and shows model mode."""
    return {"status": "ok", "model_mode": get_mode(), "version": "1.0.0"}
