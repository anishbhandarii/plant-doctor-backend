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
from auth import create_token, get_current_user, hash_password, require_admin, verify_password
from database import (
    create_user, delete_scan, get_all_users, get_history, get_stats,
    get_user_by_email, get_user_by_id, get_user_count_by_role,
    get_user_stats, save_scan, toggle_user_active, update_user_language,
)
from detector import detect_disease, get_mode
from fastapi.responses import FileResponse
from storage import get_image_path, save_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024
SUPPORTED_LANGUAGES = ["english", "hindi", "nepali", "french","german", "korean", "chinese"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "farmer"
    preferred_language: str = "english"
    region: str = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: str
    full_name: str
    role: str
    preferred_language: str

class LanguageUpdateRequest(BaseModel):
    language: str

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
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    with open("treatments.json", encoding="utf-8") as f:
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
    if body.role not in ["farmer", "admin"]:
        raise HTTPException(status_code=400, detail="Role must be farmer or admin")
    if body.preferred_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language must be one of: {SUPPORTED_LANGUAGES}")
    try:
        user = create_user(
            email=body.email,
            password_hash=hash_password(body.password),
            full_name=body.full_name,
            role=body.role,
            preferred_language=body.preferred_language,
            region=body.region,
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
    token = create_token({
        "sub":                user["email"],
        "user_id":            user["id"],
        "role":               user.get("role", "farmer"),
        "preferred_language": user.get("preferred_language", "english"),
    })
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user.get("role", "farmer"),
        preferred_language=user.get("preferred_language", "english"),
    )

# ---------------------------------------------------------------------------
# Protected routes — require valid JWT
# ---------------------------------------------------------------------------
@app.post("/diagnose")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '10')}/minute")
async def diagnose(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    """Accept a leaf image, run disease detection, return organic treatment advice."""
    # Use form language if provided, else fall back to user's preferred language
    if not language:
        language = current_user.get("preferred_language", "english")
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted")

    # Read and validate file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Max 5MB.")

    try:
        detection  = detect_disease(image_bytes)
        advice     = await get_treatment(detection, language, image_bytes)
        user_id    = current_user["user_id"]
        session_id = str(user_id)

        # If vision corrected the diagnosis, override TFLite values before saving
        if (
            advice.get("result_type") == "model_plus_vision_review"
            and advice.get("verified") is False
            and advice.get("confirmed_crop")
            and advice.get("confirmed_disease")
        ):
            detection["crop"]      = advice["confirmed_crop"]
            detection["disease"]   = advice["confirmed_disease"]
            detection["raw_label"] = (
                f"{advice['confirmed_crop']}___"
                f"{advice['confirmed_disease'].replace(' ', '_')}"
            )

        # Save image first (use unix timestamp as temp scan_id), then save scan record
        image_info = save_image(image_bytes, user_id, int(datetime.now().timestamp()))

        # Build the full response so it can be stored and returned identically
        full_result = {
            "scan_id":            0,  # placeholder — updated after insert
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "user_id":            user_id,
            "language":           language,
            **detection,
            **advice,
            "image_filename":     image_info["image_filename"],
            "original_size_kb":   image_info["original_size_kb"],
            "compressed_size_kb": image_info["compressed_size_kb"],
            "compression_ratio":  image_info["compression_ratio"],
        }

        scan_id = save_scan(detection, advice, language, session_id, image_info, full_response=full_result)
        full_result["scan_id"] = scan_id
        return full_result
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
    """Return the last 20 scans for the logged-in user, with full treatment data merged in."""
    user_id = current_user["user_id"]
    scans = get_history(session_id=str(user_id), limit=20)

    enriched = []
    for scan in scans:
        enriched_scan = {k: v for k, v in scan.items() if k != "full_response"}
        if scan.get("full_response"):
            enriched_scan.update(scan["full_response"])
        # Preserve the real database id even if older stored payloads contain scan_id=0.
        enriched_scan["scan_id"] = scan["scan_id"]
        enriched.append(enriched_scan)

    return {"user_id": user_id, "scans": enriched}


@app.delete("/history/{scan_id}")
async def delete_history_scan(scan_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a scan owned by the logged-in user."""
    deleted = delete_scan(scan_id=scan_id, user_id=current_user["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"message": "Scan deleted"}


@app.get("/me/stats")
async def me_stats(current_user: dict = Depends(get_current_user)):
    """Return scan statistics for the currently logged-in user."""
    return get_user_stats(current_user["user_id"])


@app.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the current user's basic info including role and language."""
    user = get_user_by_id(current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id":            user["id"],
        "email":              user["email"],
        "full_name":          user["full_name"],
        "role":               user.get("role", "farmer"),
        "preferred_language": user.get("preferred_language", "english"),
        "region":             user.get("region"),
    }


@app.patch("/me/language")
async def update_language(body: LanguageUpdateRequest, current_user: dict = Depends(get_current_user)):
    """Update the current user's preferred language."""
    if body.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language must be one of: {SUPPORTED_LANGUAGES}")
    user = update_user_language(current_user["user_id"], body.language)
    return {
        "user_id":            user["id"],
        "email":              user["email"],
        "full_name":          user["full_name"],
        "role":               user.get("role", "farmer"),
        "preferred_language": user.get("preferred_language", "english"),
        "region":             user.get("region"),
    }



@app.get("/images/test")
async def images_test():
    """Debug endpoint — lists files in data/images/ with no auth required."""
    try:
        files = os.listdir("./data/images")
    except FileNotFoundError:
        files = []
    return {"count": len(files), "files": sorted(files)}


@app.get("/images/{filename}")
async def serve_image(filename: str, current_user: dict = Depends(get_current_user)):
    """Return a saved leaf image by filename. Requires login."""
    path = get_image_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/stats")
async def stats(current_user: dict = Depends(require_admin)):
    """Return aggregated scan stats, cache stats, and model mode. Admin only."""
    return {
        **get_stats(),
        **get_cache_stats(),
        "model_mode": get_mode(),
    }

# ---------------------------------------------------------------------------
# Admin routes — require admin role
# ---------------------------------------------------------------------------
@app.get("/admin/users")
async def admin_list_users(
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(require_admin),
):
    """Return all users (admin only)."""
    return get_all_users(limit=limit, offset=offset)


@app.get("/admin/users/{user_id}")
async def admin_get_user(user_id: int, current_user: dict = Depends(require_admin)):
    """Return a single user by id (admin only)."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/admin/users/{user_id}/toggle")
async def admin_toggle_user(user_id: int, current_user: dict = Depends(require_admin)):
    """Flip is_active for a user (admin only)."""
    try:
        return toggle_user_active(user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/admin/overview")
async def admin_overview(current_user: dict = Depends(require_admin)):
    """Return system overview: user counts, scan stats, model mode (admin only)."""
    return {
        "user_counts":         get_user_count_by_role(),
        "scan_stats":          get_stats(),
        "model_mode":          get_mode(),
        "supported_languages": SUPPORTED_LANGUAGES,
        "version":             "1.0.0",
    }


# ---------------------------------------------------------------------------
# Public routes — no token required
# ---------------------------------------------------------------------------
@app.get("/diseases")
async def diseases():
    """Return all known diseases grouped by crop name."""
    with open("treatments.json", encoding="utf-8") as f:
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
