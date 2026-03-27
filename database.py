# database.py — SQLite scan history and stats using sqlite-utils

import os
from datetime import datetime, timezone

import sqlite_utils

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "./data/plantdoctor.db")

# Ensure the data directory exists before opening the database
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

db = sqlite_utils.Database(DB_PATH)

# Create scans table if it does not already exist
if "scans" not in db.table_names():
    db["scans"].create({
        "id":           int,
        "timestamp":    str,
        "session_id":   str,
        "crop":         str,
        "disease":      str,
        "confidence":   float,
        "health_score": int,
        "is_healthy":   int,   # 0 or 1
        "language":     str,
        "urgency":      str,
        "spread_risk":  str,
        "mode":         str,   # "real" or "mock"
        "result_type":  str,   # tier label
    }, pk="id")
else:
    # Add result_type column to existing tables that predate this field
    try:
        db.execute("ALTER TABLE scans ADD COLUMN result_type TEXT DEFAULT 'model_plus_llm'")
    except Exception:
        pass  # column already exists — safe to ignore

# Create users table if it does not already exist
if "users" not in db.table_names():
    db["users"].create({
        "id":            int,
        "email":         str,
        "password_hash": str,
        "full_name":     str,
        "created_at":    str,
        "is_active":     int,   # 1 = active, 0 = disabled
    }, pk="id")
    db["users"].create_index(["email"], unique=True)

print(f"Database ready: {DB_PATH}")


# ---------------------------------------------------------------------------
# Function 1: save a scan record
# ---------------------------------------------------------------------------
def save_scan(detection: dict, advice: dict, language: str, session_id: str) -> int:
    """Insert one scan result into the scans table and return its new id."""
    # Pull spread_risk from the treatments data if analyzer loaded it,
    # otherwise leave blank — we import here to avoid circular issues at top level
    from analyzer import TREATMENTS
    entry = TREATMENTS.get(detection.get("raw_label", ""), {})

    row = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "session_id":   session_id,
        "crop":         detection["crop"],
        "disease":      detection["disease"],
        "confidence":   detection["confidence"],
        "health_score": detection["health_score"],
        "is_healthy":   1 if detection["is_healthy"] else 0,
        "language":     language,
        "urgency":      advice.get("urgency", "unknown"),
        "spread_risk":  entry.get("spread_risk", ""),
        "mode":         detection["mode"],
        "result_type":  advice.get("result_type", "model_plus_llm"),
    }

    db["scans"].insert(row)
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# Function 2: get scan history for a session
# ---------------------------------------------------------------------------
def get_history(session_id: str, limit: int = 20) -> list:
    """Return up to {limit} most recent scans for a session, newest first."""
    rows = db.execute(
        """
        SELECT timestamp, crop, disease, health_score, urgency, language, mode, confidence
        FROM scans
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [session_id, limit]
    ).fetchall()

    columns = ["timestamp", "crop", "disease", "health_score", "urgency", "language", "mode", "confidence"]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Function 3: aggregate stats across all scans
# ---------------------------------------------------------------------------
def get_stats() -> dict:
    """Return total scans, healthy count, average confidence, and top 5 diseases."""
    try:
        totals = db.execute(
            "SELECT COUNT(*), SUM(is_healthy), AVG(confidence) FROM scans"
        ).fetchone()

        total_scans    = totals[0] or 0
        healthy_count  = int(totals[1] or 0)
        avg_confidence = round(totals[2], 1) if totals[2] is not None else 0.0

        top_diseases_rows = db.execute(
            """
            SELECT disease, COUNT(*) as cnt
            FROM scans
            WHERE is_healthy = 0
            GROUP BY disease
            ORDER BY cnt DESC
            LIMIT 5
            """
        ).fetchall()

        top_diseases = [{"disease": row[0], "count": row[1]} for row in top_diseases_rows]

    except Exception:
        # Empty or uninitialised database — return safe zeros
        total_scans    = 0
        healthy_count  = 0
        avg_confidence = 0.0
        top_diseases   = []

    return {
        "total_scans":    total_scans,
        "healthy_count":  healthy_count,
        "avg_confidence": avg_confidence,
        "top_diseases":   top_diseases,
    }


# ---------------------------------------------------------------------------
# User management functions
# ---------------------------------------------------------------------------
def create_user(email: str, password_hash: str, full_name: str) -> dict:
    """Insert a new user and return their record (without password_hash).

    Raises ValueError if the email is already registered.
    """
    if db["users"].count_where("email = ?", [email]) > 0:
        raise ValueError("Email already registered")

    row = {
        "email":         email,
        "password_hash": password_hash,
        "full_name":     full_name,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "is_active":     1,
    }
    db["users"].insert(row)
    user_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    return {
        "id":         user_id,
        "email":      email,
        "full_name":  full_name,
        "created_at": row["created_at"],
        "is_active":  1,
    }


def get_user_by_email(email: str) -> dict | None:
    """Return the full user row (including password_hash) or None if not found."""
    rows = list(db["users"].rows_where("email = ?", [email], limit=1))
    return rows[0] if rows else None


def get_user_by_id(user_id: int) -> dict | None:
    """Return the user row without password_hash, or None if not found."""
    rows = list(db["users"].rows_where("id = ?", [user_id], limit=1))
    if not rows:
        return None
    user = rows[0]
    user.pop("password_hash", None)
    return user
