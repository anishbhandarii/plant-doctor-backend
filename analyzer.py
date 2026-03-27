# analyzer.py — calls Claude haiku for organic treatment advice (text only, never images)

import json
import os

import anthropic

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
with open("./treatments.json", "r") as _f:
    TREATMENTS: dict = json.load(_f)

_cache: dict = {}
claude_calls: int = 0
cache_hits: int = 0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
async def get_treatment(detection: dict, language: str = "english") -> dict:
    """Return organic treatment advice for a detected plant disease.

    Skips Claude for healthy plants and low-confidence detections.
    Caches results so the same disease+language pair never calls Claude twice.
    Falls back to treatments.json data if the Claude API call fails.
    """
    global cache_hits, claude_calls

    raw_label  = detection["raw_label"]
    crop       = detection["crop"]
    disease    = detection["disease"]
    confidence = detection["confidence"]
    is_healthy = detection["is_healthy"]

    entry = TREATMENTS.get(raw_label, {})

    # --- Fast exit 1: healthy plant ---
    if is_healthy:
        return {
            "status": "healthy",
            "explanation": f"Your {crop} looks healthy! Keep up the good work.",
            "severity_message": "No disease detected. No action needed.",
            "treatment_steps": [],
            "local_materials": "",
            "prevention": entry.get("prevention", []),
            "urgency": "none",
            "when_to_escalate": "Monitor weekly. See an expert if you notice any changes."
        }

    # --- Fast exit 2: low confidence ---
    if confidence < 60:
        return {
            "status": "unclear",
            "explanation": "The image was not clear enough to make a confident diagnosis.",
            "severity_message": "Please retake the photo for a better result.",
            "treatment_steps": [],
            "local_materials": "",
            "prevention": [],
            "urgency": "unknown",
            "when_to_escalate": "If the plant looks very sick, consult a local agronomist.",
            "retake_tips": [
                "Take photo in bright natural daylight, not indoors",
                "Hold phone 20-30cm away from the leaf",
                "Make sure the diseased area fills the frame",
                "Keep the phone steady to avoid blur"
            ]
        }

    # --- Fast exit 3: cache hit ---
    cache_key = f"{raw_label}_{language}"
    if cache_key in _cache:
        cache_hits += 1
        return _cache[cache_key]

    # --- Claude API call ---
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are an agricultural advisor helping smallholder farmers in rural areas.
Always respond in the language the user specifies — no exceptions.
Use simple words a farmer with basic education can understand.
Always prioritize organic treatments using materials available in any village.
Give specific quantities for every treatment step.
Be encouraging and practical.
Respond ONLY with valid JSON. No markdown fences, no text outside the JSON."""

    user_prompt = f"""
Crop: {crop}
Disease detected: {disease}
Confidence: {confidence}%
Urgency: {entry.get('urgency', 'medium')}
Spread risk: {entry.get('spread_risk', 'medium')}
Affects: {entry.get('affects', 'leaves')}

Verified organic treatments for this disease:
{json.dumps(entry.get('organic', []), indent=2)}

Respond in {language}.
Return this exact JSON with no other text:
{{
  "explanation": "2 sentence plain explanation of what this disease is and what causes it",
  "severity_message": "one sentence on what happens to the harvest if untreated",
  "treatment_steps": ["step with exact quantity", "step 2", "step 3", "step 4"],
  "local_materials": "comma separated list of materials to gather",
  "prevention": ["tip 1", "tip 2", "tip 3"],
  "urgency": "{entry.get('urgency', 'medium')}",
  "when_to_escalate": "one sentence on when to seek expert help"
}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        claude_calls += 1
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["status"] = "diagnosed"
        _cache[cache_key] = result
        return result

    except Exception as e:
        print(f"[analyzer] Claude call failed: {e} — using fallback")
        fallback = {
            "status": "diagnosed",
            "explanation": f"{disease} detected on your {crop}.",
            "severity_message": f"Urgency is {entry.get('urgency', 'medium')}. Act as soon as possible.",
            "treatment_steps": entry.get("organic", ["Consult a local agronomist."]),
            "local_materials": entry.get("local_materials", ""),
            "prevention": entry.get("prevention", []),
            "urgency": entry.get("urgency", "medium"),
            "when_to_escalate": "If the disease spreads to more than half the plant, seek expert help immediately."
        }
        _cache[cache_key] = fallback
        return fallback


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def get_cache_stats() -> dict:
    """Return current Claude call count and cache hit count."""
    return {
        "claude_calls": claude_calls,
        "cache_hits": cache_hits
    }
