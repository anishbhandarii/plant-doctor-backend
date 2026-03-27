# analyzer.py — calls Claude haiku for organic treatment advice (text only, never images)

import base64
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
vision_calls: int = 0


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
async def get_treatment(detection: dict, language: str = "english", image_bytes: bytes = None) -> dict:
    """Return organic treatment advice for a detected plant disease.

    Tier 1 (>=90%): Haiku text-only — result_type model_plus_llm
    Tier 2 (75-90%): Haiku text-only with caution note — result_type model_plus_llm_caution
    Tier 3 (<75%):  Sonnet vision review — result_type model_plus_vision_review
    Skips Claude for healthy plants and low-confidence (<60%) detections.
    Caches tier-1/2 results; never caches vision results.
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

    # --- Determine tier ---
    if confidence >= 90:
        tier = 1
        result_type = "model_plus_llm"
    elif confidence >= 75:
        tier = 2
        result_type = "model_plus_llm_caution"
    else:
        tier = 3
        result_type = "model_plus_vision_review"

    # --- Tier 3: vision review (no caching — each image is unique) ---
    if tier == 3:
        result = await vision_review(detection, image_bytes, language)
        result["result_type"] = result_type
        return result

    # --- Fast exit 3: cache hit (tier 1 and 2 only) ---
    cache_key = f"{raw_label}_{language}"
    if cache_key in _cache:
        cache_hits += 1
        return _cache[cache_key]

    # --- Claude haiku text call (tier 1 and 2) ---
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
        # Non-Latin scripts (Hindi, Bengali, etc.) need more tokens per word
        max_tok = 800 if language.lower() == "english" else 1200
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tok,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        claude_calls += 1
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["status"] = "diagnosed"
        result["result_type"] = result_type
        # Tier 2: add caution note
        if tier == 2:
            result["caution"] = (
                f"Confidence is moderate ({confidence}%). "
                "This diagnosis is likely correct but please monitor "
                "the plant closely over the next 3 days."
            )
        _cache[cache_key] = result
        return result

    except Exception as e:
        print(f"[analyzer] Claude call failed: {e} — using fallback")
        fallback = {
            "status": "diagnosed",
            "result_type": result_type,
            "explanation": f"{disease} detected on your {crop}.",
            "severity_message": f"Urgency is {entry.get('urgency', 'medium')}. Act as soon as possible.",
            "treatment_steps": entry.get("organic", ["Consult a local agronomist."]),
            "local_materials": entry.get("local_materials", ""),
            "prevention": entry.get("prevention", []),
            "urgency": entry.get("urgency", "medium"),
            "when_to_escalate": "If the disease spreads to more than half the plant, seek expert help immediately."
        }
        if tier == 2:
            fallback["caution"] = (
                f"Confidence is moderate ({confidence}%). "
                "This diagnosis is likely correct but please monitor "
                "the plant closely over the next 3 days."
            )
        _cache[cache_key] = fallback
        return fallback


# ---------------------------------------------------------------------------
# Tier-3 vision review using Claude Sonnet
# ---------------------------------------------------------------------------
async def vision_review(detection: dict, image_bytes: bytes, language: str) -> dict:
    """Send image + TFLite result to Claude Sonnet for visual verification.

    Falls back to a haiku text call if image_bytes is missing or the vision
    call fails — the farmer always gets a useful response.
    """
    global vision_calls, claude_calls

    crop       = detection["crop"]
    disease    = detection["disease"]
    confidence = detection["confidence"]
    raw_label  = detection["raw_label"]
    entry      = TREATMENTS.get(raw_label, {})

    if image_bytes is None:
        print("[analyzer] Vision review requested but no image provided — falling back to haiku")
        # Treat as tier-2 fallback: haiku text + caution
        return await _haiku_fallback(detection, language, entry, result_type="model_plus_vision_review")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode()
        }
    }

    text_block = {
        "type": "text",
        "text": f"""A plant disease classifier identified this image as:
Crop: {crop}
Disease: {disease}
Confidence: {confidence}% (LOW — needs verification)

Please examine the image carefully and:
1. Confirm or correct this diagnosis
2. If corrected, explain what you see in the image
3. Provide organic treatment advice in {language}

Respond ONLY as valid JSON:
{{
  "verified": true or false,
  "original_diagnosis": "{disease}",
  "confirmed_diagnosis": "disease name if different, or same if confirmed",
  "explanation": "what you see in the image — 2 sentences",
  "severity_message": "impact on harvest if untreated",
  "treatment_steps": ["step with quantity", "step 2", "step 3"],
  "local_materials": "comma list",
  "prevention": ["tip 1", "tip 2", "tip 3"],
  "urgency": "low|medium|high",
  "when_to_escalate": "when to seek expert help",
  "vision_note": "one sentence on why the confidence was low"
}}"""
    }

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": [image_block, text_block]}]
        )
        vision_calls += 1
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["status"] = "diagnosed"
        return result

    except Exception as e:
        print(f"[analyzer] Vision review failed: {e} — falling back to haiku")
        return await _haiku_fallback(detection, language, entry, result_type="model_plus_vision_review")


async def _haiku_fallback(detection: dict, language: str, entry: dict, result_type: str) -> dict:
    """Call haiku text-only as a fallback when vision review is unavailable."""
    global claude_calls

    crop     = detection["crop"]
    disease  = detection["disease"]
    confidence = detection["confidence"]
    client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    user_prompt = f"""
Crop: {crop}
Disease detected: {disease}
Confidence: {confidence}% (low — treat as uncertain)

Verified organic treatments:
{json.dumps(entry.get('organic', []), indent=2)}

Respond in {language}.
Return this exact JSON with no other text:
{{
  "explanation": "2 sentence plain explanation",
  "severity_message": "impact on harvest if untreated",
  "treatment_steps": ["step with quantity", "step 2", "step 3"],
  "local_materials": "comma list",
  "prevention": ["tip 1", "tip 2", "tip 3"],
  "urgency": "{entry.get('urgency', 'medium')}",
  "when_to_escalate": "when to seek expert help"
}}"""
    try:
        max_tok = 800 if language.lower() == "english" else 1200
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tok,
            messages=[{"role": "user", "content": user_prompt}]
        )
        claude_calls += 1
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["status"] = "diagnosed"
        result["result_type"] = result_type
        result["caution"] = (
            f"Confidence is moderate ({confidence}%). "
            "This diagnosis is likely correct but please monitor "
            "the plant closely over the next 3 days."
        )
        return result
    except Exception as e:
        print(f"[analyzer] Haiku fallback also failed: {e}")
        return {
            "status": "diagnosed",
            "result_type": result_type,
            "explanation": f"{disease} detected on your {crop}.",
            "severity_message": f"Urgency is {entry.get('urgency', 'medium')}. Act as soon as possible.",
            "treatment_steps": entry.get("organic", ["Consult a local agronomist."]),
            "local_materials": entry.get("local_materials", ""),
            "prevention": entry.get("prevention", []),
            "urgency": entry.get("urgency", "medium"),
            "when_to_escalate": "If the disease spreads to more than half the plant, seek expert help immediately.",
            "caution": f"Confidence is low ({confidence}%). Please seek expert confirmation.",
        }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def get_cache_stats() -> dict:
    """Return current Claude call count, cache hit count, and vision call count."""
    return {
        "claude_calls":  claude_calls,
        "cache_hits":    cache_hits,
        "vision_calls":  vision_calls,
    }
