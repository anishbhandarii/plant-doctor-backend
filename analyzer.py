# analyzer.py — calls Claude haiku for organic treatment advice (text only, never images)

import base64
import json
import os

import anthropic

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
with open("./treatments.json", "r", encoding="utf-8") as _f:
    TREATMENTS: dict = json.load(_f)

_cache: dict = {}
claude_calls: int = 0
cache_hits: int = 0
vision_calls: int = 0

# Mojibake that can appear when UTF-8 bytes are decoded as latin-1/cp1252
_MOJIBAKE = {
    'â€"':  '—',
    'â€™':  '\u2019',
    'â€œ':  '\u201c',
    'â€\x9d': '\u201d',
    'â€':   '\u201d',
    'â€¦':  '…',
    'Ã©':   'é',
    'Ã¨':   'è',
    'Ã ':   'à',
}


def _clean_str(s: str) -> str:
    for bad, good in _MOJIBAKE.items():
        s = s.replace(bad, good)
    return s


def _clean(obj):
    """Recursively fix mojibake in every string inside a dict/list."""
    if isinstance(obj, str):
        return _clean_str(obj)
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


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

    entry = _clean(TREATMENTS.get(raw_label, {}))

    # --- Fast exit 1: healthy plant ---
    if is_healthy:
        return {
            "status": "healthy",
            "result_type": "healthy",
            "explanation": f"Your {crop} looks healthy! Keep up the good work.",
            "severity_message": "No disease detected. No action needed.",
            "treatment_steps": [],
            "local_materials": "",
            "prevention": entry.get("prevention", []),
            "urgency": "none",
            "when_to_escalate": "Monitor weekly. See an expert if you notice any changes.",
            "inorganic": {},
        }

    # --- Fast exit 2: low confidence ---
    if confidence < 60:
        return {
            "status": "unclear",
            "result_type": "unclear",
            "explanation": "The image was not clear enough to make a confident diagnosis.",
            "severity_message": "Please retake the photo for a better result.",
            "treatment_steps": [],
            "local_materials": "",
            "prevention": [],
            "urgency": "unknown",
            "when_to_escalate": "If the plant looks very sick, consult a local agronomist.",
            "inorganic": {},
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

    system_prompt = """You are an agricultural advisor helping smallholder farmers in rural areas. Do not be too harsh with the advices but be detailed.

CRITICAL LANGUAGE RULE — THIS IS YOUR MOST IMPORTANT INSTRUCTION:
You MUST respond entirely in the language specified by the user.
This is non-negotiable. Do not switch languages under any circumstances.

Language rules:
- If language is "nepali": respond in Nepali (नेपाली भाषामा लेख्नुहोस्)
  Use Devanagari script. Nepali is NOT Hindi. They are different languages.
  Nepali uses different vocabulary and grammar than Hindi.
- If language is "hindi": respond in Hindi (हिंदी में लिखें)
  Use Devanagari script.
- If language is "chinese": respond in Simplified Chinese (用简体中文回复)
- If language is "korean": respond in Korean (한국어로 답변하세요)
- If language is "french": respond in French (Répondez en français)
- If language is "german": respond in German (Antworten Sie auf Deutsch)
- If language is "english": respond in English

Every single field in the JSON response must be in the specified language.
The explanation, severity_message, treatment_steps, local_materials,
prevention, and when_to_escalate must ALL be in the target language.
Do not mix languages. Do not default to Hindi for Nepali requests.

Other rules:
- Use simple words a farmer with basic education can understand
- Always prioritize organic treatments
- Give specific quantities for every treatment step
- Be encouraging and practical
- Respond ONLY with valid JSON. No markdown, no text outside JSON."""

    user_prompt = f"""
Language for response: {language}
IMPORTANT: You MUST write your entire response in {language}.
{"If language is nepali: Write in Nepali language using Nepali words, NOT Hindi." if language == "nepali" else ""}
{"If language is hindi: Write in Hindi language." if language == "hindi" else ""}

Crop: {crop}
Disease detected: {disease}
Confidence: {confidence}%
Urgency: {entry.get('urgency', 'medium')}
Spread risk: {entry.get('spread_risk', 'medium')}
Affects: {entry.get('affects', 'leaves')}

Verified organic treatments:
{json.dumps(entry.get('organic', []), indent=2)}

Respond ONLY in {language} language as valid JSON:
{{
  "explanation": "...",
  "severity_message": "...",
  "treatment_steps": ["...", "...", "..."],
  "local_materials": "...",
  "prevention": ["...", "...", "..."],
  "urgency": "{entry.get('urgency', 'medium')}",
  "when_to_escalate": "..."
}}"""

    HIGH_TOKEN_LANGUAGES = ["nepali", "hindi", "chinese", "korean"]
    MID_TOKEN_LANGUAGES = ["french", "german"]

    try:
        max_tok = (
            1500 if language in HIGH_TOKEN_LANGUAGES
            else 1200 if language in MID_TOKEN_LANGUAGES
            else 800
        )
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
        result["inorganic"] = entry.get("inorganic", {})
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
            "when_to_escalate": "If the disease spreads to more than half the plant, seek expert help immediately.",
            "inorganic": entry.get("inorganic", {}),
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
    entry      = _clean(TREATMENTS.get(raw_label, {}))

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

    _vision_high_tok = ["nepali", "hindi", "chinese", "korean"]
    _vision_mid_tok  = ["french", "german"]
    vision_max_tok = (
        1500 if language in _vision_high_tok
        else 1200 if language in _vision_mid_tok
        else 800
    )

    text_block = {
        "type": "text",
        "text": f"""A plant disease classifier identified this image as:
Crop: {crop}
Disease: {disease}
Confidence: {confidence}% (LOW — needs verification)

Language for response: {language}
IMPORTANT: You MUST write your entire response in {language}.
{"Write in Nepali language using Nepali words, NOT Hindi." if language == "nepali" else ""}

Please examine the image carefully and:
1. Confirm or correct this diagnosis
2. If corrected, explain what you see in the image
3. Provide organic treatment advice in {language}

Respond ONLY in {language} as valid JSON:
{{
  "verified": true or false,
  "original_diagnosis": "{disease}",
  "confirmed_diagnosis": "full diagnosis string e.g. Grape Downy Mildew",
  "confirmed_crop": "crop name only — no extra words, no latin names, no parentheses. Good: Grape  Bad: Grapevine (Vitis vinifera)",
  "confirmed_disease": "disease name only — no parentheses, no scientific names. Good: Downy Mildew  Bad: Downy Mildew (Plasmopara viticola)",
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
            max_tokens=vision_max_tok,
            messages=[{"role": "user", "content": [image_block, text_block]}]
        )
        vision_calls += 1
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["status"] = "diagnosed"
        result["inorganic"] = entry.get("inorganic", {})
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

    _fb_high_tok = ["nepali", "hindi", "chinese", "korean"]
    _fb_mid_tok  = ["french", "german"]
    fb_max_tok = (
        1500 if language in _fb_high_tok
        else 1200 if language in _fb_mid_tok
        else 800
    )

    user_prompt = f"""
Language for response: {language}
IMPORTANT: You MUST write your entire response in {language}.
{"Write in Nepali language using Nepali words, NOT Hindi." if language == "nepali" else ""}

Crop: {crop}
Disease detected: {disease}
Confidence: {confidence}% (low — treat as uncertain)

Verified organic treatments:
{json.dumps(entry.get('organic', []), indent=2)}

Respond ONLY in {language} language as valid JSON:
{{
  "explanation": "...",
  "severity_message": "...",
  "treatment_steps": ["...", "...", "..."],
  "local_materials": "...",
  "prevention": ["...", "...", "..."],
  "urgency": "{entry.get('urgency', 'medium')}",
  "when_to_escalate": "..."
}}"""
    try:
        max_tok = fb_max_tok
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
        result["inorganic"] = entry.get("inorganic", {})
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
            "inorganic": entry.get("inorganic", {}),
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
