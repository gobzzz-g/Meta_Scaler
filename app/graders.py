from typing import Any

# Strict bounds: scores must be in (0, 1), never exactly 0.0 or 1.0
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99

def _clamp(score: float) -> float:
    return float(max(_MIN_SCORE, min(_MAX_SCORE, score)))

def _safe_lower(val: Any) -> str:
    return (val or "").lower()

def grade_easy(action: Any, expected_category: str) -> float:
    score = 0.0
    category = _safe_lower(getattr(action, "category", ""))
    
    if category == _safe_lower(expected_category):
        score += 1.0
        
    return _clamp(score)

def grade_medium(action: Any, expected_category: str) -> float:
    score = 0.0
    category = _safe_lower(getattr(action, "category", ""))
    response = _safe_lower(getattr(action, "response", ""))
    
    if category == _safe_lower(expected_category):
        score += 0.5
        
    if len(response) > 20 and getattr(action, "resolve", False):
        score += 0.5
        
    return _clamp(score)

def grade_hard(action: Any, state_data: Any, expected_category: str) -> float:
    score = 0.0
    category = _safe_lower(getattr(action, "category", ""))
    response = _safe_lower(getattr(action, "response", ""))
    sentiment = _safe_lower(getattr(state_data, "sentiment", ""))
    
    if category == _safe_lower(expected_category):
        score += 0.3
        
    if getattr(action, "resolve", False) and len(response) > 30:
        score += 0.3
        
    # Strict rule for angry customers
    if sentiment == "angry" and any(word in response for word in ["sorry", "apologize", "understand", "frustrating"]):
        score += 0.4
    elif sentiment != "angry":
        score += 0.4
        
    return _clamp(score)