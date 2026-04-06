from .models import Action, Observation

def grade_easy(action: Action, expected_category: str) -> float:
    cat = (action.category or "").lower()
    if cat and expected_category.lower() in cat:
        return 1.0
    return 0.0

def grade_medium(action: Action, expected_category: str) -> float:
    score = 0.0
    cat = (action.category or "").lower()
    if cat and expected_category.lower() in cat:
        score += 0.4
    
    resp = (action.response or "").lower()
    if resp:
        if any(w in resp for w in ["sorry", "apologize", "please", "thank"]):
            score += 0.3
        if len(action.response or "") > 10:
            score += 0.3
    return float(min(1.0, max(0.0, score)))

def grade_hard(action: Action, obs: Observation, expected_category: str) -> float:
    score = 0.0
    resp = (action.response or "").lower()
    
    if any(w in resp for w in ["understand", "sorry", "frustrat"]):
        score += 0.25
    if "?" in resp or any(w in resp for w in ["provide", "could you"]):
        score += 0.25
    if action.resolve:
        score += 0.5
    return float(min(1.0, max(0.0, score)))
