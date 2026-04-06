from typing import Any

def grade_action(action: Any, observation: Any) -> float:
    score = 0.0

    category = (getattr(action, "category", "") or "").lower()
    response = (getattr(action, "response", "") or "").lower()
    issue = (getattr(observation, "user_message", "") or "").lower()

    # ---------------------------
    # 1. Category match (0.5)
    # ---------------------------
    if any(k in issue for k in ["payment", "card", "billing", "refund"]):
        if category == "billing":
            score += 0.5
    elif any(k in issue for k in ["login", "error", "account", "bug"]):
        if category == "tech":
            score += 0.5
    else:
        if category == "general":
            score += 0.5

    # ---------------------------
    # 2. Response quality (0.3)
    # ---------------------------
    if len(response) > 25:
        score += 0.3

    # ---------------------------
    # 3. Actionability (0.2)
    # ---------------------------
    if any(k in response for k in ["check", "update", "reset", "try", "restart", "please"]):
        score += 0.2

    # ---------------------------
    # Prevent zero-lock
    # ---------------------------
    if score == 0.0 and len(response) > 20:
        score = 0.2

    return float(max(0.0, min(1.0, score)))