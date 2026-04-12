from .models import Action, Observation, Reward

def calculate_reward(obs: Observation, action: Action, expected_category: str, max_steps: int) -> Reward:
    score = 0.0
    metrics = {}
    
    # Correct category extraction
    cat = (action.category or "").lower()
    if cat and expected_category.lower() in cat:
        score += 0.3
        metrics["category_correct"] = 0.3
    
    # Helpful & Empathetic check (Deterministic heuristic)
    resp = (action.response or "").lower()
    if resp:
        if any(word in resp for word in ["sorry", "apologize", "understand", "help"]):
            empathy_score = 0.2 if obs.sentiment in ["angry", "frustrated"] else 0.1
            score += empathy_score
            metrics["empathy"] = empathy_score
            
        # Angry customer strict rule
        if obs.sentiment == "angry" and not any(w in resp for w in ["sorry", "apologize", "understand"]):
            score -= 0.25
            metrics["angry_penalty"] = -0.25
            
        # Anti-generic response penalty
        generic_phrases = ["i will help you", "let me help", "i understand your issue"]
        if any(phrase in resp for phrase in generic_phrases) and len(action.response or "") < 60:
            score -= 0.1
            metrics["generic_penalty"] = -0.1
            
        if any(word in resp for word in ["step", "fix", "update", "here is", "resolved"]):
            score += 0.3
            metrics["helpfulness"] = 0.3

        # Repetition penalty
        past_responses = [msg["content"].lower() for msg in obs.history if msg["role"] == "agent"]
        if resp in past_responses:
            score -= 0.2
            metrics["repetition_penalty"] = -0.2
            
    # Penalties
    if action.escalate:
        score -= 0.1
        metrics["escalation_penalty"] = -0.1
        
    if action.resolve and not action.escalate:
        score += 0.2
        metrics["resolution_bonus"] = 0.2
        
        # Efficiency bonus
        if obs.step_count < max_steps:
            efficiency_bonus = 0.1 * (max_steps - obs.step_count)
            score += efficiency_bonus
            metrics["efficiency_bonus"] = efficiency_bonus
            
    final_score = float(max(0.01, min(0.99, score)))
    return Reward(score=final_score, metrics=metrics)
