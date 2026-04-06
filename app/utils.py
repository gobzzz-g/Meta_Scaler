import random
import uuid

ISSUE_TEMPLATES = {
    "billing": ["I was overcharged on my last invoice.", "How do I update my credit card?", "Cancel my subscription."],
    "tech": ["The app keeps crashing on startup.", "I can't log in to my account.", "API is returning 500 errors."],
    "general": ["What are your business hours?", "Where can I find the documentation?", "Do you offer enterprise plans?"]
}
SENTIMENTS = ["angry", "frustrated", "neutral", "polite"]

def generate_ticket(level: str):
    category = random.choice(list(ISSUE_TEMPLATES.keys()))
    message = random.choice(ISSUE_TEMPLATES[category])
    sentiment = random.choice(SENTIMENTS)
    
    return {
        "id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
        "category": category,
        "message": message,
        "sentiment": sentiment,
        "level": level
    }
