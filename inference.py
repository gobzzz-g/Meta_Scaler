import os
import requests
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(api_key=OPENAI_API_KEY)

def fallback_response():
    return "I’m sorry for the inconvenience. I will help resolve your issue."

def run_inference(level="medium"):
    print("[START]")
    try:
        res = requests.post(f"{API_BASE_URL}/reset", json={"level": level})
        obs = res.json()["observation"]
        
        done = False
        while not done:
            print(f"[STEP]")
            
            prompt = f"Categorize this issue. Respond with JSON: {{'category': 'billing|tech|general', 'response': '...', 'resolve': true}}. Issue: {obs['user_message']}"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                action_data = json.loads(response.choices[0].message.content)
            except Exception:
                # Deterministic fallback
                action_data = {
                    "category": "general",
                    "response": fallback_response(),
                    "escalate": True,
                    "resolve": True
                }
            
            print(f"[STEP]")
            step_res = requests.post(f"{API_BASE_URL}/step", json=action_data).json()
            obs = step_res["observation"]
            done = step_res["done"]
            
        print("[END]")
        
    except Exception:
        print("[END]")

if __name__ == "__main__":
    run_inference("hard")
