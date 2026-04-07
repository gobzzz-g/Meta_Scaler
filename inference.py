import os
import requests
import json
import random

# Safe import (avoid crash if package issue)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = "http://localhost:7860"


# -----------------------------
# SAFE CLIENT INITIALIZATION ✅
# -----------------------------
def get_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or OpenAI is None:
            print("WARNING: OpenAI client not available")
            return None

        return OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key
        )

    except Exception as e:
        print(f"Client Init Error: {e}")
        return None


client = get_client()


# -----------------------------
# Smart classification
# -----------------------------
def classify_issue(issue: str):
    issue = issue.lower()
    if any(k in issue for k in ["payment", "card", "billing", "refund", "charged"]):
        return "billing"
    elif any(k in issue for k in ["login", "error", "account", "bug", "crash"]):
        return "tech"
    return "general"


# -----------------------------
# Context-aware response
# -----------------------------
def generate_response(issue: str, category: str):
    issue_lower = issue.lower()

    if category == "billing":
        if "refund" in issue_lower:
            return "I understand your concern. Please check your refund status in the billing section."
        if "charged" in issue_lower:
            return "I’m sorry for the inconvenience. Please verify your transaction history."
        return "Please update your payment method and retry."

    elif category == "tech":
        if "login" in issue_lower:
            return "Please reset your password using the forgot password option."
        if "error" in issue_lower:
            return "Please restart the app or clear cache."
        return "Please check your internet connection and restart the app."

    return "Please provide more details so I can assist you better."


# -----------------------------
# Follow-up variation
# -----------------------------
def followup_response():
    return "Please try the steps I shared and let me know if the issue continues."


# -----------------------------
# SAFE LLM CALL ✅
# -----------------------------
def get_llm_action(issue):
    if client is None:
        return None

    try:
        prompt = f"""Categorize this issue and respond in JSON:
{{"category":"billing|tech|general","response":"...","escalate":true/false,"resolve":true/false}}
Issue: {issue}"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except:
            return None

    except Exception as e:
        print(f"LLM Error: {e}")
        return None


# -----------------------------
# MAIN INFERENCE (FULL SAFE) 🚀
# -----------------------------
def run_inference(level="medium"):
    TASK_NAME = f"supportdesk_{level}"
    ENV_NAME = "SupportDeskEnv"

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")

    total_reward = 0.0
    steps_taken = 0
    done = False
    fixed_category = None

    try:
        # RESET SAFE
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"level": level}, timeout=5)
            obs = res.json().get("observation", {})
        except Exception as e:
            print(f"Reset Error: {e}")
            print(f"[END] success=False steps=0 score=0.0")
            return

        while not done and steps_taken < 5:
            steps_taken += 1

            issue = obs.get("user_message", "Help needed")

            # Keep category consistent
            if fixed_category:
                category = fixed_category
            else:
                category = classify_issue(issue)
                fixed_category = category

            # Try LLM
            action_data = get_llm_action(issue)

            # Fallback (PRIMARY SAFE PATH)
            if not action_data:
                if steps_taken == 1:
                    response_text = generate_response(issue, category)
                else:
                    response_text = followup_response()

                action_data = {
                    "category": category,
                    "response": response_text,
                    "escalate": False,
                    "resolve": True
                }

            # STEP SAFE
            try:
                step_res = requests.post(
                    f"{ENV_URL}/step",
                    json=action_data,
                    timeout=5
                ).json()
            except Exception as e:
                print(f"Step Error: {e}")
                break

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            action_str = json.dumps(action_data, separators=(',', ':'))
            print(f"[STEP] step={steps_taken} action={action_str} reward={reward} done={done}")

        # FINAL SCORE
        score = total_reward / max(steps_taken, 1)
        score = max(0.0, min(1.0, score))

        success = score >= 0.6

        print(f"[END] success={success} steps={steps_taken} score={score}")

    except Exception as e:
        print(f"Fatal Error: {e}")
        print(f"[END] success=False steps={steps_taken} score=0.0")


# -----------------------------
# ENTRY POINT (SAFE) ✅
# -----------------------------
if __name__ == "__main__":
    try:
        run_inference("hard")
    except Exception as e:
        print(f"[END] success=False steps=0 score=0.0")