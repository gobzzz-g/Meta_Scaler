import os
import requests
import json
import random
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = "http://localhost:7860"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=os.getenv("OPENAI_API_KEY")
)

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
            return "I understand your concern. Please check your refund status in the billing section. If it is still pending, I can guide you further."
        if "charged" in issue_lower:
            return "I’m sorry for the inconvenience. Please verify your transaction history in billing. If the charge is incorrect, I can help you raise a refund request."
        return "I understand your concern. Please update your payment method in the billing section and retry the transaction."

    elif category == "tech":
        if "login" in issue_lower:
            return "I’m sorry for the inconvenience. Please reset your password using the forgot password option and try logging in again."
        if "error" in issue_lower:
            return "I’m sorry for the inconvenience. Please restart the app or clear cache. This usually resolves such errors."
        return "I’m sorry for the inconvenience. Please restart the app or check your internet connection."

    return "Thanks for reaching out. Please provide more details so I can assist you better."

# -----------------------------
# Follow-up variation
# -----------------------------
def followup_response():
    options = [
        "Please try the steps I shared and let me know if the issue continues.",
        "Let me know if that resolves your issue or if you need further help.",
        "Feel free to reach out again if the problem persists."
    ]
    return options[0]  # deterministic for evaluation

# -----------------------------
# MAIN INFERENCE
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
        # Reset
        res = requests.post(f"{ENV_URL}/reset", json={"level": level})
        obs = res.json().get("observation", {})

        while not done and steps_taken < 5:
            steps_taken += 1

            issue = obs.get("user_message", "Help needed")

            # Keep category consistent
            if fixed_category:
                category = fixed_category
            else:
                category = classify_issue(issue)
                fixed_category = category

            action_data = None

            # Optional LLM usage
            if os.getenv("OPENAI_API_KEY"):
                try:
                    prompt = f"""Categorize this issue and respond in JSON:
{{"category":"billing|tech|general","response":"...","escalate":true/false,"resolve":true/false}}
Issue: {issue}"""

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )

                    try:
                        action_data = json.loads(response.choices[0].message.content)
                    except:
                        action_data = None
                except:
                    action_data = None

            # Deterministic fallback (primary)
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

            # Step
            step_res = requests.post(f"{ENV_URL}/step", json=action_data).json()

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            action_str = json.dumps(action_data, separators=(',', ':'))
            print(f"[STEP] step={steps_taken} action={action_str} reward={reward} done={done}")

        # Score
        score = total_reward / max(steps_taken, 1)
        score = max(0.0, min(1.0, score))

        success = score >= 0.6

        print(f"[END] success={success} steps={steps_taken} score={score}")

    except:
        print(f"[END] success=False steps={steps_taken} score=0.0")


if __name__ == "__main__":
    run_inference("hard")