import os
import requests
import json
from openai import OpenAI

# -----------------------------
# FORCE LITELLM CLIENT (REQUIRED)
# -----------------------------
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

ENV_URL = "http://localhost:7860"
MODEL_NAME = "gpt-4o-mini"


# -----------------------------
# CLASSIFICATION (FALLBACK)
# -----------------------------
def classify_issue(issue: str):
    issue = issue.lower()

    if any(k in issue for k in ["payment", "card", "billing", "refund", "charged", "transaction"]):
        return "billing"

    elif any(k in issue for k in ["login", "error", "account", "bug", "crash", "password"]):
        return "tech"

    return "general"


# -----------------------------
# FALLBACK RESPONSE
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
        if "error" in issue_lower or "crash" in issue_lower:
            return "Please restart the app or clear cache."
        return "Please check your internet connection and restart the app."

    return "Thanks for reaching out. Please provide more details so I can assist you better."


# -----------------------------
# FOLLOW-UP
# -----------------------------
def followup_response():
    return "Please try the steps I shared and let me know if the issue continues."


# -----------------------------
# LLM ACTION (MANDATORY)
# -----------------------------
def get_llm_action(issue):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer support assistant. Respond ONLY in JSON format with keys: category, response, escalate, resolve."
                },
                {"role": "user", "content": issue}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)

            return {
                "category": data.get("category", "general"),
                "response": data.get("response", "Thanks for reaching out."),
                "escalate": data.get("escalate", False),
                "resolve": data.get("resolve", True)
            }

        except:
            return None

    except Exception as e:
        print(f"LLM Error: {e}")
        return None


# -----------------------------
# MAIN INFERENCE
# -----------------------------
def run_inference(level="easy"):
    print(f"[START] task=supportdesk_{level} env=SupportDeskEnv model={MODEL_NAME}")

    total_reward = 0.0
    steps_taken = 0
    done = False
    fixed_category = None

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"level": level}, timeout=5)
        obs = res.json().get("observation", {})

        while not done and steps_taken < 5:
            steps_taken += 1

            issue = obs.get("user_message", "Help needed")

            if fixed_category:
                category = fixed_category
            else:
                category = classify_issue(issue)
                fixed_category = category

            # 🔥 FORCE LLM CALL
            action_data = get_llm_action(issue)

            # ✅ fallback ONLY if LLM fails
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

            try:
                step_res = requests.post(
                    f"{ENV_URL}/step",
                    json=action_data,
                    timeout=5
                ).json()
            except:
                break

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            print(f"[STEP] step={steps_taken} action={json.dumps(action_data)} reward={reward} done={done}")

        score = total_reward / max(steps_taken, 1)
        success = score >= 0.6

        print(f"[END] success={success} steps={steps_taken} score={score}")

    except Exception:
        print(f"[END] success=False steps={steps_taken} score=0.0")


# -----------------------------
# ENTRY POINT (3 TASKS REQUIRED)
# -----------------------------
if __name__ == "__main__":
    run_inference("easy")
    run_inference("medium")
    run_inference("hard")