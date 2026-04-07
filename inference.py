import os
import requests
import json
from openai import OpenAI

ENV_URL = "http://localhost:7860"
MODEL_NAME = "gpt-4o-mini"

# -----------------------------
# 🔥 FORCE CLIENT INIT (CRITICAL)
# -----------------------------
client = None

try:
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL"),
        api_key=os.environ.get("API_KEY", "dummy-key-if-missing")
    )
    print("✅ OpenAI client initialized")

except Exception as e:
    print(f"Client Init Error: {e}")
    client = None


# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify_issue(issue: str):
    issue = issue.lower()

    if any(k in issue for k in [
        "payment", "card", "billing", "refund",
        "charged", "transaction", "money"
    ]):
        return "billing"

    if any(k in issue for k in [
        "login", "error", "account", "bug",
        "crash", "password", "not working",
        "issue", "problem"
    ]):
        return "tech"

    return "tech"


# -----------------------------
# FALLBACK RESPONSE
# -----------------------------
def generate_response(issue: str, category: str):
    issue_lower = issue.lower()

    if category == "billing":
        return "I understand your concern. Please check your billing details or retry the transaction."

    elif category == "tech":
        return "I’m sorry for the inconvenience. Please restart the app or check your internet connection."

    return "Thanks for reaching out."


# -----------------------------
# FOLLOW-UP
# -----------------------------
def followup_response():
    return "Please try the steps I shared and let me know if the issue continues."


# -----------------------------
# 🔥 LLM ACTION (MANDATORY)
# -----------------------------
def get_llm_action(issue):
    global client
    if client is None:
        try:
            client = OpenAI(
                base_url=os.environ.get("API_BASE_URL"),
                api_key=os.environ.get("API_KEY", "dummy-key-if-missing")
            )
        except Exception as e:
            print(f"Fallback Client Init Error: {e}")
            return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Respond ONLY in JSON format: {\"category\":\"billing|tech|general\",\"response\":\"...\",\"escalate\":false,\"resolve\":true}"
                },
                {"role": "user", "content": issue}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)

            return {
                "category": data.get("category", "tech"),
                "response": data.get("response", "Thanks for reaching out."),
                "escalate": data.get("escalate", False),
                "resolve": data.get("resolve", True)
            }

        except Exception as parse_e:
            print(f"Parse Error: {parse_e}")
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

            # 🔥 ALWAYS TRY LLM FIRST
            action_data = get_llm_action(issue)

            # fallback if LLM fails
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
            except Exception as e:
                print(f"Step Error: {e}")
                break

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            print(f"[STEP] step={steps_taken} action={json.dumps(action_data)} reward={reward} done={done}")

        score = total_reward / max(steps_taken, 1)
        success = score >= 0.6

        print(f"[END] success={success} steps={steps_taken} score={score}")

    except Exception as e:
        print(f"[END] success=False steps={steps_taken} score=0.0")


# -----------------------------
# ENTRY POINT (3 TASKS REQUIRED)
# -----------------------------
if __name__ == "__main__":
    run_inference("easy")
    run_inference("medium")
    run_inference("hard")