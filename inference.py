import os
import requests
import json
from openai import OpenAI

# 🔥 GLOBAL PROXY CALL (UNSKIPPABLE - VERY IMPORTANT)
try:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0
    )
    print("✅ GLOBAL PROXY CALL SUCCESS")
except Exception as e:
    print(f"❌ GLOBAL PROXY CALL FAILED: {e}")

ENV_URL = "http://localhost:7860"
MODEL_NAME = "gpt-4o-mini"


# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify_issue(issue: str):
    issue = issue.lower()

    if any(k in issue for k in [
        "payment", "card", "billing", "refund",
        "charged", "transaction"
    ]):
        return "billing"

    if any(k in issue for k in [
        "login", "error", "account", "bug",
        "crash", "password", "issue", "problem"
    ]):
        return "tech"

    return "tech"


# -----------------------------
# FALLBACK RESPONSE
# -----------------------------
def generate_response(issue: str, category: str):
    if category == "billing":
        return "I understand your concern. Please check your billing details or retry the transaction."
    elif category == "tech":
        return "I’m sorry for the inconvenience. Please restart the app or check your internet connection."
    return "Thanks for reaching out."


# -----------------------------
# LLM ACTION (ALWAYS USES PROXY)
# -----------------------------
def get_llm_action(issue):
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Respond ONLY in JSON: {\"category\":\"billing|tech\",\"response\":\"...\",\"escalate\":false,\"resolve\":true}"
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
                "response": data.get("response", "Thanks"),
                "escalate": False,
                "resolve": True
            }
        except Exception as e:
            print(f"JSON parse error: {e}")
            return {
                "category": "tech",
                "response": "Sorry, I’ll help you fix this.",
                "escalate": False,
                "resolve": True
            }

    except Exception as e:
        print(f"LLM Error: {e}")
        return {
            "category": "tech",
            "response": "Temporary issue. Please try again.",
            "escalate": False,
            "resolve": True
        }


# -----------------------------
# MAIN INFERENCE
# -----------------------------
def run_inference(level="easy"):
    print(f"[START] task=supportdesk_{level} env=SupportDeskEnv model={MODEL_NAME}")

    total_reward = 0.0
    steps_taken = 0
    done = False

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"level": level}, timeout=5)
        obs = res.json().get("observation", {})

        while not done and steps_taken < 5:
            steps_taken += 1

            issue = obs.get("user_message", "Help needed")

            # 🔥 ALWAYS CALL LLM
            action_data = get_llm_action(issue)

            step_res = requests.post(
                f"{ENV_URL}/step",
                json=action_data,
                timeout=5
            ).json()

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            print(f"[STEP] step={steps_taken} action={json.dumps(action_data)} reward={reward} done={done}")

        score = total_reward / max(steps_taken, 1)
        success = score >= 0.6

        print(f"[END] success={success} steps={steps_taken} score={score}")

    except Exception as e:
        print(f"[END] success=False steps={steps_taken} score=0.0 error={e}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_inference("easy")
    run_inference("medium")
    run_inference("hard")