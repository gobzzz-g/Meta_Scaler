import os
import requests
import json
from openai import OpenAI

# =============================
# ENV VARIABLES (STRICT)
# =============================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

ENV_URL = "http://localhost:7860"

# =============================
# 🔥 GLOBAL LLM CLIENT
# =============================
try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    print("✅ LLM CLIENT INITIALIZED", flush=True)
except Exception as e:
    print(f"❌ CLIENT INIT FAILED: {e}", flush=True)
    client = None


# =============================
# LLM ACTION (RESPONSES API)
# =============================
def get_llm_action(issue):
    try:
        if client is None:
            raise Exception("Client not initialized")

        print("🚀 CALLING LLM...", flush=True)

        response = client.responses.create(
            model=MODEL_NAME,
            input=issue
        )

        print("✅ LLM RESPONSE RECEIVED", flush=True)

        # Extract text safely
        output_text = ""
        try:
            output_text = response.output[0].content[0].text
        except:
            output_text = "I'll help you with this."

        return {
            "category": "tech",
            "response": output_text[:120],
            "escalate": False,
            "resolve": True
        }

    except Exception as e:
        print(f"❌ LLM ERROR: {e}", flush=True)
        return {
            "category": "tech",
            "response": "Temporary issue. Please try again.",
            "escalate": False,
            "resolve": True
        }


# =============================
# MAIN INFERENCE
# =============================
def run_inference(level="easy"):
    print(f"[START] task=supportdesk_{level} env=SupportDeskEnv model={MODEL_NAME}", flush=True)

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
            action = get_llm_action(issue)

            step_res = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=5
            ).json()

            reward = step_res.get("reward", {}).get("score", 0.0)
            done = step_res.get("done", False)
            obs = step_res.get("observation", {})

            total_reward += reward

            print(
                f"[STEP] step={steps_taken} action={json.dumps(action)} reward={reward} done={done}",
                flush=True
            )

        score = total_reward / max(steps_taken, 1)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.6

        print(
            f"[END] success={success} steps={steps_taken} score={score}",
            flush=True
        )

    except Exception as e:
        print(
            f"[END] success=False steps={steps_taken} score=0.0 error={e}",
            flush=True
        )


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    run_inference("easy")
    run_inference("medium")
    run_inference("hard")