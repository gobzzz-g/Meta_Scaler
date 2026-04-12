import requests
import json
import os
from openai import OpenAI

ENV_URL = "http://localhost:7860"

# =============================
# LLM CLIENT SETUP (LiteLLM Proxy)
# =============================
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

USE_LLM = bool(API_BASE_URL and API_KEY)

if USE_LLM:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"✅ LLM proxy configured: {API_BASE_URL}", flush=True)
else:
    client = None
    print("⚠️ Running in LOCAL FALLBACK MODE — API_BASE_URL or API_KEY not found.", flush=True)


# =============================
# LLM-BASED RESPONSE
# =============================
def llm_response(msg, name, sentiment="neutral", history=None):
    """Use the LLM via the provided proxy to generate a support response."""

    # Build conversation messages for the LLM
    system_prompt = f"""You are an expert AI customer support agent. You must respond to the customer and classify their issue.

RULES:
1. Classify the issue into exactly one category: "billing", "tech", or "general".
2. Always address the customer by name: {name}.
3. If the customer sentiment is angry or frustrated, start with a sincere apology and acknowledge their frustration before providing help.
4. If the customer sentiment is polite, thank them for reaching out.
5. Provide specific, actionable step-by-step solutions (at least 3 steps).
6. Do NOT give generic filler responses. Be concrete and helpful.
7. Always aim to resolve the issue in your response.
8. Do NOT repeat previous responses if there is conversation history.

CATEGORY GUIDE:
- "billing": payment issues, charges, invoices, subscriptions, refunds, credit cards, pricing, costs
- "tech": login problems, passwords, crashes, errors, bugs, API issues, technical problems
- "general": business hours, documentation, enterprise plans, general questions

Customer sentiment: {sentiment}

You MUST respond with valid JSON only, no markdown, no code fences. Use this exact format:
{{"category": "billing|tech|general", "response": "your detailed response here", "escalate": false, "resolve": true}}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    if history:
        for h in history:
            role = "assistant" if h.get("role") == "agent" else "user"
            messages.append({"role": role, "content": h.get("content", "")})
    else:
        messages.append({"role": "user", "content": msg})

    # If the last message in history is from the user, no need to add msg again
    # But if history is empty or last is agent, add the user message
    if history and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": msg})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )

        raw = completion.choices[0].message.content.strip()

        # Clean up potential markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)

        # Validate and ensure required fields
        category = result.get("category", "general").lower()
        if category not in ("billing", "tech", "general"):
            category = "general"

        return {
            "category": category,
            "response": result.get("response", ""),
            "escalate": bool(result.get("escalate", False)),
            "resolve": bool(result.get("resolve", True)),
        }

    except Exception as e:
        print(f"⚠️ LLM call failed: {e}, falling back to deterministic.", flush=True)
        return smart_response(msg, name, sentiment=sentiment, history=history)


# =============================
# DETERMINISTIC FALLBACK
# =============================
def smart_response(msg, name, sentiment="neutral", history=None):
    """Deterministic, sentiment-aware response engine (fallback)."""
    msg_lower = msg.lower().strip()
    sentiment_lower = sentiment.lower().strip()
    step_index = len([m for m in (history or []) if m.get("role") == "agent"])

    # --- Empathy prefix ---
    if sentiment_lower in ("angry", "frustrated"):
        empathy_prefix = f"I understand your frustration, {name}, and I sincerely apologize for the inconvenience. "
    elif sentiment_lower == "polite":
        empathy_prefix = f"Thank you for reaching out, {name}. "
    else:
        empathy_prefix = f"Hi {name}, I understand your concern. "

    category = _classify_category(msg_lower)
    response_body = _build_response(category, msg_lower, step_index)
    full_response = empathy_prefix + response_body

    return {
        "category": category,
        "response": full_response,
        "escalate": False,
        "resolve": True,
    }


def _classify_category(msg):
    """Strict keyword-based intent classification."""
    if any(kw in msg for kw in ["cancel subscription", "cancel my subscription"]):
        return "billing"
    if any(kw in msg for kw in ["overcharg", "invoice", "billing"]):
        return "billing"
    if any(kw in msg for kw in ["credit card", "payment", "card"]):
        return "billing"
    if any(kw in msg for kw in ["log in", "login", "password", "can't log"]):
        return "tech"
    if any(kw in msg for kw in ["crash", "crashing", "error", "500"]):
        return "tech"
    if any(kw in msg for kw in ["api", "not working", "bug"]):
        return "tech"
    if any(kw in msg for kw in ["hours", "business hour"]):
        return "general"
    if any(kw in msg for kw in ["documentation", "docs"]):
        return "general"
    if any(kw in msg for kw in ["enterprise", "plan"]):
        return "general"
    if any(kw in msg for kw in ["issue", "problem", "trouble", "broken"]):
        return "tech"
    if any(kw in msg for kw in ["charge", "refund", "money", "pay", "price", "cost", "bill"]):
        return "billing"
    return "general"


def _build_response(category, msg, step_index):
    if category == "tech":
        return _tech_response(msg, step_index)
    elif category == "billing":
        return _billing_response(msg, step_index)
    else:
        return _general_response(msg, step_index)


def _tech_response(msg, step_index):
    responses = [
        "Here is how to fix your issue: Step 1 — Go to the login page and click 'Forgot Password'. Step 2 — Enter your registered email. Step 3 — Check your inbox for the reset link and update your password. This should resolve the issue immediately.",
        "To further help resolve this: Step 1 — Clear your browser cache and cookies. Step 2 — Try logging in using an incognito window. Step 3 — If the issue persists, verify your email address is correct in your profile settings.",
        "Here is an additional step: Step 1 — Check if your account is locked by visiting Account Security. Step 2 — If locked, use the unlock link sent to your email. Step 3 — Update your password to a new secure one.",
        "One more thing to try: Step 1 — Make sure you are using the correct login URL. Step 2 — Disable any VPN or proxy that might block access. Step 3 — Try a different browser or device.",
        "Final step to resolve this: Step 1 — Contact support with your registered email for a manual account unlock. Step 2 — Reset your password via the emailed link. Step 3 — Log in on a fresh browser session. Issue resolved.",
    ]
    return responses[step_index % len(responses)]


def _billing_response(msg, step_index):
    responses = [
        "Here is how to resolve your billing concern: Step 1 — Go to Account Settings and open the Billing section. Step 2 — Review the flagged charge and click 'Dispute'. Step 3 — Our billing team will review and issue a correction within 24 hours. Issue resolved.",
        "To further help with billing: Step 1 — Download your recent invoices for reference. Step 2 — Check your plan renewal date. Step 3 — Update any outdated payment information. The update should apply immediately.",
        "Here is an additional billing step: Step 1 — Verify no duplicate subscriptions exist. Step 2 — Review any pending refund requests. Step 3 — Our billing team will follow up within 24 hours.",
        "One more approach for billing: Step 1 — Check your email for any billing notifications. Step 2 — Review the FAQ section for common billing questions. Step 3 — Submit a detailed support ticket if needed. Issue resolved.",
        "Final billing resolution: Step 1 — Contact our billing team with your account ID. Step 2 — Provide the specific charge in question. Step 3 — A resolution will be provided within one business day. The fix is complete.",
    ]
    return responses[step_index % len(responses)]


def _general_response(msg, step_index):
    responses = [
        "Here is the information you need: Step 1 — Visit your account dashboard for a quick overview of all options. Step 2 — Check the Help Center at help.company.com for guides. Step 3 — Use the search feature to find your specific answer. Issue resolved.",
        "To further help with your question: Step 1 — Navigate to the relevant section in your Account Settings. Step 2 — Review the FAQ page for common answers. Step 3 — Submit a detailed request via Contact Us for personalized help.",
        "Here is an additional step: Step 1 — Browse our knowledge base for step-by-step tutorials. Step 2 — Join our community forum for peer support. Step 3 — Schedule a live support session for complex questions. Issue resolved.",
        "One more approach: Step 1 — Check your email for any recent updates from our team. Step 2 — Review your account notifications. Step 3 — Call our support line during business hours for immediate help.",
        "Final step to help you: Step 1 — Use our chatbot for instant answers. Step 2 — Download our mobile app for on-the-go account management. Step 3 — Our team is ready to help with any additional details. Issue resolved.",
    ]
    return responses[step_index % len(responses)]


# =============================
# ACTION DISPATCHER
# =============================
def get_llm_action(obs):
    user_message = obs.get("user_message", "")
    name = obs.get("customer_name", "Customer")
    sentiment = obs.get("sentiment", "neutral")
    history = obs.get("history", [])

    if USE_LLM:
        print(f"🤖 LLM CALL | sentiment={sentiment} | msg={user_message[:60]}", flush=True)
        return llm_response(user_message, name, sentiment=sentiment, history=history)
    else:
        print(f"⚠️ FALLBACK | sentiment={sentiment} | msg={user_message[:60]}", flush=True)
        return smart_response(user_message, name, sentiment=sentiment, history=history)


# =============================
# MAIN
# =============================
def run_inference(level="easy"):
    print(f"[START] task=supportdesk_{level} env=SupportDeskEnv model=gpt-4o-mini use_llm={USE_LLM}", flush=True)

    total_reward = 0.0
    steps_taken = 0

    res = requests.post(f"{ENV_URL}/reset", json={"level": level})
    obs = res.json().get("observation", {})

    while steps_taken < 5:
        steps_taken += 1

        action = get_llm_action(obs)

        step_res = requests.post(f"{ENV_URL}/step", json=action).json()

        reward = step_res.get("reward", {}).get("score", 0.5)

        total_reward += reward

        done = step_res.get("done", False)
        obs = step_res.get("observation", {})

        print(
            f"[STEP] step={steps_taken} action={json.dumps(action)[:120]}... reward={reward} done={done}",
            flush=True,
        )

        if done:
            break

    score = total_reward / max(steps_taken, 1)

    print(f"[END] success={score >= 0.6} steps={steps_taken} score={score:.4f}", flush=True)


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    run_inference("easy")
    run_inference("medium")
    run_inference("hard")