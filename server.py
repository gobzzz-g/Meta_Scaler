from fastapi import FastAPI
from app.env import SupportDeskEnv
from app.models import Action

app = FastAPI(title="SupportDeskEnv")
env = SupportDeskEnv()

# -----------------------------
# ROOT CHECK
# -----------------------------
@app.get("/")
def root():
    return {"message": "SupportDeskEnv is running 🚀"}

# -----------------------------
# RESET (FIXED - BODY OPTIONAL)
# -----------------------------
@app.post("/reset")
async def reset(req: dict = {}):
    level = req.get("level", "medium")
    obs = await env.reset(level)
    return {"observation": obs.dict()}

# Optional GET (for browser testing)
@app.get("/reset")
async def reset_get():
    obs = await env.reset("medium")
    return {"observation": obs.dict()}

# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result

# -----------------------------
# STATE
# -----------------------------
@app.get("/state")
async def state():
    obs = await env.state()
    return {"observation": obs.dict()}

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)