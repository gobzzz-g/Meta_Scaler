from fastapi import FastAPI
from app.env import SupportDeskEnv
from app.models import Action
from pydantic import BaseModel

app = FastAPI(title="SupportDeskEnv")
env = SupportDeskEnv()

class ResetRequest(BaseModel):
    level: str = "medium"

@app.post("/reset")
async def reset(req: ResetRequest):
    obs = await env.reset(req.level)
    return {"observation": obs.dict()}

@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result

@app.get("/state")
async def state():
    obs = await env.state()
    return {"observation": obs.dict()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
