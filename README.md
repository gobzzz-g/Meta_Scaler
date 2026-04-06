---
title: SupportDeskEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SupportDeskEnv - AI Customer Support Simulation Environment

## Overview
A real-world customer support environment to test LLM agents on issue classification, multi-step conversation, and decision making.

## Relevance
Robust evaluation of customer-facing AI agents is critical to verify empathetic alignment and accurate routing prior to production deployment.

## Interaction APIs
The environment serves a FastAPI interface:
- `POST /reset` - Resets environment, returns initial observation.
- `POST /step` - Takes an agent action, returns new observation, reward, done flag, and info.
- `GET /state` - Returns current state.

## Tasks
* **EASY**: Classification only.
* **MEDIUM**: Classification + single polite response.
* **HARD**: Multi-step conversational resolution emphasizing empathy and solution delivery.

## Setup & Run Local
```bash
pip install -r requirements.txt
python server.py
```

## Docker & Hugging Face Deployment
```bash
docker build -t supportdesk .
docker run -p 7860:7860 supportdesk
```
Works immediately on Hugging Face Spaces (Port 7860 exposed).

## Inference Script
Ensure variables `OPENAI_API_KEY` and `MODEL_NAME` are set.
```bash
python inference.py
```