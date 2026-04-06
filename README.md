---
title: SupportDeskEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 SupportDeskEnv — OpenEnv for Real-World Customer Support AI

SupportDeskEnv is a production-grade OpenEnv environment that simulates real-world customer support workflows, enabling evaluation of AI agents on decision-making, empathy, and multi-step reasoning.

---

## 🚀 Overview

Modern AI agents must handle more than just tasks — they must:
- Understand user intent
- Respond with empathy
- Take correct actions
- Resolve issues efficiently

SupportDeskEnv models this challenge by simulating realistic support tickets and evaluating agent behavior using structured rewards.

---

## 🌍 Real-World Use Case

This environment replicates real customer support scenarios such as:
- Login failures
- Payment and billing issues
- General account queries

It captures:
- Emotional context (e.g., frustrated users)
- Multi-step interactions
- Resolution workflows

---

## 🧩 OpenEnv Specification

Fully compliant with OpenEnv:

- ✅ Typed `Observation`, `Action`, `Reward` models (Pydantic)
- ✅ `reset()`, `step()`, `state()` APIs
- ✅ `openenv.yaml` included
- ✅ Validated via `openenv validate`

---

## ⚙️ Action Space

```json
## ⚙️ Action Space

```json
{
  "category": "billing | tech | general",
  "response": "string",
  "escalate": "boolean",
  "resolve": "boolean"
}

## ⚙️ Action Space

```json
{
  "category": "billing | tech | general",
  "response": "string",
  "escalate": "boolean",
  "resolve": "boolean"
}

