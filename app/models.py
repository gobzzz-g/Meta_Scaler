from pydantic import BaseModel, Field
from typing import List, Optional

class Observation(BaseModel):
    ticket_id: str = Field(description="Unique ID for the support ticket")
    user_message: str = Field(description="The latest message from the customer")
    sentiment: str = Field(description="Customer sentiment (angry, frustrated, neutral, polite)")
    history: List[dict] = Field(default_factory=list, description="Conversation history")
    step_count: int = Field(default=0, description="Current step in the conversation")
    task_level: str = Field(description="Task difficulty level (easy, medium, hard)")

class Action(BaseModel):
    category: Optional[str] = Field(default=None, description="Classified issue category (billing, tech, general)")
    response: Optional[str] = Field(default="", description="Agent's response to the customer")
    escalate: bool = Field(default=False, description="Whether to escalate the ticket to a human")
    resolve: bool = Field(default=False, description="Whether the issue is considered resolved")

class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Overall reward score")
    metrics: dict = Field(default_factory=dict, description="Detailed metrics breakdown")
