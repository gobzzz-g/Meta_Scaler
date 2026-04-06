from typing import Dict, Any
from .models import Observation, Action, Reward
from .utils import generate_ticket
from .tasks import get_task
from .reward import calculate_reward
from .graders import grade_easy, grade_medium, grade_hard

class SupportDeskEnv:
    def __init__(self):
        self.state_data = None
        self.expected_category = None
        self.task_config = None

    async def reset(self, level: str = "medium") -> Observation:
        self.task_config = get_task(level)
        ticket = generate_ticket(self.task_config.level)
        self.expected_category = ticket["category"]
        
        self.state_data = Observation(
            ticket_id=ticket["id"],
            user_message=ticket["message"],
            sentiment=ticket["sentiment"],
            history=[{"role": "user", "content": ticket["message"]}],
            step_count=0,
            task_level=self.task_config.level
        )
        return self.state_data

    async def step(self, action: Action) -> Dict[str, Any]:
        if not self.state_data:
            await self.reset()
            
        self.state_data.step_count += 1
        self.state_data.history.append({"role": "agent", "content": action.response or ""})
        
        # Pass max_steps to calculate_reward for the efficiency bonus
        reward = calculate_reward(self.state_data, action, self.expected_category, self.task_config.max_steps)
        
        done = action.resolve or action.escalate or self.state_data.step_count >= self.task_config.max_steps
        
        # Grading based on level
        if self.task_config.level == "easy":
            task_score = grade_easy(action, self.expected_category)
        elif self.task_config.level == "medium":
            task_score = grade_medium(action, self.expected_category)
        else:
            task_score = grade_hard(action, self.state_data, self.expected_category)
            
        reward.metrics["grader_score"] = task_score

        if not done:
            self.state_data.user_message = "Can you explain more?"
            self.state_data.history.append({"role": "user", "content": self.state_data.user_message})

        return {
            "observation": self.state_data.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": {"expected_category": self.expected_category, "task_score": task_score}
        }

    async def state(self) -> Observation:
        if not self.state_data:
            await self.reset()
        return self.state_data
