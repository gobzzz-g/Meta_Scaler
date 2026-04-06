class TaskConfig:
    def __init__(self, level: str, max_steps: int):
        self.level = level
        self.max_steps = max_steps

EASY_TASK = TaskConfig(level="easy", max_steps=1)
MEDIUM_TASK = TaskConfig(level="medium", max_steps=1)
HARD_TASK = TaskConfig(level="hard", max_steps=5)

def get_task(level: str) -> TaskConfig:
    tasks = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK}
    return tasks.get(level.lower(), EASY_TASK)
