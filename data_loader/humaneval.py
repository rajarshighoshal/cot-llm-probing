from datasets import load_dataset


class HumanEval:
    """HumanEval dataset (164 Python problems)."""

    def __init__(self):
        ds = load_dataset("openai_humaneval", split="test")
        self.data = list(ds)

    def get_prompts(self):
        return [item["prompt"] for item in self.data]

    def get_problems(self):
        """Return list of dicts with id, prompt, test, entry_point."""
        return [
            {
                "id": item["task_id"],
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }
            for item in self.data
        ]

    def __len__(self):
        return len(self.data)
