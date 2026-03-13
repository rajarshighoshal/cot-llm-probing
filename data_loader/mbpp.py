from datasets import load_dataset


class MBPP:
    """MBPP dataset (500 test problems)."""

    def __init__(self, split="test"):
        ds = load_dataset("google-research-datasets/mbpp", split=split)
        self.data = list(ds)

    def get_prompts(self):
        """Return prompts formatted for code completion models."""
        prompts = []
        for item in self.data:
            tests = "\n".join(f"# {t}" for t in item["test_list"])
            prompt = f"# Task: {item['text']}\n# Tests:\n{tests}\ndef"
            prompts.append(prompt)
        return prompts

    def get_problems(self):
        """Return list of dicts with id, prompt, test, solution."""
        problems = []
        for item in self.data:
            tests = "\n".join(f"# {t}" for t in item["test_list"])
            prompt = f"# Task: {item['text']}\n# Tests:\n{tests}\ndef"
            test_code = "\n".join(item["test_list"])
            problems.append({
                "id": item["task_id"],
                "prompt": prompt,
                "test": test_code,
                "entry_point": None,
                "solution": item["code"],
            })
        return problems

    def __len__(self):
        return len(self.data)
