import re

from datasets import load_dataset


def _extract_func_name(test_list):
    """Extract function name from the first assert statement."""
    for test in test_list:
        match = re.search(r'assert\s+(\w+)\s*\(', test)
        if match:
            return match.group(1)
    return "solution"


class MBPP:
    """MBPP dataset (500 test problems)."""

    def __init__(self, split="test"):
        ds = load_dataset("google-research-datasets/mbpp", split=split)
        self.data = list(ds)

    def _make_prompt(self, item):
        """Build prompt that includes the function signature."""
        func_name = _extract_func_name(item["test_list"])
        tests = "\n".join(f"# {t}" for t in item["test_list"])
        return f"# Task: {item['text']}\n# Tests:\n{tests}\ndef {func_name}("

    def get_prompts(self):
        """Return prompts formatted for code completion models."""
        return [self._make_prompt(item) for item in self.data]

    def get_problems(self):
        """Return list of dicts with id, prompt, test, solution."""
        problems = []
        for item in self.data:
            prompt = self._make_prompt(item)
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
