import json

from datasets import load_dataset


class LiveCodeBench:
    """LiveCodeBench dataset — function-based problems only (LeetCode platform).

    Uses the lite version for faster evaluation. Filters to LeetCode problems
    only since they are function-based (like HumanEval) and match our eval pipeline.
    AtCoder/Codeforces use stdin/stdout which requires a different eval approach.
    """

    def __init__(self, difficulty=None, version="release_v5"):
        # Use community clone (more reliable than official custom loader)
        ds = load_dataset("bzantium/livecodebench", version, split="test")

        # Filter to LeetCode problems only (function-based, like HumanEval)
        data = [item for item in ds if item["platform"] == "leetcode"]

        if difficulty:
            data = [item for item in data if item["difficulty"] == difficulty]

        self.data = data

    def _make_prompt(self, item):
        """Build a code completion prompt from starter code and problem description."""
        starter = item["starter_code"].strip()
        description = item["question_content"].strip()
        imports = "from typing import List, Optional, Tuple, Dict, Set\nfrom collections import defaultdict, deque, Counter\n\n"
        # Include problem description as a docstring-style comment
        desc_lines = "\n".join(f"# {line}" for line in description.split("\n") if line.strip())
        return f"{imports}{desc_lines}\n{starter}\n"

    def get_prompts(self):
        return [self._make_prompt(item) for item in self.data]

    def get_problems(self):
        """Return list of dicts compatible with our eval pipeline."""
        problems = []
        for item in self.data:
            prompt = self._make_prompt(item)

            # Parse test cases
            try:
                test_cases = json.loads(item["private_test_cases"])
            except (json.JSONDecodeError, TypeError):
                test_cases = json.loads(item["public_test_cases"])

            # Parse metadata for function name
            meta = json.loads(item.get("metadata", "{}") or "{}")
            func_name = meta.get("func_name", None)

            # Build test code from test cases
            test_lines = []
            for tc in test_cases:
                # Multi-arg inputs are newline-separated, join with commas
                inp = ", ".join(tc["input"].strip().split("\n"))
                expected = tc["output"].strip()
                if func_name:
                    test_lines.append(
                        f"assert str(Solution().{func_name}({inp})) == str({expected})"
                    )

            problems.append({
                "id": item["question_id"],
                "prompt": prompt,
                "test": "\n".join(test_lines),
                "entry_point": func_name,
                "difficulty": item["difficulty"],
                "platform": "leetcode",
                "title": item["question_title"],
                "question_content": item["question_content"],
            })
        return problems

    def __len__(self):
        return len(self.data)
