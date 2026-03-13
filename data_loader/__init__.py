from data_loader.humaneval import HumanEval
from data_loader.mbpp import MBPP


def get_dataset(name, **kwargs):
    """Factory for datasets."""
    datasets = {
        "humaneval": HumanEval,
        "mbpp": MBPP,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")
    return datasets[name](**kwargs)
