from data_loader.humaneval import HumanEval
from data_loader.mbpp import MBPP
from data_loader.livecodebench import LiveCodeBench


def get_dataset(name, **kwargs):
    """Factory for datasets."""
    datasets = {
        "humaneval": HumanEval,
        "mbpp": MBPP,
        "livecodebench": LiveCodeBench,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")
    return datasets[name](**kwargs)
