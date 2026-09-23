"""Small reproducibility helpers shared by training and tests."""

import torch


def make_data_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
