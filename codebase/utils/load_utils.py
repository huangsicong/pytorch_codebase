import torch
import numpy as np
from torch.utils.data import random_split
from .experiment_utils import get_cpu_type
"""
Utils for loading data
"""


def split_data(hparams, dset, split_percent):
    """Splits a PyTorch dataset according to the split_percentage 

    Arguments:
        hparams: a Hparam object
        dset: PyTorch dataset to split on 
        split_percent: The percentage of dset that goes into the split

    Returns:
        subdset of size len(dset) * (1-split_percent) and subdset of size len(dset)*(split_percent)
    """
    torch.set_default_tensor_type(get_cpu_type(hparams))
    len_dset = len(dset)
    valt_length = int(len_dset * split_percent)
    train_length = len_dset - valt_length
    train_dset, valt_dset = random_split(
        dset, (train_length, valt_length),
        generator=torch.Generator(device="cpu").manual_seed(
            hparams.random_seed))
    torch.set_default_tensor_type(hparams.dtype)
    return train_dset, valt_dset
