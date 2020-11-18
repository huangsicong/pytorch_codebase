# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/utils.py

import numpy as np
from math import pi as pi
import torch
from ..models.registry import get_model
from .experiment_utils import note_taking
from torch import nn
from torch import optim
import math
import numbers
from torch.nn import functional as F
from torch.distributions.normal import Normal

def log_normal_likelihood(x, mean, logvar):
    """Implementation WITH constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py

    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """

    dim = list(mean.size())[1]
    logvar = torch.zeros(mean.size()) + logvar
    return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(1) +
                   torch.log(torch.tensor(2 * pi)) * dim)


def log_mean_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(torch.mean(torch.exp(x - max_), dim)) + torch.squeeze(max_)


def log_normal(x, mean, logvar):
    """
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    log normal WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out later
    Args:s
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """
    return -0.5 * (logvar.sum(1) + (
        (x - mean).pow(2) / torch.exp(logvar)).sum(1))
