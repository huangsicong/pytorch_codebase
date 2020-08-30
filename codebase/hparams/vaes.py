# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import register
from .defaults import *
from .hparam import Hparam as hp


# ----------------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------------
def vae_experiment():
    Hparam = default_experiment()
    Hparam.model_name = "deep_vae"
    Hparam.chkt_step = -2  #-2 means take the best, -1 means take the latest
    Hparam.train_print_freq = 100
    Hparam.start_checkpointing = 0
    Hparam.checkpointing_freq = 100
    Hparam.learning_rate = 1e-4
    Hparam.resume_training = True
    Hparam.n_test_batch = 10
    Hparam.original_experiment = True
    Hparam.model_name = "deep_vae"
    Hparam.group_list = ["icml", "VAEs", "RD"]
    return Hparam


# ----------------------------------------------------------------------------
# rd
# ----------------------------------------------------------------------------
@register("vae2_template_test")
def vae2_rd():
    Hparam = vae_experiment()
    Hparam.train_first = True
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "vae2_template_test"
    return Hparam
