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
    Hparam.start_checkpointing = 0
    Hparam.checkpointing_freq = 100
    Hparam.learning_rate = 1e-4
    Hparam.resume_training = True
    Hparam.n_val_batch = 10
    Hparam.original_experiment = True
    Hparam.model_name = "deep_vae"
    Hparam.train_print_freq = 50
    return Hparam


def dcvae_experiment():
    Hparam = vae_experiment()
    Hparam.conv_params = conv_params()
    Hparam.learning_rate = 3e-4
    Hparam.weight_decay = 3e-5
    Hparam.model_train.batch_size = 128
    Hparam.model_train.epochs = 200
    Hparam.model_train.z_size = 100
    Hparam.model_name = "dc_vae"
    Hparam.gauss_weight_init = True
    Hparam.encoder_name = "conv_encoder"
    Hparam.decoder_name = "conv_decoder"
    Hparam.group_list = ["DCVAE", "randomseed_1"]
    return Hparam


@register("dcvae100_mnist")
def dcvae():
    Hparam = dcvae_experiment()
    Hparam.dataset.normalize = None
    Hparam.dataset.input_dims = (1, 32, 32)
    Hparam.dataset.keep_scale = False
    return Hparam


@register("dcvae100_fmnist")
def dcvae():
    Hparam = dcvae_experiment()
    Hparam.dataset = FashionMNIST()
    Hparam.dataset.input_dims = (1, 32, 32)
    Hparam.dataset.normalize = None
    Hparam.dataset.keep_scale = False
    return Hparam


@register("dcvae100_cifar10")
def dcvae():
    Hparam = dcvae_experiment()
    Hparam.dataset = cifar10()  #this also sets dim to (3,32,32)
    Hparam.dataset.input_dims = (3, 32, 32)
    Hparam.dataset.normalize = None
    Hparam.conv_params.nf = 64
    Hparam.conv_params.nc = 3
    Hparam.dataset.keep_scale = False
    return Hparam
