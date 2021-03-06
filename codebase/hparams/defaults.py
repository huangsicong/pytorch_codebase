# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import register
from .hparam import Hparam as hp


def MNIST():
    """ dataset """
    return hp(name="mnist", train=True, input_dims=[1, 28, 28])


def KMNIST():
    """ dataset """
    return hp(name="kmnist", train=True, input_dims=[1, 28, 28])


def SVHN():
    """ dataset """
    return hp(
        name="svhn", train=True, input_dims=[3, 32, 32], normalize=[0.5, 0.5])


def FashionMNIST():
    """ dataset """
    return hp(name="fmnist", train=True, input_dims=[1, 28, 28])


def CIFAR10():
    """ dataset """
    return hp(
        name="cifar10",
        train=True,
        input_dims=[3, 32, 32],
        normalize=[0.5, 0.5])


def CELEBA():
    """ dataset """
    return hp(
        name="celeba",
        val_name="celeba_valid",
        train=True,
        input_dims=[3, 218, 178],
        normalize=[0.5, 0.5])


def conv_params():
    """
        Conv params for DCVAE
    """
    return hp(nc=1, nf=32)


def defualt_decoder():
    """ model_train """
    return hp(
        z_size=10,
        batch_size=100,  #used for VAE training, testing
        epochs=1000,
        x_var=0.03)


def default_experiment():
    Hparam = hp(
        output_root_dir="./runoutputs",
        checkpoint_root_dir="./runoutputs",
        data_dir="./datasets",
        dataset=MNIST(),
        cuda=True,
        verbose=True,
        random_seed=6,
        checkpointing_freq=None,
        chkt_step=-2,  #-2 means take the best, -1 means take the latest
        specific_model_path=None,
        model_train=defualt_decoder(),
        save_step_sizes=False)
    return Hparam


def mnist_fid_default():
    Hparam = default_experiment()
    Hparam.cnn_model_name = "mnist_cnn"
    Hparam.representation = hp()
    Hparam.representation.test_batch_size = 1000
    Hparam.representation.batch_size = 64
    Hparam.representation.epochs = 10
    Hparam.representation.lr = 0.01
    Hparam.representation.momentum = 0.5
    Hparam.representation.log_interval = 10
    Hparam.rep_train_first = False
    Hparam.load_rep = "cnn_mnist"
    Hparam.rd.target_dist = "MNIST_fid"
    Hparam.chkt_step = -2  #-2 means take the best, -1 means take the latest
    Hparam.original_experiment = True
    Hparam.rd.max_beta = 0.33
    Hparam.rd.anneal_steps = 100000
    return Hparam
