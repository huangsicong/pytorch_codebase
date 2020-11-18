# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang
""" 
This is the root script to train generative models
"""

#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch import optim
from .hparams.registry import get_hparams
from .data.load_data import *
from .utils.experiment_utils import *
from .algorithms.trainers import VAETrainer
import argparse
import random
import sys
import subprocess
from .utils.aae_utils import aae_bridge
from .utils.gan_utils import gan_bridge
sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--hparam_set", default="default", type=str)
parser.add_argument("--overwrite", default=False, type=bool)
parser.add_argument("--e_name")

args = parser.parse_args()
args_dict = vars(args)
hparams = get_hparams(args.hparam_set)
hparams.overwrite = args.overwrite
set_random_seed(hparams.random_seed)


def main(writer, hparams):
    if hparams.cnn_model_name is not None:
        if "mnist" in hparams.cnn_model_name:
            cnn_model = get_mnist_representation(hparams.representation,
                                                 hparams)
    # Make sure to include vae in your VAE models' names, same for gan and aae.
    if hparams.original_experiment:
        if "vae" in hparams.model_name:
            note_taking("About to run experiment on {} with z size={}".format(
                hparams.model_name, hparams.model_train.z_size))
            vae_trainer = VAETrainer(writer, hparams)
            model = vae_trainer.run()
        elif "gan" in hparams.model_name:
            model = gan_bridge(hparams)
            note_taking(
                "About to run GAN experiment on {} with z size={}".format(
                    hparams.model_name, hparams.model_train.z_size))
        elif "aae" in hparams.model_name:
            model = aae_bridge(hparams)
            note_taking(
                "About to run AAE experiment on {} with z size={}".format(
                    hparams.model_name, hparams.model_train.z_size))
    else:
        model = load_user_model(hparams)
        note_taking("Loaded user model {} with z size={}".format(
            hparams.model_name, hparams.model_train.z_size))

    if hparams.overwrite_variance:
        model.x_logvar = nn.Parameter(
            torch.log(
                torch.tensor(
                    hparams.overwrite_variance, dtype=hparams.tensor_type)),
            requires_grad=False)
        note_taking("NOTICE! model.x_logvar is overwrited: {}".format(
            model.x_logvar))
    sample_images(hparams, model, hparams.epoch, best=False)
    x_var = None
    if model.x_logvar:
        x_var = torch.exp(model.x_logvar).detach().cpu().numpy().item()
    note_taking(
        "Loaded the generative model: {} with decoder variance {}".format(
            hparams.model_name, x_var))
    model.eval()

    # template to time your code
    for i in range(2):
        # whatever loop
        end_time = datetime.now()
        hour_summery = compute_duration(hparams.messenger.start_time, end_time)
        note_taking("Iteration {} finished. Took {} hours".format(
            i, hour_summery))
    logging(
        args_dict,
        hparams.to_dict(),
        hparams.messenger.results_dir,
        hparams.messenger.dir_path,
        stage="final")


if __name__ == '__main__':

    writer = initialze_run(hparams, args)
    start_time = datetime.now()
    hparams.messenger.start_time = start_time
    dir_path = os.path.dirname(os.path.realpath(__file__))
    git_label = subprocess.check_output(
        ["cd " + dir_path + " && git describe --always && cd .."],
        shell=True).strip()
    if hparams.verbose:
        note_taking("The git label is {}".format(git_label))
    main(writer, hparams)
    end_time = datetime.now()
    hour_summery = compute_duration(start_time, end_time)
    writer_json_path = hparams.messenger.tboard_path + "/tboard_summery.json"
    writer.export_scalars_to_json(writer_json_path)
    writer.close()
    note_taking(
        "Experiment finished, results written at: {}. Took {} hours".format(
            hparams.messenger.results_dir, hour_summery))
