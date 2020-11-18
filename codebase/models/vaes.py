# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import torch
import torch.nn as nn
from ..utils.vae_utils import log_normal, log_mean_exp, log_normal_likelihood
from ..utils.computation_utils import singleton_repeat
from .registry import register, get_encoder, get_decoder
from torch.nn import functional as F
from torch.distributions.normal import Normal


@register("deep_blurry_vae")
def get_vae(hparams):
    """ 
    The deep VAE model used in the experiments. 
    This will is also used for blurry vae by setting hparams.blur_std 
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            self.enc = get_encoder(hparams)
            self.dec = get_decoder(hparams)
            self.observation_log_likelihood_fn = log_normal_likelihood
            self.hparams = hparams
            self.x_logvar = self.dec.x_logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):
            return self.dec(z)

        def forward(self, x, num_iwae=1):
            flattened_x = x.view(-1, hparams.dataset.input_vector_length)
            flattened_x_k = singleton_repeat(flattened_x, num_iwae)

            mu, logvar = self.enc(flattened_x_k)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)
            x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
            likelihood = self.observation_log_likelihood_fn(
                flattened_x_k, x_mean, x_logvar_full)
            elbo = likelihood + logpz - logqz

            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)


@register("dc_vae")
def get_vae(hparams):
    """ 
    Based upon the DCGAN version of VAE model used in the Likelihood Regret paper
    https://arxiv.org/abs/2003.02977
    Author's code is at https://github.com/XavierXiao/Likelihood-Regret
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            self.enc = get_encoder(hparams)
            self.dec = get_decoder(hparams)
            self.likelihoodfn = nn.CrossEntropyLoss(reduction='none')
            self.x_logvar = self.dec.x_logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):
            _, mean, logvar = self.dec(z)
            return mean, logvar

        def forward(self, x, num_iwae=1):
            B = x.size(0)
            mu, logvar = self.enc(x)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            reconstructed, mean, _ = self.dec(z)
            target = (x.view(-1) * 255).long()  #line from author's code
            recon = reconstructed.view(-1, 256)  #line from author's code
            neg_likelihood = torch.sum(self.likelihoodfn(recon, target)) / B
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            elbo = -neg_likelihood + logpz - logqz
            return mean, elbo, mu, logvar, neg_likelihood, logqz - logpz, z

    return VAE(hparams)


@register("vae_linear_fixed_var")
def get_vae(hparams):
    """ 
    The VAE model used for Analytical solution.
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            self.enc = get_encoder(hparams)
            self.dec = get_decoder(hparams)
            self.observation_log_likelihood_fn = log_normal_likelihood
            self.x_logvar = self.dec.x_logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):
            return self.dec(z)

        def forward(self, x, num_iwae=1):
            flattened_x = x.view(-1, hparams.dataset.input_vector_length)
            flattened_x_k = singleton_repeat(flattened_x, num_iwae)
            mu, logvar = self.enc(flattened_x_k)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)

            likelihood = self.observation_log_likelihood_fn(
                flattened_x_k, x_mean, x_logvar)
            elbo = likelihood + logpz - logqz

            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)
