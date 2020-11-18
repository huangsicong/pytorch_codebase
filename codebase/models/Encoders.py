import torch.nn as nn
from ..utils.cnn_utils import get_layer, get_conv, weights_init
from .registry import register


@register("linear_encoder")
def get_encoder(hparams):
    """ 
    The VAE model used for Analytical solution.
    """

    class Encoder(nn.Module):

        def __init__(self, hparams):
            super(Encoder, self).__init__()
            self.z_dim = hparams.model_train.z_size
            self.enc = nn.Linear(hparams.dataset.input_vector_length,
                                 self.z_dim * 2)

        def forward(self, x):
            latent = self.enc(x)
            mean, logvar = latent[:, :self.z_dim], latent[:, self.z_dim:]
            return mean, logvar

    return Encoder(hparams)


@register("tanh_linear_vae_encoder")
def get_encoder(hparams):

    class Encoder(nn.Module):

        def __init__(self, hparams):
            super(Encoder, self).__init__()
            self.z_dim = hparams.model_train.z_size
            self.enc = nn.Sequential(
                nn.Linear(hparams.dataset.input_vector_length, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, 1024),
                nn.Tanh(),
                nn.Linear(1024, self.z_dim * 2),
            )

        def forward(self, x):
            latent = self.enc(x)
            mean, logvar = latent[:, :self.z_dim], latent[:, self.z_dim:]
            return mean, logvar

    return Encoder(hparams)


@register("conv_encoder")
def get_encoder(hparams):
    """
        DCVAE Encoder used in the likelihood regret paper:
        https://arxiv.org/pdf/2003.02977.pdf
    """

    class Encoder(nn.Module):

        def __init__(self, hparams):
            super(Encoder, self).__init__()
            nf, nc = hparams.conv_params.nf, hparams.conv_params.nc
            input_channels = hparams.dataset.input_dims[0]
            self.z_dim = hparams.model_train.z_size
            encoder_channels = [
                input_channels, nf, 2 * nf, 4 * nf, 2 * self.z_dim
            ]
            num_convs, norm_relus = len(encoder_channels) - 1, [
                True, True, True, False
            ]
            enc = get_layer(nn.Conv2d, encoder_channels, [4] * num_convs,
                            [2, 2, 2, 1], [1, 1, 1, 0], norm_relus, norm_relus)
            enc.append(nn.Flatten())
            self.enc = nn.Sequential(*enc)
            if hparams.gauss_weight_init:
                self.apply(weights_init)

        def forward(self, x):
            latent = self.enc(x)
            mean, logvar = latent[:, :self.z_dim], latent[:, self.z_dim:]
            return mean, logvar

    return Encoder(hparams)
