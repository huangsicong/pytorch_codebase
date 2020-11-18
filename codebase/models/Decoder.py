import numpy as np
import torch
import torch.nn as nn
from ..utils.cnn_utils import get_layer, get_conv, weights_init
from .registry import register
from torch.nn import functional as F
from ..utils.guassian_blur import GaussianSmoothing
from torch.distributions.categorical import Categorical


@register("linear_decoder")
def get_decoder(hparams):
    """ 
    The VAE model used for Analytical solution.
    """

    class Decoder(nn.Module):

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            self.dec = nn.Linear(hparams.model_train.z_size,
                                 hparams.dataset.input_vector_length)
            self.x_logvar = nn.Parameter(
                torch.log(torch.tensor(1, dtype=hparams.tensor_type)),
                requires_grad=True)

        def forward(self, z):
            return self.dec(z), self.x_logvar

    return Decoder(hparams)


@register("deep_vae_decoder")
def get_decoder(hparams):
    """ 
    The deep VAE model used in the experiments. 
    """

    class Decoder(nn.Module):

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            self.dec = nn.Sequential(
                nn.Linear(hparams.model_train.z_size, 1024), nn.Tanh(),
                nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1024),
                nn.Tanh(), nn.Linear(1024, hparams.dataset.input_vector_length),
                nn.Sigmoid())
            self.x_logvar = nn.Parameter(
                torch.log(
                    torch.tensor(
                        hparams.model_train.x_var, dtype=hparams.tensor_type)),
                requires_grad=True)

        def forward(self, z):
            return self.dec(z), self.x_logvar

    return Decoder(hparams)


@register("blurry_decoder")
def get_decoder(hparams):
    """ 
    The deep blurry VAE model used in the experiments. 
    """

    class Decoder(nn.Module):

        def __init__(self, hparams):

            super(Decoder, self).__init__()
            self.d_dim1 = hparams.dataset.input_dims[1]
            self.d_dim2 = hparams.dataset.input_dims[2]
            self.input_len = hparams.dataset.input_vector_length
            self.dec = nn.Sequential(
                nn.Linear(hparams.model_train.z_size, 1024), nn.Tanh(),
                nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1024),
                nn.Tanh(), nn.Linear(1024, self.input_len), nn.Sigmoid())
            self.x_logvar = nn.Parameter(
                torch.log(
                    torch.tensor(
                        hparams.model_train.x_var, dtype=hparams.tensor_type)),
                requires_grad=False)

            self.Guassian_Kernel = None
            if hparams.blur_std is not None:
                self.Guassian_Kernel = GaussianSmoothing(
                    channels=1,
                    kernel_size=5,
                    sigma=hparams.blur_std,
                    tensor_type=hparams.tensor_type)
            self.hparams = hparams

        def forward(self, z):
            output_img = self.dec(z).view(-1, self.d_dim1, self.d_dim2)
            output_img = torch.unsqueeze(output_img, 1)
            output_img = F.pad(output_img, (2, 2, 2, 2), mode='reflect')
            if self.Guassian_Kernel:
                blurred_img = self.Guassian_Kernel(output_img)
            else:
                blurred_img = output_img
            return blurred_img.view(-1, self.input_len), self.x_logvar

    return Decoder(hparams)


@register("conv_decoder_mean")
def get_decoder(hparams):

    class Decoder(nn.Module):
        """
            DC VAE decoder used in the likelihood regret paper:
            https://arxiv.org/pdf/2003.02977.pdf
            Gives mean during forward pass and image sample for sample fn
        """

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            nf, nc = hparams.conv_params.nf, hparams.conv_params.nc
            input_channels = hparams.dataset.input_dims[0]
            self.nc = nc
            self.z_dim = hparams.model_train.z_size
            decoder_channels = [self.z_dim, 4 * nf, 2 * nf, nf, 256 * nc]
            num_convs, norm_relus = len(decoder_channels) - 1, [
                True, True, True, False
            ]
            decoder = get_layer(nn.ConvTranspose2d, decoder_channels,
                                [4] * num_convs, [1, 2, 2, 2], [0, 1, 1, 1],
                                norm_relus, norm_relus)
            self.dec = nn.Sequential(*decoder)
            if hparams.gauss_weight_init:
                self.apply(weights_init)
            self.x_logvar = None

        def get_logits(self, z):
            z = z.view(z.size(0), self.z_dim, 1, 1)
            output = self.dec(z)
            B, C, W, H = output.shape
            #reshape output to B x nc x 32 x 32 x 256
            output = output.view(B, self.nc, 256, W, H).permute(0, 1, 3, 4,
                                                                2).contiguous()
            return output

        def sample(self, z):
            with torch.no_grad():
                logits = self.get_logits(z)
                B, C, W, H, categories = logits.size()
                m = Categorical(logits=logits.view(-1, categories))
                sample = m.sample()
                sample = sample.reshape(B, C, W, H).float() / 255.
            return sample, None

        def forward(self, x):
            logits = self.get_logits(x)
            mean = torch.softmax(
                logits.detach(), dim=-1)  #getting it out of the comp graph
            mean = (mean * torch.arange(0, 256)).sum(-1)
            mean = mean / 255.
            return logits, mean, self.x_logvar

    return Decoder(hparams)


@register("conv_decoder")
def get_decoder(hparams):

    class Decoder(nn.Module):
        """
            DC VAE decoder used in the likelihood regret paper:
            https://arxiv.org/pdf/2003.02977.pdf
            Gives sample instead of mean
        """

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            nf, nc = hparams.conv_params.nf, hparams.conv_params.nc
            input_channels = hparams.dataset.input_dims[0]
            self.nc = nc
            self.z_dim = hparams.model_train.z_size
            decoder_channels = [self.z_dim, 4 * nf, 2 * nf, nf, 256 * nc]
            num_convs, norm_relus = len(decoder_channels) - 1, [
                True, True, True, False
            ]
            decoder = get_layer(nn.ConvTranspose2d, decoder_channels,
                                [4] * num_convs, [1, 2, 2, 2], [0, 1, 1, 1],
                                norm_relus, norm_relus)
            self.dec = nn.Sequential(*decoder)
            if hparams.gauss_weight_init:
                self.apply(weights_init)
            self.x_logvar = None

        def get_logits(self, z):
            z = z.view(z.size(0), self.z_dim, 1, 1)
            output = self.dec(z)
            B, C, W, H = output.shape
            #reshape output to B x nc x 32 x 32 x 256
            output = output.view(B, self.nc, 256, W, H).permute(0, 1, 3, 4,
                                                                2).contiguous()
            return output

        def get_image_sample(self, logits):
            with torch.no_grad():
                B, C, W, H, categories = logits.size()
                m = Categorical(logits=logits.view(-1, categories))
                sample = m.sample()
                sample = sample.reshape(B, C, W, H).float() / 255.
            return sample

        def forward(self, x):
            logits = self.get_logits(x)
            mean = self.get_image_sample(logits.detach())
            return logits, mean, self.x_logvar

    return Decoder(hparams)
