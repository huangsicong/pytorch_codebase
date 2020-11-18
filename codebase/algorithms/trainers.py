from .trainerbase import Trainerbase
from ..models.registry import get_model
from ..utils.experiment_utils import sample_images, save_recon
import torch
from torch import optim


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        param.grad = None


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


"""
    The children classes for trainers must only take writer and hparam as parameters in 
    their init function
"""


class VAETrainer(Trainerbase):

    def __init__(self, writer, hparams):
        model = get_model(hparams).to(hparams.device)
        params_list = []
        if hparams.freeze_decoder:
            params_list.extend(model.enc.parameters())
            freeze_model(model.dec)
        elif hparams.freeze_encoder:
            params_list.extend(model.dec.parameters())
            freeze_model(model.enc)
        else:
            params_list.extend(model.parameters())
        optimizer = optim.Adam(
            params_list,
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay if hparams.weight_decay else 0)
        super().__init__(model, optimizer, writer, hparams, sample_images)

    def run_batch(self, data, istrain, to_save_recon=False):
        if istrain:
            self.optimizer.zero_grad()
            recon_batch, elbo, _, _, _, _, _ = self.model(data)
            loss = -elbo
            loss.backward()
            self.optimizer.step()
        else:
            train_bool = self.model.training
            self.model.eval()
            with torch.no_grad():
                recon_batch, elbo, _, _, _, _, _ = self.model(data)
                loss = -elbo
            self.model.train(train_bool)
        if to_save_recon:
            save_recon(data, recon_batch.detach(), self.hparams, self.model,
                       self.epoch + 1, False)
            # need to add self.epoch by 1 because the epoch count hasn't been incremented
            # while run batch plots, as its still going through that epoch
        return loss.item()
