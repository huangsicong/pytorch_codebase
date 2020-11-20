import torch
import os
import numpy as np
import torch.nn as nn
from ..models.registry import get_model
from ..utils.experiment_utils import note_taking, load_checkpoint, save_checkpoint, get_chechpoint_path
from ..data.load_data import load_training_data
from tqdm import tqdm
from ..utils.experiment_utils import get_cpu_type
import shutil


class Trainerbase:
    """
    This serves to be class to be inherited by other algorithms,
    it will work out of the box as long as run_batch is implemented
    Note that to the child can sample reconstructions in run_batch and give 
        function pointer to sample_images
    Important parameters:
        sample_images: a function that takes in parameters hparams, model, epoch, best
    """

    def __init__(self, model, optimizer, writer, hparams, sample_images=None):
        self.writer = writer
        self.hparams = hparams
        self.epoch, self.validation_loss_list = 0, []
        self.model = model
        self.optimizer = optimizer
        self.sample_images = sample_images

    def load(self, checkpoint_path, optimizer, reset_optimizer=False):
        self.epoch, self.validation_loss_list = load_checkpoint(
            path=checkpoint_path,
            optimizer=optimizer,
            reset_optimizer=reset_optimizer,
            model=self.model)

    def save(self, checkpoint_path, save_optimizer_state=True):
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=self.optimizer,
            save_optimizer_state=save_optimizer_state,
            model=self.model,
            epoch=self.epoch,
            validation_loss_list=self.validation_loss_list)

    def run(self):
        """
        Entry point for all the trainers, call this function to start the run
        it will load the checkpoint first if path is specified
        """
        hparams = self.hparams
        if hparams.checkpoint_path is not None:
            opt_load = self.optimizer
            if hparams.freeze_decoder or hparams.freeze_encoder:
                opt_load = None
            self.load(hparams.checkpoint_path, self.model, opt_load)
        self.train_and_val()
        return self.model

    def run_epoch(self, epoch, loader, istrain=True, term=None, tosample=False):
        """
        Runs one full pass on a dataloader
        """
        running_loss, count = 0.0, 0
        torch.set_default_tensor_type(get_cpu_type(self.hparams))
        for i, (data, _) in enumerate(loader):
            if term == count:
                break
            data = data.to(
                device=self.hparams.device, dtype=self.hparams.tensor_type)
            to_save_recon = (i == 1 and not istrain and tosample)
            loss_cpu = self.run_batch(
                data, istrain, to_save_recon=to_save_recon)
            running_loss += loss_cpu
            count += 1
            torch.set_default_tensor_type(self.hparams.dtype)
        torch.set_default_tensor_type(self.hparams.dtype)
        running_loss = running_loss / count if count else 0
        return running_loss

    def run_batch(self, data, istrain, to_save_recon=False):
        """
        For children to implement, runs training on a batch of samples
        """
        raise NotImplementedError

    def train_and_val(self):
        """
        Trains starting from scratch or a checkpoint and runs 
        validation in between
        """
        hparams = self.hparams
        num_epochs = hparams.model_train.epochs
        train_loader, validation_loader = load_training_data(hparams)
        start = self.epoch + 1
        for epoch in tqdm(range(start, num_epochs + 1)):
            train_loss = self.run_epoch(epoch, train_loader, istrain=True)
            note_taking(f'Training Epoch {epoch} Train loss is {train_loss}')
            tosample = (epoch % hparams.train_print_freq) == 0
            save_ckpt = (epoch % hparams.checkpointing_freq) == 0
            validation_loss = self.run_epoch(
                epoch,
                validation_loader,
                istrain=False,
                term=hparams.n_val_batch,
                tosample=tosample)
            note_taking(
                f'Training Epoch {epoch} Validation loss is {validation_loss}')
            best = False
            if self.validation_loss_list and validation_loss < min(
                    self.validation_loss_list):
                best = True
                note_taking(
                    f'new min val loss is {validation_loss} at epoch {epoch}')
            if tosample and self.sample_images:
                self.sample_images(hparams, self.model, epoch, best=best)
            self.validation_loss_list.append(validation_loss)
            self.epoch += 1
            if best or save_ckpt or (epoch == num_epochs):
                path_extra = 'best.pth' if best else f'checkpoint_epoch{epoch}.pth'
                checkpoint_path = os.path.join(hparams.messenger.checkpoint_dir,
                                               path_extra)
                self.save(checkpoint_path, save_optimizer_state=True)
        checkpoint_path = get_chechpoint_path(hparams)
        if checkpoint_path is None:
            hparams.chkt_step = -1
            checkpoint_path = get_chechpoint_path(hparams)
            note_taking(
                'Training done. Did not find the best checkpoint,  loading from the latest instead {}'
                .format(checkpoint_path))
        else:
            note_taking(
                "Training done. About to load checkpoint to eval from: {}".
                format(checkpoint_path))
        hparams.epoch = self.epoch
        self.load(checkpoint_path, self.optimizer, reset_optimizer=False)
        if self.sample_images:
            self.sample_images(hparams, self.model, self.epoch, best=True)
