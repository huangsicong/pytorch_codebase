import os
import torch
from tqdm import tqdm
from ..utils.experiment_utils import note_taking, load_checkpoint, save_checkpoint, get_chechpoint_path
from ..utils.experiment_utils import get_cpu_type
from ..data.loaderbase import LoaderBase


class Trainerbase:
    """Abstract class for the various trainers in the codebase

    This serves to be class to be inherited by other algorithms,
    it will work out of the box as long as run_batch is implemented
    Note that to the child can sample reconstructions in run_batch and give 
        function pointer to sample_images

    Arguments:
        model: a PyTorch model that is callable
        optimizer: a PyTorch optimizer
        writer: writer class in the codebase
        hparams: an instance of the Hparam object
        sample_images: a function that takes in parameters hparams, model, epoch, best
    """

    def __init__(self, model, optimizer, writer, hparams, sample_images=None):
        self.writer = writer
        self.hparams = hparams
        self.epoch, self.validation_loss_list = 0, []
        self.model = model
        self.optimizer = optimizer
        self.sample_images = sample_images
        self.loader = LoaderBase(hparams)

    def load(self, checkpoint_path, optimizer, reset_optimizer=False):
        """Loads the model and optimizer while setting self.epoch 
            and self.validation_loss_list to the checkpoint

        Arguments:
            checkpoint_path (str): place to load the checkpoint
            optimizer: the PyTorch optimizer to mutate, can be None
        """
        self.epoch, self.validation_loss_list = load_checkpoint(
            path=checkpoint_path,
            optimizer=optimizer,
            reset_optimizer=reset_optimizer,
            model=self.model)

    def save(self, checkpoint_path, save_optimizer_state=True):
        """Save the current model, optimizer, epoch and 
            validation_loss_list

        Arguments:
            checkpoint_path (str): place to save the checkpoint
        """
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
            self.load(hparams.checkpoint_path, opt_load, self.model)
        self.train_and_val()
        return self.model

    def run_epoch(self, epoch, loader, istrain=True, term=None, tosample=False):
        """Runs one full pass on the given dataloader

        Arguments:
            epoch (int): which epoch it's currently going to run
            loader: a PyTorch dataloader 
            istrain (boolean): indicates if it's running training or not training(test/val)
            term (int or None): how many batches to run before terminating
            tosample (boolean): to sample the image or not
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
        """For children to implement, runs training on a batch of samples

        Arguments:
            data (Tensor): a PyTorch tensor
            istrain (boolean): indicates if it's running training or not training(test/val)
            to_save_recon (boolean): to save the reconstructed image or not
        """
        raise NotImplementedError

    def done_training(self):
        """A call back for when training is done to load the best or latest checkpoint
            and then sample_images from it if sample_images is given
        """
        hparams = self.hparams
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

    def train_and_val(self):
        """
        Trains starting from scratch or a checkpoint and runs 
            validation in between
        """
        hparams = self.hparams
        num_epochs = hparams.model_train.epochs
        train_loader, validation_loader = self.loader.get_train_and_val()
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
        self.done_training()
