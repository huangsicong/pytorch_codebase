import torch
from torchvision import transforms
from torch.utils.data import random_split
from .registry import get_dset
from ..utils.experiment_utils import get_cpu_type


class LoaderBase:
    """Base calss for all objects that manage dataloaders

    Arguments:
        hparams: an instance of the Hparam object
        extra_transforms(iterable): 
            iterable transforms to be applied after resize, totensor and normalize,
            can be None
    """

    def __init__(self, hparams, extra_transforms=None):
        no_workers = hparams.num_workers if hparams.num_workers else 4
        pin_memory = hparams.pin_memory if hparams.pin_memory else False
        resize, normalize = hparams.dataset.input_dims, hparams.dataset.normalize
        transform_list = []
        if resize:
            transform_list.append(transforms.Resize(resize[1:]))
        transform_list.append(transforms.ToTensor())
        if normalize:
            mean_norm, std_norm = normalize
            transform_list.append(
                transforms.Normalize((mean_norm, mean_norm, mean_norm),
                                     (std_norm, std_norm, std_norm)))
        if extra_transforms:
            transform_list.extend(extra_transforms)
        self.transform = transforms.Compose(transform_list)
        self.loader_args = {'num_workers': no_workers, 'pin_memory': pin_memory}
        self.hparams = hparams

    def _get_loader(self, dset, shuffle, overwrite_batch_size=None):
        """Private function to get a dataloader from a dataset object using
        the appropriate parameters in hparams

        Arguments:
            dset: a pytorch dataset object
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        batch_size = self.hparams.model_train.batch_size
        if overwrite_batch_size:
            batch_size = overwrite_batch_size
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=shuffle, **self.loader_args)
        return loader

    def get_loader(self,
                   name,
                   istrain=True,
                   shuffle=False,
                   overwrite_batch_size=None):
        """
        Get dataloader with the specified dataset name using hparam settings

        Arguments:
            name (string): the name of the dataset
            istrain (boolean): indicates if we want to get the training dataloader or test
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        dset = get_dset(name, self.hparams.data_dir, self.transform, istrain)
        return self._get_loader(
            dset, shuffle=shuffle, overwrite_batch_size=overwrite_batch_size)

    def get_train_and_val(self, percent_val=0.1, shuffle=False):
        """Gets the train and validation dataloader using the hparams parameters

        Arguments:
            percent_val (float): percent of the training loader we want to use as validation, 
                range between 0 and 1
            shuffle (boolean): to shuffle the dataset when loading
        """
        train_dset = get_dset(self.hparams.dataset.name, self.hparams.data_dir,
                              self.transform, True)
        val_length = int(len(train_dset) * percent_val)
        train_length = len(train_dset) - val_length
        torch.set_default_tensor_type(get_cpu_type(self.hparams))
        train_dset, val_dset = random_split(
            train_dset, (train_length, val_length),
            generator=torch.Generator(device="cpu").manual_seed(
                self.hparams.random_seed))
        torch.set_default_tensor_type(self.hparams.dtype)
        return self._get_loader(
            train_dset, shuffle=shuffle), self._get_loader(
                val_dset, shuffle=shuffle)
