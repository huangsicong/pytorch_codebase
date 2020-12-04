from .registry import register
from torchvision import datasets
"""
Dataset registry for all the datasets in the code base 

Note that each function must return a pytorch dataset to be loaded into dataloaders
"""


def get_str(train):
    if train:
        split_str = "train"
    else:
        split_str = "test"
    return split_str


@register("cifar10")
def get_dset(hparams, transform, train):
    dset = datasets.CIFAR10(
        hparams.data_dir, train=train, download=True, transform=transform)
    return dset


@register("mnist")
def get_dset(hparams, transform, train):
    dset = datasets.MNIST(
        hparams.data_dir, train=train, download=True, transform=transform)
    return dset


@register("fmnist")
def get_dset(hparams, transform, train):
    dset = datasets.FashionMNIST(
        hparams.data_dir, train=train, download=True, transform=transform)
    return dset


@register("cifar100")
def get_dset(hparams, transform, train):
    dset = datasets.CIFAR100(
        hparams.data_dir, train=train, download=True, transform=transform)
    return dset


@register("svhn")
def get_dset(hparams, transform, train):
    dset = datasets.SVHN(
        hparams.data_dir,
        split=get_str(train),
        download=True,
        transform=transform)
    return dset


@register("kmnist")
def get_dset(hparams, transform, train):
    dset = datasets.KMNIST(
        hparams.data_dir, train=train, download=True, transform=transform)
    return dset


@register("celeba")
def get_dset(hparams, transform, train):
    dset = datasets.CelebA(
        hparams.data_dir,
        split=get_str(train),
        download=True,
        transform=transform)
    return dset


@register("celeba_valid")
def get_dset(hparams, transform, train):
    dset = datasets.CelebA(
        hparams.data_dir, split='valid', download=True, transform=transform)
    return dset
