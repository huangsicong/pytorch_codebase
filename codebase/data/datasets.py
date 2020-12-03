from .registry import register
from torchvision import datasets
"""
Dataset registry for all the datasets in the code base 

Note that each function must return a pytorch dataset to be loaded into dataloaders
"""


@register("cifar10")
def get_dset(data_dir, transform, train):
    dset = datasets.CIFAR10(
        data_dir, train=train, download=True, transform=transform)
    return dset


@register("mnist")
def get_dset(data_dir, transform, train):
    dset = datasets.MNIST(
        data_dir, train=train, download=True, transform=transform)
    return dset


@register("fmnist")
def get_dset(data_dir, transform, train):
    dset = datasets.FashionMNIST(
        data_dir, train=train, download=True, transform=transform)
    return dset


@register("cifar100")
def get_dset(data_dir, transform, train):
    dset = datasets.CIFAR100(
        data_dir, train=train, download=True, transform=transform)
    return dset
