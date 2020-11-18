#!/usr/bin/env python3

from .registry import register
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
from ..utils.experiment_utils import note_taking


def init_new_FashionMNIST(batch_size,
                          hparams,
                          train=False,
                          shuffle=False,
                          data="FashionMNIST",
                          cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            hparams.data_dir,
            train=train,
            download=True,
            transform=transforms.Compose((transforms.Resize(
                (hparams.dataset.input_dims[1], hparams.dataset.input_dims[2])),
                                          transforms.ToTensor()))),
        batch_size=batch_size,
        shuffle=(True if shuffle else False),
        **kwargs)
    return loader


@register("FashionMNIST_train_valid")
def FashionMNIST_train_valid(batch_size, hparams):
    kwargs = {'num_workers': 4, 'pin_memory': True} if hparams.cuda else {}
    train_dataset = datasets.FashionMNIST(
        hparams.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose((transforms.Resize(
            (hparams.dataset.input_dims[1], hparams.dataset.input_dims[2])),
                                      transforms.ToTensor())))
    valid_dataset = datasets.FashionMNIST(
        hparams.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose((transforms.Resize(
            (hparams.dataset.input_dims[1], hparams.dataset.input_dims[2])),
                                      transforms.ToTensor())))
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = 10000
    np.random.seed(hparams.random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)
    return train_loader, valid_loader


@register("FashionMNIST_eval_train")
def FashionMNIST_eval_train(batch_size, hparams):
    return init_new_FashionMNIST(
        batch_size,
        hparams,
        train=True,
        shuffle=False,
        data="FashionMNIST",
        cuda=hparams.cuda)


@register("FashionMNIST_eval_test")
@register("FashionMNIST_test")
def FashionMNIST_test(batch_size, hparams):
    """ 
    This loader is for evaluating RD. It supports loading only a particular digit class, 
    by spesifying hparams.dataset.label.
     """
    if hparams.dataset.label is None:
        return init_new_FashionMNIST(
            batch_size,
            hparams,
            train=False,
            shuffle=False,
            data="FashionMNIST",
            cuda=hparams.cuda)
    else:
        full_test_loader = init_new_FashionMNIST(
            batch_size,
            hparams,
            train=False,
            shuffle=False,
            data="FashionMNIST",
            cuda=hparams.cuda)

        mask = torch.ones(batch_size) * hparams.dataset.label
        mask = mask.to(device=hparams.device, dtype=hparams.tensor_type)
        for i, (batch, labels) in enumerate(full_test_loader, 0):
            labels = labels.to(hparams.tensor_type).to(hparams.device)
            mask = torch.eq(labels, mask)
            target_data = batch[mask]
            save_image(
                target_data.view(-1, hparams.dataset.input_dims[0],
                                 hparams.dataset.input_dims[1],
                                 hparams.dataset.input_dims[2]),
                hparams.messenger.image_dir + "original_label{}.png".format(
                    hparams.dataset.label),
            )
            break
        target_data = target_data.to(
            device=hparams.device, dtype=hparams.tensor_type)
        if hparams.messenger.running_rd:
            hparams.rd.batch_size = target_data.size()[0]
            hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains
        note_taking(
            "Loaded category data for FashionMNIST {} digit class of size {}".
            format(hparams.dataset.label, target_data.size()))
        return [(target_data, hparams.dataset.label)]


@register("FashionMNIST_test_single_class")
def FashionMNIST_test(batch_size, hparams):
    """ 
    This loader is for evaluating RD. It supports loading only a particular digit class, 
    by specifying hparams.dataset.label.
     """
    if hparams.dataset.label is None:
        return init_new_FashionMNIST(
            batch_size,
            hparams,
            train=False,
            shuffle=True,
            data="FashionMNIST",
            cuda=hparams.cuda)
    else:
        full_test_loader = init_new_FashionMNIST(
            batch_size,
            hparams,
            train=False,
            shuffle=True,
            data="FashionMNIST",
            cuda=hparams.cuda)

        mask = torch.ones(batch_size) * hparams.dataset.label
        mask = mask.to(device=hparams.device, dtype=hparams.tensor_type)
        for i, (batch, labels) in enumerate(full_test_loader, 0):
            labels = labels.to(hparams.tensor_type).to(hparams.device)
            mask = torch.eq(labels, mask)
            target_data = batch[mask]
            save_image(
                target_data.view(-1, hparams.dataset.input_dims[0],
                                 hparams.dataset.input_dims[1],
                                 hparams.dataset.input_dims[2]),
                hparams.messenger.image_dir + "original_label{}.png".format(
                    hparams.dataset.label),
            )
            break
        target_data = target_data.to(
            device=hparams.device, dtype=hparams.tensor_type)
        if hparams.messenger.running_rd:
            hparams.rd.batch_size = target_data.size()[0]
            hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains
        note_taking(
            "Loaded category data for FashionMNIST {} digit class of size {}".
            format(hparams.dataset.label, target_data.size()))
        return [(target_data, hparams.dataset.label)]
