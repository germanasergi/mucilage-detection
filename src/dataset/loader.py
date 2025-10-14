import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


def define_loaders(
    train_dataset,
    val_dataset,
    train=False,
    batch_size=32,
    num_workers=0,

):
    """
    Define data loaders for training and validation datasets.

    If `distributed` is True, the data loaders will use DistributedSampler for shuffling the
    training dataset and OrderedDistributedSampler for sampling the validation dataset.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The batch size for training data loader. Default to 32.
        num_workers (int): Number of workers to use for the dataloaders. Default to 0.
        train (bool): Define if train or not to return only one loader.
    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    if train:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, val_loader



    return train_loader


import segmentation_models_pytorch as smp
import torch.nn as nn

def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None,
):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)

        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
            activation=None,
        )

        # Add ReLU activation after the model
        if activation == "relu":
            model = nn.Sequential(
                model,
                nn.ReLU()
            )
        if activation == "sigmoid":
            model = nn.Sequential(
                model,
                nn.Sigmoid()
            )



        return model
    
    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")