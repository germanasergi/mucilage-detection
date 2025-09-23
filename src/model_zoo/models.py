import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Union
from torchgeo.models import get_weight
from torchgeo.models.api import WeightsEnum
from torch.hub import load_state_dict_from_url
from loguru import logger

def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None
):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)


        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            # encoder_weights=encoder_weights,
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
    



class CNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=8, log_features=False):
        super(CNN, self).__init__()
        self.log_features = log_features

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        if self.log_features:
            print("Block 1:", x.shape, x.mean().item(), x.std().item())

        # Block 2
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        if self.log_features:
            print("Block 2:", x.shape, x.mean().item(), x.std().item())

        # Block 3
        x = self.conv3(x)
        x = self.gn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        if self.log_features:
            print("Block 3:", x.shape, x.mean().item(), x.std().item())

        # Classifier
        x = self.classifier(x)
        if self.log_features:
            print("Classifier output:", x.shape)

        return x
    

class MILResNet(nn.Module):
    def __init__(self, model_name="resnet18", in_channels=7, num_classes=2, pretrained=True):
        super().__init__()
        # --- Backbone from timm ---
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # no FC classifier
            in_chans=in_channels,
            global_pool=""  # keep spatial feature maps
        )
        feat_dim = self.backbone.num_features  # e.g. 512 for resnet18

        # --- Attention module ---
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # --- Bag-level classifier ---
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Get spatial features: [B, C, H, W]
        features = self.backbone.forward_features(x)  
        B, C, H, W = features.shape

        # Flatten into instances: [B, N, C]
        instances = features.view(B, C, H * W).permute(0, 2, 1)

        # Attention weights
        attn_logits = self.attention(instances)  # [B, N, 1]
        attn = torch.softmax(attn_logits, dim=1)

        # Weighted sum = bag representation
        bag_repr = torch.sum(attn * instances, dim=1)  # [B, C]

        # Bag-level prediction
        logits = self.classifier(bag_repr)  # [B, num_classes]

        return logits, attn.view(B, H, W)
    

def build_timm_model(model_name: str, in_channels: int, num_classes: int, pretrained=True):
    """
    Build a timm model with flexible input channels.

    Args:
        model_name: Name of the timm model (e.g., 'resnet18', 'efficientnet_b0')
        in_channels: Number of input channels
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights

    Returns:
        model: torch.nn.Module
    """
    logger.info(f"Building timm model {model_name} | in_channels={in_channels}, num_classes={num_classes}")
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels) #, img_size=256

    # If in_channels != 3, timm automatically adjusts the first conv
    # but you could also manually modify it if needed:
    # if in_channels != 3:
    #     logger.info("Adjusting first conv layer for custom input channels")
    #     conv_layer = model.conv_stem if hasattr(model, 'conv_stem') else model.conv1
    #     model.conv_stem = nn.Conv2d(
    #         in_channels, conv_layer.out_channels,
    #         kernel_size=conv_layer.kernel_size,
    #         stride=conv_layer.stride,
    #         padding=conv_layer.padding,
    #         bias=False
    #     )

    return model

