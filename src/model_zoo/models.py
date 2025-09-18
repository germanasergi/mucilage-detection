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
    


def define_model_timm(
        model_name: str,
        num_classes: int,
        input_channels: int = 3,
        freeze_backbone: bool = False,
        weights: Union[str, WeightsEnum, None, bool] = None,
        bands: dict = None,
        selected_channels = list,
        **kwargs) -> nn.Module:
    """
    Create a model with PyTorchGeo weights support.
    
    Args:
        model_name: Name of the model architecture (e.g., 'resnet50', 'efficientnet_b0')
        weights: Weight loading strategy:
            - True: Load ImageNet pretrained weights via TIMM
            - WeightsEnum: Load specific PyTorchGeo weights
            - str: Either a path to weights file or PyTorchGeo weight name
            - None/False: No pretrained weights
        num_classes: Number of output classes
        input_channels: Number of input channels (default: 3 for RGB)
        freeze_backbone: Whether to freeze backbone parameters
        bands: Optional mapping of channel indices to satellite band names
                        e.g., {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", ...}
                        Common satellite sensors:
                        - Sentinel-2: {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", 5: "SWIR2", ...}
                        - Landsat-8: {0: "Coastal", 1: "Blue", 2: "Green", 3: "Red", 4: "NIR", 5: "SWIR1", 6: "SWIR2", ...}
        **kwargs: Additional arguments passed to timm.create_model
    
    Returns:
        torch.nn.Module: Configured model with loaded weights
    """
    use_timm_pretrained_weights_imagenet = weights is True
    
    logger.info(f'Creating Model: {model_name} with weights: {weights}')

    # Create model with TIMM
    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        in_chans=input_channels,
        pretrained=use_timm_pretrained_weights_imagenet,
        **kwargs
    )
    
    logger.info("Loading weights")
    # Load PyTorchGeo weights
    if weights and weights is not True:
        try:
            # Handle different weight types
            if isinstance(weights, WeightsEnum):
                logger.info(f"Loading PyTorchGeo weights: {weights}")
                state_dict = weights.get_state_dict(progress=True)

            elif isinstance(weights, str):
                if weights.endswith('.pth') or weights.endswith('.pt'):
                    # Load from file path
                    logger.info(f"Loading weights from file: {weights}")
                    state_dict = torch.load(weights, map_location='cpu')
                    # Handle different state dict formats
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                else:
                    # Load by PyTorchGeo weight name
                    logger.info(f"Loading PyTorchGeo weights by name: {weights}")
                    weight_enum = get_weight(weights)
                    state_dict = weight_enum.get_state_dict(progress=True)
            else:
                raise ValueError(f"Unsupported weight type: {type(weights)}")
            
            # Use flexible loading with satellite band information
            load_state_dict_with_flexibility(model, state_dict, strict=False, bands=bands,selected_channels=selected_channels)
            logger.success("✓ Weights loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            print("Continuing with model initialization...")
    
    # Handle backbone freezing
    if freeze_backbone:
        logger.info("Freezing backbone parameters...")
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier/head parameters
        try:
            # Try different common classifier attribute names
            classifier_attrs = ['classifier', 'head', 'fc']
            classifier = None
            
            for attr in classifier_attrs:
                if hasattr(model, attr):
                    classifier = getattr(model, attr)
                    break
            
            # Also try get_classifier method
            if classifier is None and hasattr(model, 'get_classifier'):
                try:
                    classifier = model.get_classifier()
                except:
                    pass
            
            if classifier is not None:
                for param in classifier.parameters():
                    param.requires_grad = True
                logger.info("✓ Classifier head unfrozen")
            else:
                logger.info("⚠ Warning: Could not find classifier to unfreeze")
                # Print available attributes for debugging
                logger.info(f"Available model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
        except Exception as e:
            logger.error(f"⚠ Warning: Could not unfreeze classifier - {e}")
    
    logger.info("Add softmax")
    ### Add the softmax 
    model = nn.Sequential(
                model,
                nn.Softmax(dim=1)
            )
    logger.success("Model is ready")
    return model


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
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels)

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