import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

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