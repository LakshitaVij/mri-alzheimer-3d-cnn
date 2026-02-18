import torch # main PyTorch library
import torch.nn as nn # neural network layers (Conv, Linear, etc.)
import torch.nn.functional as F # functional ops (relu, pooling, etc.)

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()  # initializes nn.Module internals

        # --- Feature extractor (3D conv stack) ---
        self.conv1 = nn.Conv3d(1,  8, 3, padding=1)   # 1 input channel because MRI is grayscale (not RGB),  8 different channels, each becomes a different “detector” for certain shapes or structures inside the 3D brain.
        self.bn1   = nn.BatchNorm3d(8)                # Normalize each of those 8 versions so their values don’t get too big or too small.

        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)   # 8 -> 16 channels
        self.bn2   = nn.BatchNorm3d(16)               # batch norm for 16 channels

        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)  # 16 -> 32 channels
        self.bn3   = nn.BatchNorm3d(32)               # batch norm for 32 channels

        self.pool  = nn.MaxPool3d(2)                  # halves D,H,W each time (downsampling)
                # --- Classifier head ---
        # AdaptiveAvgPool3d makes output size fixed no matter the input size
        self.gap   = nn.AdaptiveAvgPool3d((1, 1, 1))  # compresses 3D volume to 1x1x1 per channel

        self.fc    = nn.Linear(32, 1)                 # final binary logit (one number)
    def forward(self, x):                              # defines the data flow
        # x shape: [B, 1, 128, 128, 128]

        x = self.conv1(x)                              # conv features
        x = self.bn1(x)                                # normalize activations
        x = F.relu(x)                                  # non-linearity
        x = self.pool(x)                               # downsample -> [B, 8, 64, 64, 64]

        x = self.conv2(x)                              # conv stack
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)                               # -> [B, 16, 32, 32, 32]

        x = self.conv3(x)                              # deeper features
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)                               # -> [B, 32, 16, 16, 16]

        x = self.gap(x)                                # -> [B, 32, 1, 1, 1]
        x = x.view(x.size(0), -1)                      # flatten -> [B, 32]

        logit = self.fc(x)                             # -> [B, 1] (raw score, NOT probability)
        return logit                                   # we use BCEWithLogitsLoss on this
model = Simple3DCNN()
x = torch.randn(4, 1, 128, 128, 128)
y = model(x)
print(y.shape)

