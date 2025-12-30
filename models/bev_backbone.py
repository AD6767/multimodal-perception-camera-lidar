import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)
    
class BEVBackbone(nn.Module):
    """
    Minimal BEV Backbone implementation with convolutional layers.
    Produces feature maps at stride = 4 by default.
    Input (1, 3, 400, 400) # Channels [0: height, 1: density, 2: intensity]
    │
    ├─ Stem (no downsampling)
    │   ├─ Conv 3*3, stride 1 → 32
    │   └─ Conv 3*3, stride 1 → 32
    │
    ├─ Downsample 1 (stride 2)
    │   ├─ Conv 3*3, stride 2 → 64
    │   └─ Conv 3*3, stride 1 → 64
    │
    ├─ Downsample 2 (stride 2)
    │   ├─ Conv 3*3, stride 2 → 64
    │   └─ Conv 3*3, stride 1 → 64
    │
    └─ Output (1, 64, 100, 100)
    BEV Feature Map: (resolution = 0.25 m)
    ┌─────────────────────────────┐
    │ f11  f12  f13  ...          │
    │ f21  f22  f23               │
    │                             │
    │ each fij is a 64-D vector   │
    │ representing ~1m^2 region   │
    └─────────────────────────────┘
    """
    def __init__(self, in_channels: int, base_channels: int = 32, out_channels: int = 64):
        super().__init__()
        # Stem (no downsample)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
        )
        # Downsample 1: stride = 2
        self.down1 = nn.Sequential(
            ConvBNReLU(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
        )
        # Downsample 2: stride = 2
        self.down2 = nn.Sequential(
            ConvBNReLU(base_channels * 2, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.stride = 4  # Total stride of the backbone
        self.out_channels = out_channels # Output feature channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Example input and output shapes:
        # Input: (1, 3, 400, 400) # [B, C_in, H, W]
        # After stem (no stride): (1, 32, 400, 400) # [B, C_base, H, W]
        # After down1 (stride=2): (1, 64, 200, 200) # [B, C_base*2, H/2, W/2] # each cell now covers 2 × 2 BEV cells
        # After down2 (stride=2): (1, 64, 100, 100) # [B, C_out, H/4, W/4] # each cell now covers 4 × 4 BEV cells
        # Output: (1, 64, 100, 100) # [B, C_out, H/4, W/4] # If BEV resolution = 0.25 m, each cell covers 1 m × 1 m area.
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return x

