import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Heatmap head (classification)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)  # Output: 1 channel heatmap
        )

        # Size head (w, l regression)
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)  # Output: 2 channels (w, l)
        )

        # Yaw head (orientation regression)
        self.yaw_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)  # Output: 1 channel (yaw)
        )

    def forward(self, x):
        heatmap = self.heatmap_head(x)
        size = self.size_head(x)
        yaw = self.yaw_head(x)
        return {'heatmap': heatmap, 'size': size, 'yaw': yaw}
