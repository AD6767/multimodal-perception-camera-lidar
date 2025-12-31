import torch
import torch.nn as nn
from models.bev_backbone import BEVBackbone
from models.detection_head import DetectionHead


class BEVDetector(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.backbone = BEVBackbone(in_channels=in_channels, base_channels=32, out_channels=64)
        self.head = DetectionHead(in_channels=self.backbone.out_channels)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs
