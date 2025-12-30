import torch
from models.bev_backbone import BEVBackbone
from models.detection_head import DetectionHead

def main():
    B, C, H, W = 1, 3, 400, 400
    x = torch.randn(B, C, H, W)

    backbone = BEVBackbone(in_channels=C) # Default base_channels=32, out_channels=64
    features = backbone(x) # Output shape: (1, 64, 100, 100) # [B, C_out, H/4, W/4]

    head = DetectionHead(in_channels=features.shape[1]) # in_channels=64
    outputs = head(features) # Outputs: dict with 'heatmap', 'size', 'yaw'

    print("Feature map:", features.shape) # torch.Size([1, 64, 100, 100])
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
    
    """
    Feature map: torch.Size([1, 64, 100, 100])
    heatmap: torch.Size([1, 1, 100, 100]) -- Each cell predicts: probability of object center
    size: torch.Size([1, 2, 100, 100]) -- Each cell predicts (regression values): (width, length)
    yaw: torch.Size([1, 1, 100, 100]) -- Each cell predicts (regression value): orientation in radians (yaw angle)
    """

if __name__ == "__main__":
    main()
