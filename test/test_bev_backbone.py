import torch
from models.bev_backbone import BEVBackbone
from bev.bev_config import BEVConfig

def main():
    B, C, H, W = 1, 3, BEVConfig().HEIGHT, BEVConfig().WIDTH
    x = torch.randn(B, C, H, W)

    model = BEVBackbone(in_channels=C, base_channels=32, out_channels=64)
    y = model(x)

    print("Input:", x.shape) # torch.Size([1, 3, 800, 704])
    print("Output:", y.shape) # torch.Size([1, 64, 200, 176])
    print("Backbone stride:", model.stride) # 4

if __name__ == "__main__":
    main()
