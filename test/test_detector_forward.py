import torch
from models.bev_detector import BEVDetector

def main():
    x = torch.randn(1, 3, 704, 800)
    model = BEVDetector(in_channels=3)
    out = model(x)

    for k, v in out.items():
        print(k, v.shape)

if __name__ == "__main__":
    main()
