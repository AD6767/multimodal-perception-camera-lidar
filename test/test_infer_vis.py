import torch
import numpy as np

from models.bev_detector import BEVDetector
from bev.decode import topk_heatmap
from visualization.bev_viz import visualize_bev_gt, draw_bev_boxes  # reuse your plotting utils
from bev.bev_config import BEVConfig

def bev_pixel_to_world(cx, cy, config: BEVConfig):
    x = config.X_MIN + cx * config.RESOLUTION
    y = config.Y_MIN + cy * config.RESOLUTION
    return x, y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = BEVConfig()

    model = BEVDetector(in_channels=3).to(device)
    model.load_state_dict(torch.load("models_bev_detector.pt", map_location=device))
    model.eval()

    # TODO: replace with real BEV tensor and GT boxes for one frame
    bev = torch.randn(1, 3, config.HEIGHT, config.WIDTH).to(device)
    gt_bev_boxes = []  # list of (x,y,w,l,yaw) in world meters

    with torch.no_grad():
        pred = model(bev)
        hm = torch.sigmoid(pred["heatmap"])
        picks = topk_heatmap(hm, K=50, thresh=0.3)[0]

        pred_boxes = []
        for score, cy, cx in picks:
            # decode regression at that cell
            w = float(pred["size"][0, 0, cy, cx].cpu())
            l = float(pred["size"][0, 1, cy, cx].cpu())
            yaw = float(pred["yaw"][0, 0, cy, cx].cpu())

            x, y = bev_pixel_to_world(cx, cy, config)
            pred_boxes.append(np.array([x, y, w, l, yaw]))

    # Plot: GT in red, preds in green
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,7))
    if len(gt_bev_boxes) > 0:
        draw_bev_boxes(ax, gt_bev_boxes, color="r")
    if len(pred_boxes) > 0:
        draw_bev_boxes(ax, pred_boxes, color="g")
    ax.set_aspect("equal")
    ax.set_title("BEV Predictions (green) vs GT (red)")
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
