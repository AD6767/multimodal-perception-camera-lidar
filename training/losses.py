import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) - predicted heatmap (after sigmoid)
        target: (B, 1, H, W) - ground truth heatmap
        """
        pos_mask = target == 1
        neg_mask = target < 1

        pos_loss = -torch.log(pred + 1e-6) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = -torch.log(1 - pred + 1e-6) * torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * neg_mask
        return (pos_loss + neg_loss).mean()
    
def size_loss(pred, target, mask):
    """
    pred: (B, 2, H, W)
    target: (B, 2, H, W)
    mask: (B, 1, H, W)  -> center locations
    """
    mask = mask.expand_as(pred)
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    return loss / (mask.sum() + 1e-6)

def yaw_loss(pred, target, mask):
    """
    pred: (B, 1, H, W)
    target: (B, 1, H, W)
    mask: (B, 1, H, W)
    """
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    return loss / (mask.sum() + 1e-6)

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmap_loss = FocalLoss()

    def forward(self, preds, targets):
        """
        preds: dict from detection head
        targets: dict from BEV target generator
        """
        heatmap_pred = torch.sigmoid(preds["heatmap"])
        heatmap_gt = targets["heatmap"]

        size_pred = preds["size"]
        size_gt = targets["size"]

        yaw_pred = preds["yaw"]
        yaw_gt = targets["yaw"]

        center_mask = heatmap_gt

        loss_hm = self.heatmap_loss(heatmap_pred, heatmap_gt)
        loss_size = size_loss(size_pred, size_gt, center_mask)
        loss_yaw = yaw_loss(yaw_pred, yaw_gt, center_mask)

        total_loss = loss_hm + loss_size + loss_yaw

        return {
            "loss_total": total_loss,
            "loss_heatmap": loss_hm,
            "loss_size": loss_size,
            "loss_yaw": loss_yaw,
        }
