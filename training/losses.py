import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, size_weight=1.0, yaw_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.yaw_weight = yaw_weight

    def forward(self, pred, target):
        """
        pred:
          heatmap logits: (B,1,H,W)
          size: (B,2,H,W)
          yaw: (B,1,H,W)

        target:
          heatmap: (B,1,H,W) in {0,1}
          size: (B,2,H,W) only defined at centers
          yaw: (B,1,H,W) only defined at centers
        """
        # Heatmap loss (dense)
        heatmap_loss = self.bce(pred["heatmap"], target["heatmap"])
        # Mask for regression = where centers exist
        with torch.no_grad():
            mask = (target["heatmap"] > 0.5).float()  # (B,1,H,W)
        # Avoid divide-by-zero if no objects in frame
        denom = mask.sum().clamp(min=1.0)
        # Size L1 at centers
        size_l1 = torch.abs(pred["size"] - target["size"])  # (B,2,H,W)
        size_l1 = (size_l1 * mask).sum() /denom
        # Yaw L1 at centers (simple)
        yaw_l1 = torch.abs(pred["yaw"] - target["yaw"])  # (B,1,H,W)
        yaw_l1 = (yaw_l1 * mask).sum() / denom

        total_loss = (self.heatmap_weight * heatmap_loss + self.size_weight * size_l1 + self.yaw_weight * yaw_l1)
        return {
            "total": total_loss,
            "heatmap": heatmap_loss.detach(),
            "size": size_l1.detach(),
            "yaw": yaw_l1.detach(),
        }


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
