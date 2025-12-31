import torch

def topk_heatmap(heatmap_sigmoid: torch.Tensor, K: int = 50, thresh: float = 0.3):
    """
    heatmap_sigmoid: (B,1,H,W) after sigmoid
    Returns list per batch: [(score, cy, cx), ...]
    """
    B, _, H, W = heatmap_sigmoid.shape
    results = []
    for b in range(B):
        hm = heatmap_sigmoid[b, 0]
        scores, idx = torch.topk(hm.flatten(), k=min(K, hm.numel()))
        picks = []
        for s, i in zip(scores, idx):
            if float(s) < thresh:
                continue
            cy = int(i // W)
            cx = int(i % W)
            picks.append((float(s), cy, cx))
        results.append(picks)
    return results
