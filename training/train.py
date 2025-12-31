import torch
from torch.optim import Adam
from models.bev_detector import BEVDetector
from training.losses import DetectionLoss


def train_one_epoch(model, loss_fn, optimizer, dataloader, device="cpu"):
    model.train()
    running = 0.0

    for step, batch in enumerate(dataloader):
        x = batch["bev"].to(device)  # (B,C,H,W)
        target = {k: v.to(device) for k, v in batch["target"].items()}

        pred = model(x)  # dict with 'heatmap', 'size', 'yaw'
        losses = loss_fn(pred, target)
        loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        if step % 20 == 0:
            print(f"step {step:04d} | total={loss.item():.4f} "
                  f"| hm={losses['heatmap']:.4f} size={losses['size']:.4f} yaw={losses['yaw']:.4f}")
            
    return running / max(len(dataloader), 1) # average loss over epoch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BEVDetector().to(device)
    loss_fn = DetectionLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # TODO: replace with real dataloader
    dataloader = []  # iterable of dicts: {"bev":..., "target":...}

    for epoch in range(5):
        avg_loss = train_one_epoch(model, loss_fn, optimizer, dataloader, device)
        print(f"epoch {epoch} avg loss = {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "models_bev_detector.pt")
    print("saved models_bev_detector.pt")

if __name__ == "__main__":
    main()
