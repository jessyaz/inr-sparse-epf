import numpy as np
import torch

from data.loader import inverse_price


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        pred = model.forward_step(batch, device)
        total += ((pred - batch["Y"].to(device)) ** 2).mean().item()
        n += 1
    return total / max(n, 1)


def train(model, train_loader, val_loader, device, cfg):
    t = cfg.train
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, cooldown=3, min_lr=1e-6)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    best_val, best_state, bad_epochs, step = float("inf"), None, 0, 0
    for epoch in range(t.num_epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            if step < t.warmup_steps:
                for g, lr in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = lr * min(1.0, (step + 1) / t.warmup_steps)

            loss = model.loss(batch, device)
            optimizer.zero_grad()
            loss.backward()
            if t.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            step += 1
            running += loss.item()
            n += 1

        val = validate(model, val_loader, device)
        if step >= t.warmup_steps:
            scheduler.step(val)
        print(f"epoch {epoch + 1:3d}  train {running / max(n, 1):.5f}  val {val:.5f}")

        if val < best_val:
            best_val, bad_epochs = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= t.patience:
                break

    model.load_state_dict(best_state)


@torch.no_grad()
def test_mae(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        preds.append(model.forward_step(batch, device).cpu().numpy())
        trues.append(batch["Y"].numpy())
    preds = inverse_price(scaler, np.concatenate(preds))
    trues = inverse_price(scaler, np.concatenate(trues))
    return float(np.abs(preds - trues).mean())
