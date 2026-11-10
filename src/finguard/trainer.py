import copy, numpy as np, torch
from sklearn.metrics import average_precision_score


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for xb, _ in loader: out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def fit(model, train_dl, val_dl, y_val, loss_fn, device="cpu", epochs=40, lr=1e-3, wd=1e-4,
        patience=6, clip=1.0, ckpt="artifacts/best_mlp.pt", use_amp=False, verbose=True):
    """Early stopping on VAL PR-AUC (not loss: loss is dominated by the majority class)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=use_amp and device == "cuda")
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp and device == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); tot += loss.item() * len(yb)
        ap = average_precision_score(y_val, predict(model, val_dl, device)); sched.step(ap)
        hist.append({"epoch": ep, "train_loss": tot / len(train_dl.dataset), "val_pr_auc": ap})
        if verbose:
            print(f"ep{ep:02d} loss={hist[-1]['train_loss']:.4f} val_PR-AUC={ap:.4f}")
        if ap > best: best, bad, best_state = ap, 0, copy.deepcopy(model.state_dict()); torch.save(best_state, ckpt)
        else:
            bad += 1
            if bad >= patience:
                if verbose: print("early stop")
                break
    model.load_state_dict(best_state)
    return model, hist
