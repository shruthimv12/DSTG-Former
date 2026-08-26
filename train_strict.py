# -*- coding: utf-8 -*-
"""Training driver for the IJIES revision (paper 20265400).

Every run uses the strict, packet-disjoint chronological protocol built in
common.py.  Checkpoint selection is explicit: the epoch with the highest
validation macro F1 is restored before the single test-set evaluation.
"""
import os, sys, json, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from train_dstg import DSTGFormer
from sklearn.metrics import f1_score

BIG = dict(d_model=128, node_dim=64, nhead=8, layers=3)


def batches(X, y, src, dst, ids, bs, device, shuffle=True):
    ids_t = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)
    if shuffle:
        ids_t = ids_t[torch.randperm(len(ids_t), device=device)]
    for i in range(0, len(ids_t), bs):
        b = ids_t[i:i + bs]
        yield X[b].long(), src[b].long(), dst[b].long(), y[b].long()


def evaluate(model, X, y, src, dst, ids, bs, device, want_probs=True):
    model.eval()
    P, T, S = [], [], []
    with torch.no_grad():
        for xb, sb, db, yb in batches(X, y, src, dst, ids, bs, device, False):
            out, _ = model(xb, sb, db)
            P.append(out.argmax(1).cpu().numpy())
            T.append(yb.cpu().numpy())
            if want_probs:
                S.append(F.softmax(out.float(), 1).cpu().numpy())
    return (np.concatenate(P), np.concatenate(T),
            np.concatenate(S) if want_probs else None)


def run(name, cfg, X, y, src, dst, tr, va, te, epochs, bs, lr, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DSTGFormer(**cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(y[tr].cpu().numpy().astype(np.int64),
                         minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = torch.tensor(counts.sum() / (NUM_CLASSES * counts),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "val_macro_f1": []}
    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = tc = tn = 0
        for xb, sb, db, yb in batches(X, y, src, dst, tr, bs, device):
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out, _ = model(xb, sb, db); loss = crit(out, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(yb); tc += (out.argmax(1) == yb).sum().item()
            tn += len(yb)
        sched.step()
        vp, vt, _ = evaluate(model, X, y, src, dst, va, bs, device, False)
        vmf1 = f1_score(vt, vp, average="macro", zero_division=0)
        vacc = float((vp == vt).mean())
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(100 * tc / tn)
        hist["val_acc"].append(100 * vacc); hist["val_macro_f1"].append(100 * vmf1)
        hist["val_loss"].append(0.0)
        if vmf1 > best["macro_f1"]:
            best = {"macro_f1": vmf1, "epoch": ep + 1, "acc": vacc,
                    "state": copy.deepcopy(model.state_dict())}
        print(f"[{name}] ep {ep+1}/{epochs} trA {100*tc/tn:.2f} "
              f"vaA {100*vacc:.2f} vaMF1 {100*vmf1:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ttime = time.time() - t0
    model.load_state_dict(best["state"])
    pr, tg, pb = evaluate(model, X, y, src, dst, te, bs, device)
    res = metrics(tg, pr, pb, extra={
        "config": name, "params": int(nparams), "seed": seed,
        "epochs": epochs, "train_time_s": ttime,
        "protocol": "strict_chronological_packet_disjoint",
        "checkpoint_rule": "max validation macro F1",
        "selected_epoch": best["epoch"],
        "val_accuracy_at_checkpoint": float(best["acc"]),
        "val_macro_f1_at_checkpoint": float(best["macro_f1"]),
        "history": hist})
    save(name, res)
    torch.save(best["state"], os.path.join(OUT, name + "_model.pt"))
    print(f"[{name}] DONE acc={res['test_accuracy']*100:.2f} "
          f"macroF1={res['f1_macro']*100:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-fc", action="store_true")
    ap.add_argument("--variant", default="full",
                    choices=["full", "no_gnn", "no_transformer",
                             "random_topology", "mean_pool", "with_pe"])
    ap.add_argument("--holdout-devices", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, src, dst, tr, va, te, blocks = load_all(mask_fc=a.mask_fc)

    if a.holdout_devices:
        hd = [ADDR_VOCAB.index(int(v)) for v in a.holdout_devices.split(",")]
        touch = torch.zeros(len(y), dtype=torch.bool)
        for h in hd:
            touch |= (src == h).any(1) | (dst == h).any(1)
        touch = touch.numpy()
        allidx = np.arange(len(y))
        seen_idx = allidx[~touch]
        rng = np.random.RandomState(a.seed)
        rng.shuffle(seen_idx)
        cut = int(0.9 * len(seen_idx))
        tr, va, te = seen_idx[:cut], seen_idx[cut:], allidx[touch]
        print(f"[holdout] devices={a.holdout_devices} train={len(tr)} "
              f"val={len(va)} test={len(te)}", flush=True)

    X = X.to(device); y = y.to(torch.uint8).to(device)
    src = src.to(torch.uint8).to(device); dst = dst.to(torch.uint8).to(device)
    print(f"[*] {a.name}: train/val/test = {len(tr)}/{len(va)}/{len(te)} "
          f"mask_fc={a.mask_fc}", flush=True)

    cfg = dict(BIG, use_pe=False)
    if a.variant == "no_gnn":            cfg.update(use_gnn=False)
    elif a.variant == "no_transformer":  cfg.update(use_transformer=False)
    elif a.variant == "random_topology": cfg.update(random_topology=True)
    elif a.variant == "mean_pool":       cfg.update(attn_pool=False)
    elif a.variant == "with_pe":         cfg.update(use_pe=True)
    run(a.name, cfg, X, y, src, dst, tr, va, te, a.epochs, a.bs, a.lr,
        device, a.seed)


if __name__ == "__main__":
    main()
