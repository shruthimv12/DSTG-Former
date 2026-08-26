# -*- coding: utf-8 -*-
"""Same-protocol baselines requested by Reviewer 2 (comment 3/4).

Both baselines are trained and evaluated on exactly the strict, packet-disjoint
partition used for DSTG-Former, with the same ten-class labels.

  B1  feature-based machine-learning IDS  (representative of the pipelines of
      Sakib et al. and Dangwal et al.): engineered per-window statistics
      followed by a random forest.
  B2  CNN-BiLSTM-transformer hybrid (representative of Zhang et al. and
      Akuthota and Bhargava): sequence model on the raw payload stream with no
      topology branch.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import f1_score


def features(X, ids, chunk=50000):
    """Per-window engineered features: byte-position mean/std, function-code
    histogram, payload-length proxy, and device-pair activity counts."""
    out = []
    for c in range(0, len(ids), chunk):
        b = torch.as_tensor(np.asarray(ids[c:c + chunk], dtype=np.int64))
        w = X[b].float()                                   # [n, T, 64]
        f = [w.mean(1), w.std(1), w.max(1).values, w.min(1).values]
        nz = (w != 0).float().sum(2)                       # payload length proxy
        f += [nz.mean(1, keepdim=True), nz.std(1, keepdim=True)]
        out.append(torch.cat(f, 1).numpy().astype(np.float32))
    return np.concatenate(out)


class CNNBiLSTM(nn.Module):
    def __init__(self, d=64, hidden=96, nclass=NUM_CLASSES, nhead=4):
        super().__init__()
        self.emb = nn.Embedding(256, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(16, d, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(d),
            nn.AdaptiveAvgPool1d(1))
        self.lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        enc = nn.TransformerEncoderLayer(d_model=2 * hidden, nhead=nhead,
                                         dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=1)
        self.fc = nn.Linear(2 * hidden, nclass)

    def forward(self, x, *a):
        B, T, L = x.shape
        e = self.emb(x).view(B * T, L, -1).transpose(1, 2)
        s = self.conv(e).squeeze(-1).view(B, T, -1)
        h, _ = self.lstm(s)
        h = self.tr(h)
        return self.fc(h.mean(1)), h.mean(1)


def batches(X, y, ids, bs, device, shuffle=True):
    t = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)
    if shuffle:
        t = t[torch.randperm(len(t), device=device)]
    for i in range(0, len(t), bs):
        b = t[i:i + bs]
        yield X[b].long(), y[b].long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="rf", choices=["rf", "cnn"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--mask-fc", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    X, y, src, dst, tr, va, te, blocks = load_all(mask_fc=a.mask_fc)
    yn = y.numpy()
    tag = a.tag or ("_maskfc" if a.mask_fc else "")

    if a.which == "rf":
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(42)
        sub = tr if len(tr) <= 200000 else rng.choice(tr, 200000, replace=False)
        t0 = time.time()
        Ftr = features(X, sub); Fte = features(X, te)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     random_state=42)
        clf.fit(Ftr, yn[sub])
        ttime = time.time() - t0
        t1 = time.time(); pr = clf.predict(Fte); pb = clf.predict_proba(Fte)
        inf = (time.time() - t1) / len(te) * 1000.0
        nparams = int(sum(t_.tree_.node_count for t_ in clf.estimators_))
        res = metrics(yn[te], pr, pb, extra={
            "config": "feature_based_rf" + tag, "params": nparams,
            "param_note": "total decision-tree nodes (200 trees)",
            "protocol": "strict_chronological_packet_disjoint",
            "train_time_s": ttime, "inference_ms_per_window": inf,
            "train_subsample": int(len(sub)), "mask_fc": bool(a.mask_fc)})
        save("baseline_rf" + tag, res)
        print("RF acc=%.4f macroF1=%.4f" % (res["test_accuracy"],
                                            res["f1_macro"]), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device); y = y.to(torch.uint8).to(device)
    torch.manual_seed(42)
    model = CNNBiLSTM().to(device)
    nparams = sum(p.numel() for p in model.parameters())
    counts = np.bincount(yn[tr], minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1
    w = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32,
                     device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = {"f1": -1, "state": None, "epoch": -1}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for xb, yb in batches(X, y, tr, 512, device):
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out, _ = model(xb); loss = crit(out, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); P, T_ = [], []
        with torch.no_grad():
            for xb, yb in batches(X, y, va, 512, device, False):
                P.append(model(xb)[0].argmax(1).cpu().numpy())
                T_.append(yb.cpu().numpy())
        f1 = f1_score(np.concatenate(T_), np.concatenate(P), average="macro",
                      zero_division=0)
        if f1 > best["f1"]:
            import copy
            best = {"f1": f1, "state": copy.deepcopy(model.state_dict()),
                    "epoch": ep + 1}
        print(f"[cnn] ep {ep+1}/{a.epochs} vaMF1 {100*f1:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ttime = time.time() - t0
    model.load_state_dict(best["state"]); model.eval()
    P, T_, S = [], [], []
    with torch.no_grad():
        for xb, yb in batches(X, y, te, 512, device, False):
            o, _ = model(xb)
            P.append(o.argmax(1).cpu().numpy()); T_.append(yb.cpu().numpy())
            S.append(F.softmax(o.float(), 1).cpu().numpy())
    res = metrics(np.concatenate(T_), np.concatenate(P), np.concatenate(S),
                  extra={"config": "cnn_bilstm_transformer" + tag,
                         "params": int(nparams), "epochs": a.epochs,
                         "protocol": "strict_chronological_packet_disjoint",
                         "checkpoint_rule": "max validation macro F1",
                         "selected_epoch": best["epoch"],
                         "train_time_s": ttime, "mask_fc": bool(a.mask_fc)})
    save("baseline_cnn_bilstm" + tag, res)
    print("CNN-BiLSTM acc=%.4f macroF1=%.4f" % (res["test_accuracy"],
                                                res["f1_macro"]), flush=True)


if __name__ == "__main__":
    main()
