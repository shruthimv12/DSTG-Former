# -*- coding: utf-8 -*-
"""
DSTG-Former: Domain-Informed Spatial-Temporal Graph-Transformer for DNP3
intrusion detection.

Improvements over the original notebook:
  * Graph edges are built from the REAL DNP3 link-layer addresses embedded in
    every payload (bytes 4-5 = destination, bytes 6-7 = source, little-endian)
    instead of random edges -> genuinely topology-aware.
  * Pure-PyTorch vectorised EC-GNN (identical math to the PyG version but runs
    the whole [B, T] block in one shot on the GPU).
  * Sinusoidal positional encoding so the transformer sees temporal order.
  * Class-weighted loss for the imbalanced classes.
  * Proper train/val/test split (70/15/15, stratified) and full test metrics.
  * Ablation configurations.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score)

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = [
    "Disable Unsolicited", "Cold Restart", "Warm Restart", "Enumerate",
    "Info", "Initialize Data", "MITM-DoS", "Replay", "Stop Application",
    "Normal",
]
NUM_CLASSES = 10

# Protocol-semantics-guided window labeling:
# in the five command-injection captures the attack consists of a handful of
# packets carrying a specific DNP3 application-layer function code (payload
# offset 12).  Windows without that function code are regular polling traffic
# and are relabeled as Normal, mirroring the flow-level labeling of the
# original dataset (attack flows vs NORMAL).
ATTACK_FC = {0: 0x15, 1: 0x0D, 2: 0x0E, 5: 0x0F, 8: 0x12}


def semantic_relabel(X, y):
    y = y.clone()
    fc = X[:, :, 12]
    for c, code in ATTACK_FC.items():
        m = y == c
        has_attack = (fc[m] == code).any(dim=1)
        idx = torch.nonzero(m, as_tuple=True)[0]
        y[idx[~has_attack]] = 9          # Normal
    return y

# DNP3 link-layer address vocabulary observed in the dataset
ADDR_VOCAB = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18]
NUM_NODES = len(ADDR_VOCAB)          # 13 devices
ADDR_LUT = torch.full((65536,), 0, dtype=torch.long)
for i, a in enumerate(ADDR_VOCAB):
    ADDR_LUT[a] = i


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # x: [B, T, d]
        return x + self.pe[:, :x.size(1)]


class DSTGFormer(nn.Module):
    """Edge-conditioned GNN over per-snapshot device graphs, followed by a
    transformer encoder over the sequence of graph embeddings.

    Stabilised design: layer normalisation on the edge messages and graph
    embeddings, a residual protocol-semantics pathway around the GNN readout,
    and an attention-based temporal readout after the transformer."""

    def __init__(self, payload=64, d_model=64, node_dim=32, nhead=4,
                 layers=2, num_classes=NUM_CLASSES, num_nodes=NUM_NODES,
                 dropout=0.1, use_gnn=True, use_transformer=True, use_pe=True,
                 random_topology=False, attn_pool=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer
        self.use_pe = use_pe
        self.random_topology = random_topology
        self.attn_pool = attn_pool
        self.num_nodes = num_nodes

        self.byte_emb = nn.Embedding(256, d_model)
        self.edge_proj = nn.Linear(payload * d_model, d_model)
        self.ln_msg = nn.LayerNorm(d_model)
        if use_gnn:
            self.node_emb = nn.Embedding(num_nodes, node_dim)
            self.node_update = nn.Linear(node_dim + d_model, node_dim)
            self.final_proj = nn.Linear(node_dim, d_model)
            self.ln_z = nn.LayerNorm(d_model)
        if use_transformer:
            self.pos_enc = PositionalEncoding(d_model)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                             dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        if attn_pool:
            self.pool_q = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, payload_bytes, src_idx, dst_idx):
        # payload_bytes: [B, T, 64] long; src/dst_idx: [B, T] long
        B, T, L = payload_bytes.shape
        # Disentangled information routing: device identity (link-layer
        # addresses, bytes 4-7) and the address-dependent first CRC (bytes
        # 8-9) are consumed exclusively by the topology branch; the semantic
        # branch sees the payload with those bytes masked.
        sem = payload_bytes.clone()
        sem[:, :, 4:10] = 0
        emb = self.byte_emb(sem)                           # [B, T, 64, d]
        msg = self.ln_msg(F.relu(self.edge_proj(emb.view(B, T, -1))))

        if self.use_gnn:
            if self.random_topology:
                dst_idx = torch.randint(0, self.num_nodes, dst_idx.shape,
                                        device=dst_idx.device)
                src_idx = torch.randint(0, self.num_nodes, src_idx.shape,
                                        device=src_idx.device)
            V, dn, dm = self.num_nodes, self.node_emb.embedding_dim, msg.size(-1)
            # scatter each edge message onto both endpoint devices
            aggr = torch.zeros(B, T, V, dm, device=msg.device, dtype=msg.dtype)
            aggr.scatter_(2, dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, dm),
                          msg.unsqueeze(2))
            aggr.scatter_(2, src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, dm),
                          msg.unsqueeze(2))
            h0 = self.node_emb.weight.view(1, 1, V, dn).expand(B, T, V, dn)
            h = F.relu(self.node_update(torch.cat([h0, aggr], dim=-1)))
            # graph readout + residual protocol-semantics pathway
            z = self.ln_z(self.final_proj(h).mean(dim=2) + msg)
        else:
            z = msg                                        # payload-only branch

        if self.use_transformer:
            if self.use_pe:
                z = self.pos_enc(z)
            z = self.transformer(z)
        if self.attn_pool:
            a = torch.softmax(self.pool_q(z), dim=1)       # [B, T, 1]
            feat = (a * z).sum(dim=1)                      # attentive readout
        else:
            feat = z.mean(dim=1)                           # mean pooling
        return self.fc(feat), feat


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(seq_len=20, relabel=True):
    d = torch.load(os.path.join(BASE, "dnp3_transformer_dataset.pt"))
    X, y = d["data"], d["labels"].long()
    if relabel:
        y = semantic_relabel(X, y)
    if seq_len < X.shape[1]:
        X = X[:, :seq_len]
    dst = ADDR_LUT[(X[:, :, 4].long() + X[:, :, 5].long() * 256).clamp(0, 65535)]
    src = ADDR_LUT[(X[:, :, 6].long() + X[:, :, 7].long() * 256).clamp(0, 65535)]
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y.numpy(),
                               random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y.numpy()[tmp],
                              random_state=42)
    return X, y, src, dst, tr, va, te


def batches(X, y, src, dst, ids, bs, device, shuffle=True):
    # X/y/src/dst are expected to already live on `device` (uint8 / int8)
    ids_t = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)
    if shuffle:
        ids_t = ids_t[torch.randperm(len(ids_t), device=device)]
    for i in range(0, len(ids_t), bs):
        b = ids_t[i:i + bs]
        yield (X[b].long(), src[b].long(), dst[b].long(), y[b].long())


# --------------------------------------------------------------------------
# Train / evaluate one configuration
# --------------------------------------------------------------------------
def run_config(name, cfg, X, y, src, dst, tr, va, te, epochs=30, bs=1024,
               lr=3e-4, save_features=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = DSTGFormer(**cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(y.cpu().numpy().astype(np.int64), minlength=NUM_CLASSES)
    w = torch.tensor((counts.sum() / (NUM_CLASSES * counts)),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, sb, db, yb in batches(X, y, src, dst, tr, bs, device):
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out, _ = model(xb, sb, db)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            tn += len(yb)
        sched.step()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, sb, db, yb in batches(X, y, src, dst, va, bs, device,
                                          shuffle=False):
                out, _ = model(xb, sb, db)
                vl += crit(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
                vn += len(yb)
        hist["train_loss"].append(tl / tn)
        hist["val_loss"].append(vl / vn)
        hist["train_acc"].append(100.0 * tc / tn)
        hist["val_acc"].append(100.0 * vc / vn)
        print(f"[{name}] ep {ep+1}/{epochs} "
              f"trL {tl/tn:.4f} trA {100*tc/tn:.2f} "
              f"vaL {vl/vn:.4f} vaA {100*vc/vn:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    train_time = time.time() - t0

    # ---- test ----
    model.eval()
    preds, targs, probs, feats = [], [], [], []
    with torch.no_grad():
        for xb, sb, db, yb in batches(X, y, src, dst, te, bs, device,
                                      shuffle=False):
            out, ft = model(xb, sb, db)
            probs.append(F.softmax(out, 1).cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            targs.append(yb.cpu().numpy())
            if save_features:
                feats.append(ft.cpu().numpy())
    preds = np.concatenate(preds); targs = np.concatenate(targs)
    probs = np.concatenate(probs)

    acc = accuracy_score(targs, preds)
    res = {
        "config": name, "params": int(nparams), "train_time_s": train_time,
        "epochs": epochs,
        "test_accuracy": acc,
        "precision_macro": precision_score(targs, preds, average="macro"),
        "recall_macro": recall_score(targs, preds, average="macro"),
        "f1_macro": f1_score(targs, preds, average="macro"),
        "precision_weighted": precision_score(targs, preds, average="weighted"),
        "recall_weighted": recall_score(targs, preds, average="weighted"),
        "f1_weighted": f1_score(targs, preds, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(targs, probs, multi_class="ovr",
                                           average="macro"),
        "report": classification_report(targs, preds,
                                        target_names=CLASS_NAMES,
                                        output_dict=True, digits=4),
        "confusion_matrix": confusion_matrix(targs, preds).tolist(),
        "history": hist,
    }
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    if save_features:
        feats = np.concatenate(feats)
        np.savez_compressed(os.path.join(OUT, f"{name}_test_feats.npz"),
                            feats=feats, targs=targs, probs=probs,
                            preds=preds)
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}_model.pt"))
    print(f"[{name}] DONE  acc={acc*100:.2f}%  f1M={res['f1_macro']*100:.2f}% "
          f"({train_time:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    print("[*] loading data ...", flush=True)
    X, y, src, dst, tr, va, te = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # park the whole dataset on the GPU in compact dtypes
    X = X.to(device)                      # uint8, ~930 MB
    y = y.to(torch.uint8).to(device)
    src = src.to(torch.uint8).to(device)
    dst = dst.to(torch.uint8).to(device)
    print(f"[*] train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    BIG = dict(d_model=128, node_dim=64, nhead=8, layers=3)
    CONFIGS = {
        "large":           dict(BIG),
        "no_gnn":          dict(BIG, use_gnn=False, use_pe=False),
        "no_transformer":  dict(BIG, use_transformer=False, use_pe=False),
        "random_topology": dict(BIG, random_topology=True, use_pe=False),
        "no_pe":           dict(BIG, use_pe=False),
        "mean_pool":       dict(BIG, attn_pool=False, use_pe=False),
    }
    for name, cfg in CONFIGS.items():
        if args.only and name != args.only:
            continue
        run_config(name, cfg, X, y, src, dst, tr, va, te,
                   epochs=args.epochs,
                   save_features=(name in ("full", "large", "no_pe")))


if __name__ == "__main__":
    main()
