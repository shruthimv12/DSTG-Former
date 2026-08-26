# -*- coding: utf-8 -*-
"""End-to-end deployment benchmark requested by Reviewer 2 (comment 6).

Measures the complete packet-to-alarm path, not just the network forward pass:
payload standardization, link-layer address decoding and graph construction,
window assembly, and inference.  Reports single-window latency, sustained
throughput, peak memory, and the intrinsic buffering delay of the sliding
window.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from train_dstg import DSTGFormer

BIG = dict(d_model=128, node_dim=64, nhead=8, layers=3, use_pe=False)


def preprocess(raw):
    """Standardization + address decoding + graph index construction, exactly
    as the online front end would perform them for one window."""
    x = torch.zeros((raw.shape[0], 64), dtype=torch.uint8)
    n = min(64, raw.shape[1])
    x[:, :n] = raw[:, :n]
    dst = ADDR_LUT[(x[:, 4].long() + x[:, 5].long() * 256).clamp(0, 65535)]
    src = ADDR_LUT[(x[:, 6].long() + x[:, 7].long() * 256).clamp(0, 65535)]
    return x, src, dst


def main():
    d = torch.load(os.path.join(BASE, "dnp3_transformer_dataset.pt"))
    X = d["data"][:20000]
    out = {}
    for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(dev)
        model = DSTGFormer(**BIG).to(device).eval()
        sd = os.path.join(OUT, "strict_main_model.pt")
        if os.path.exists(sd):
            model.load_state_dict(torch.load(sd, map_location=device))
        nparams = sum(p.numel() for p in model.parameters())
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- single-window, end-to-end latency ----
        lat = []
        with torch.no_grad():
            for i in range(300):
                w = X[i]
                t0 = time.perf_counter()
                xb, sb, db = preprocess(w)
                xb = xb.unsqueeze(0).long().to(device)
                sb = sb.unsqueeze(0).long().to(device)
                db = db.unsqueeze(0).long().to(device)
                model(xb, sb, db)
                if dev == "cuda":
                    torch.cuda.synchronize()
                if i >= 50:
                    lat.append((time.perf_counter() - t0) * 1000.0)
        lat = np.array(lat)

        # ---- sustained throughput, batch 256 ----
        bs = 256
        Xb = X[:bs * 20]
        with torch.no_grad():
            for warm in range(2):
                t0 = time.perf_counter()
                nw = 0
                for i in range(0, len(Xb), bs):
                    blk = Xb[i:i + bs]
                    xs, ss, ds = [], [], []
                    for w in blk:
                        a, b, c = preprocess(w)
                        xs.append(a); ss.append(b); ds.append(c)
                    xb = torch.stack(xs).long().to(device)
                    sb = torch.stack(ss).long().to(device)
                    db = torch.stack(ds).long().to(device)
                    model(xb, sb, db)
                    nw += len(blk)
                if dev == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
        thr = nw / dt

        rec = {"params": int(nparams),
               "latency_ms_mean": float(lat.mean()),
               "latency_ms_p95": float(np.percentile(lat, 95)),
               "throughput_windows_per_s": float(thr),
               "throughput_packets_per_s": float(thr * STRIDE),
               "detection_delay_packets": SEQ_LEN,
               "decision_interval_packets": STRIDE}
        if dev == "cuda":
            rec["peak_gpu_memory_MB"] = float(
                torch.cuda.max_memory_allocated() / 1024 ** 2)
        rec["model_size_MB"] = float(nparams * 4 / 1024 ** 2)
        out[dev] = rec
        print(dev, json.dumps(rec, indent=1), flush=True)
    save("deployment_benchmark", out)


if __name__ == "__main__":
    main()
