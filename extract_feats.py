# -*- coding: utf-8 -*-
"""Extract test-set window representations from the strict-protocol model and
project them with t-SNE for the qualitative figure."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from train_dstg import DSTGFormer
from sklearn.manifold import TSNE

BIG = dict(d_model=128, node_dim=64, nhead=8, layers=3, use_pe=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y, src, dst, tr, va, te, blocks = load_all()
model = DSTGFormer(**BIG).to(device).eval()
model.load_state_dict(torch.load(os.path.join(OUT, "strict_main_model.pt"),
                                 map_location=device))
rng = np.random.RandomState(0)
sel = np.concatenate([rng.choice(te[y.numpy()[te] == c],
                                 min(700, int((y.numpy()[te] == c).sum())),
                                 replace=False)
                      for c in range(NUM_CLASSES)])
F_, Y_ = [], []
with torch.no_grad():
    for i in range(0, len(sel), 512):
        b = torch.as_tensor(sel[i:i + 512], dtype=torch.long)
        _, f = model(X[b].long().to(device), src[b].long().to(device),
                     dst[b].long().to(device))
        F_.append(f.cpu().numpy())
        Y_.append(y.numpy()[sel[i:i + 512]])
F_ = np.concatenate(F_); Y_ = np.concatenate(Y_)
print("features", F_.shape, flush=True)
Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
         learning_rate="auto").fit_transform(F_)
np.savez_compressed(os.path.join(OUT, "strict_main_tsne.npz"), Z=Z, y=Y_)
print("saved t-SNE", flush=True)
