#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPE/LPE Transformer v5.1
- fast/full dual mode:
    fast:  --debug_fast 1 --fast_modes clinical,clinical+plgf,full --fast_epochs 25 --fast_seeds 42,7
    full:  去掉 --debug_fast / --skip_eval，跑全部模态 + 校准/DCA/DeLong/重要性
- speed: EMA eval model built once (was rebuilt every epoch)
- fix:  rank-Gaussian for metabolomics now correctly applied to EV via stored train mapping
"""

import argparse
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as scipy_stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, brier_score_loss, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ============================================================================
# DeLong utilities
# ============================================================================

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[J] = T
    return out


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = _compute_midrank(preds_sorted_transposed[r, :m])
        ty[r, :] = _compute_midrank(preds_sorted_transposed[r, m:])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_auc_ci(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = (-y_true).argsort(kind="mergesort")
    m = int(y_true.sum())
    aucs, cov = _fast_delong(y_score[np.newaxis, order], m)
    a = float(aucs[0])
    var = float(max(cov[0, 0], 1e-12))
    z = scipy_stats.norm.ppf(0.975)
    se = np.sqrt(var)
    return a, a - z * se, a + z * se


def delong_roc_test(y_true, pred1, pred2):
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    order = (-y_true).argsort(kind="mergesort")
    m = int(y_true.sum())
    aucs, cov = _fast_delong(np.vstack((pred1, pred2))[:, order], m)
    l = np.array([[1.0, -1.0]])
    var = float(max((l @ cov @ l.T)[0, 0], 1e-12))
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * scipy_stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def holm_correction(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    idx = np.argsort(p)
    q = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * p[i])
        q[i] = min(run, 1.0)
    return q


def pigf_rename(s):
    return str(s).replace("PLGF", "PIGF")

# ============================================================================
# Args / Config
# ============================================================================

@dataclass
class Args:
    data_path: str
    output_dir: str = "./output_v5"
    num_epochs: int = 220
    seeds: str = "5,10,20,42,7,123,2024,3407"
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    modality_dropout: float = 0.15
    input_dropout: float = 0.10
    focal_gamma: float = 1.5
    pos_weight_cap: float = 4.0
    early_stopping_patience: int = 35
    ema_decay: float = 0.995
    clip_range: float = 4.0
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    rank_metab: int = 1
    prior_clinical: float = 1.0
    prior_metab: float = 0.22
    prior_lab: float = 1.10
    prior_plgf: float = 1.00
    cross_dropout: float = 0.20
    head_dropout: float = 0.30
    temperature: float = 0.025
    keep_margin: float = 0.035
    run_shap: int = 0
    shap_nsamples: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # ---- fast mode ----
    debug_fast: int = 0
    fast_modes: str = "clinical,clinical+plgf,full"
    fast_epochs: int = 25
    fast_seeds: str = "42,7"
    fast_patience: int = 8
    skip_eval: int = 0


class Config:
    CLINICAL_NAMES = ["IVF", "PMH", "Birth_history", "RPL", "age", "BMI", "MAP"]
    METABOLOMICS_NAMES = [
        "2-Hydroxy-3-methylbutyric acid",
        "Alpha-Linolenic acid",
        "Docosapentaenoic acid (DPA)",
        "Homovanillic acid",
        "Hydroxyphenyllactic acid",
        "Isovalerylcarnitine",
        "L-Malic acid",
        "Palmitelaidic acid",
        "Sarcosine",
    ]
    LAB_END_NAME = "UA"

    MODE_SETS = {
        "clinical": ("c",),
        "metab": ("m",),
        "lab": ("l",),
        "clinical+metab": ("c", "m"),
        "clinical+lab": ("c", "l"),
        "metab+lab": ("m", "l"),
        "clinical+plgf": ("c", "p"),
        "full": ("c", "m", "l"),
    }
    MODALITY_ORDER = ("c", "m", "l", "p")

    def __init__(self, args: Args):
        self.args = args
        self.DEVICE = torch.device(args.device)
        self.STD_STATS = None
        self.CLINICAL_COLS = None
        self.METABOLOMICS_COLS = None
        self.LAB_COLS = None
        self.PLGF_COLS = None
        self.N_CLINICAL = 0
        self.N_METAB = 0
        self.N_LAB = 0
        self.N_PLGF = 0

    def detect_columns(self, df):
        cols = list(df.columns)
        n = len(cols)

        def find_one(name):
            for i, c in enumerate(cols):
                if name in str(c) or str(c) == name:
                    return i
            return None

        self.CLINICAL_COLS = [i for i in (find_one(x) for x in self.CLINICAL_NAMES) if i is not None]
        self.METABOLOMICS_COLS = [i for i in (find_one(x) for x in self.METABOLOMICS_NAMES) if i is not None]

        start = None; end = None
        self.PLGF_COLS = []
        for i, c in enumerate(cols):
            cu = str(c).strip().upper()
            if ("PIGF" in cu or "PLGF" in cu) and start is None:
                start = i
                self.PLGF_COLS = [i]
            if str(c).strip() == self.LAB_END_NAME:
                end = i
        self.LAB_COLS = list(range(start, end + 1)) if start is not None and end is not None else []

        if not self.CLINICAL_COLS:
            self.CLINICAL_COLS = list(range(3, min(10, n)))
        if not self.METABOLOMICS_COLS:
            self.METABOLOMICS_COLS = list(range(10, min(19, n)))
        if not self.LAB_COLS:
            s = self.PLGF_COLS[0] if self.PLGF_COLS else 19
            self.LAB_COLS = list(range(s, n))
        if not self.PLGF_COLS:
            self.PLGF_COLS = [self.LAB_COLS[0]]

        if max(self.CLINICAL_COLS[-1], self.METABOLOMICS_COLS[-1], self.LAB_COLS[-1]) >= n:
            raise IndexError(f"Column index out of range: n_cols={n}")

        self.N_CLINICAL = len(self.CLINICAL_COLS)
        self.N_METAB = len(self.METABOLOMICS_COLS)
        self.N_LAB = len(self.LAB_COLS)
        self.N_PLGF = len(self.PLGF_COLS)

        print(f"[detect] cols={n} clinical={self.N_CLINICAL} metab={self.N_METAB} "
              f"lab={self.N_LAB} plgf={self.PLGF_COLS}")

# ============================================================================
# Preprocessing: train-fitted winsorize + robust scaler + optional rank-Gauss
# ============================================================================

def _fit_prep(X, lo, hi, rank_gauss=False):
    X = X.astype(float)
    finite = np.isfinite(X)
    col_med = np.array([np.median(X[finite[:, j], j]) if finite[:, j].any() else 0.0
                        for j in range(X.shape[1])])
    X = np.nan_to_num(X, nan=col_med, posinf=col_med + 1e6, neginf=col_med - 1e6)
    ql = np.quantile(X, lo, axis=0)
    qh = np.quantile(X, hi, axis=0)
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr[iqr == 0] = 1.0
    rank_map = None
    if rank_gauss:
        n = X.shape[0]
        r = scipy_stats.rankdata(X, axis=0)
        Z = scipy_stats.norm.ppf((r - 0.5) / n)
        rank_map = (np.sort(X, axis=0), np.sort(Z, axis=0))
    return {"ql": ql, "qh": qh, "med": med, "iqr": iqr, "rank": rank_map}


def _apply_prep(X, st, clip_range):
    X = X.astype(float)
    med, iqr = st["med"], st["iqr"]
    X = np.nan_to_num(X, nan=med, posinf=med + 10 * iqr, neginf=med - 10 * iqr)
    Xw = np.clip(X, st["ql"], st["qh"])
    if st["rank"] is not None:
        ref, zref = st["rank"]
        n = ref.shape[0]
        out = np.zeros_like(Xw)
        for j in range(Xw.shape[1]):
            pos = np.clip(np.searchsorted(ref[:, j], Xw[:, j]), 0, n - 1)
            out[:, j] = zref[pos, j]
        Z = out
    else:
        Z = (Xw - med) / iqr
    return np.clip(Z, -clip_range, clip_range).astype(np.float32)


class EPEDataset(Dataset):
    def __init__(self, data_path, dataset_type, config: Config):
        a = config.args
        self.df = pd.read_csv(data_path, sep="\t", engine="python")
        if len(self.df.columns) < 10:
            self.df = pd.read_csv(data_path, sep=r"\s+", engine="python")
        self.all_columns = list(self.df.columns)

        low = [str(c).strip().lower() for c in self.all_columns]
        def find_col(cands, default):
            for i, name in enumerate(low):
                if name in cands:
                    return i
            return default

        self.ID_COL = find_col({"id", "sample_id", "sampleid"}, 0)
        self.DATASET_COL = find_col({"dataset", "data_set", "split"}, 1)
        self.GROUP_COL = find_col({"group", "label", "diagnosis", "disease"}, 2)

        if config.STD_STATS is None:
            config.detect_columns(self.df)
        self.config = config

        self.df.iloc[:, self.DATASET_COL] = self.df.iloc[:, self.DATASET_COL].astype(str).str.strip()
        self.df.iloc[:, self.GROUP_COL] = self.df.iloc[:, self.GROUP_COL].astype(str).str.strip()

        gv = set(self.df.iloc[:, self.GROUP_COL].str.upper().values)
        if "EPE" in gv:
            self.disease_type = "EPE"; self.positive_label = "EPE"
        elif "LPE" in gv:
            self.disease_type = "LPE"; self.positive_label = "LPE"
        else:
            raise ValueError(f"Unknown disease type: {sorted(gv)}")

        alias = {
            "training": "Training", "train": "Training",
            "validation": "EV", "valid": "EV", "val": "EV", "ev": "EV",
            "test": "EV", "testing": "EV", "external": "EV", "ext": "EV",
            "internalval": "Training", "internal": "Training",
        }
        dn = self.df.iloc[:, self.DATASET_COL].str.lower().map(lambda x: alias.get(x, x))
        if isinstance(dataset_type, (list, set, tuple)):
            req = {alias.get(str(x).strip().lower(), str(x).strip()) for x in dataset_type}
            mask = dn.isin(list(req)).values
            desc = "+".join(map(str, dataset_type))
        else:
            tgt = "Training" if dataset_type == "InternalVal" else dataset_type
            mask = (dn == tgt).values
            desc = tgt
        if mask.sum() == 0:
            raise ValueError(f"No samples found for dataset_type='{desc}'")
        self.all_ids = self.df.iloc[:, self.ID_COL].values
        self.sample_ids = self.all_ids[mask]
        self.df = self.df[mask].reset_index(drop=True)
        self.labels = (self.df.iloc[:, self.GROUP_COL].str.upper() == self.positive_label).astype(int).values

        Xc = self.df.iloc[:, config.CLINICAL_COLS].values.astype(float)
        Xm = self.df.iloc[:, config.METABOLOMICS_COLS].values.astype(float)
        Xl = self.df.iloc[:, config.LAB_COLS].values.astype(float)
        Xp = self.df.iloc[:, config.PLGF_COLS].values.astype(float)

        is_train_pool = np.any(dn[mask].values == "Training")
        if is_train_pool and config.STD_STATS is None:
            config.STD_STATS = {
                "clinical": _fit_prep(Xc, a.winsor_lower, a.winsor_upper, rank_gauss=False),
                "metab": _fit_prep(Xm, a.winsor_lower, a.winsor_upper, rank_gauss=bool(a.rank_metab)),
                "lab": _fit_prep(Xl, a.winsor_lower, a.winsor_upper, rank_gauss=False),
                "plgf": _fit_prep(Xp, a.winsor_lower, a.winsor_upper, rank_gauss=False),
            }

        if config.STD_STATS is None:
            raise ValueError("Instantiate training-pool dataset first.")

        self.clinical = _apply_prep(Xc, config.STD_STATS["clinical"], a.clip_range)
        self.metabolomics = _apply_prep(Xm, config.STD_STATS["metab"], a.clip_range)
        self.lab = _apply_prep(Xl, config.STD_STATS["lab"], a.clip_range)
        self.plgf = _apply_prep(Xp, config.STD_STATS["plgf"], a.clip_range)

        self.clinical_names = [pigf_rename(self.all_columns[i]) for i in config.CLINICAL_COLS]
        self.metabolomics_names = [pigf_rename(self.all_columns[i]) for i in config.METABOLOMICS_COLS]
        self.lab_names = [pigf_rename(self.all_columns[i]) for i in config.LAB_COLS]
        self.plgf_names = [pigf_rename(self.all_columns[i]) for i in config.PLGF_COLS]
        print(f"{desc}: n={len(self)} disease={self.disease_type} pos={self.labels.mean():.2%}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.clinical[idx], dtype=torch.float32),
            torch.tensor(self.metabolomics[idx], dtype=torch.float32),
            torch.tensor(self.lab[idx], dtype=torch.float32),
            torch.tensor(self.plgf[idx], dtype=torch.float32),
            torch.tensor([self.labels[idx]], dtype=torch.float32),
        )

    def get_sample_ids(self):
        return self.sample_ids.tolist()

# ============================================================================
# Model
# ============================================================================

def make_encoder_layer(d_model, nhead, dropout):
    kwargs = dict(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                  dropout=dropout, activation="gelu")
    try:
        layer = nn.TransformerEncoderLayer(**kwargs, batch_first=True, norm_first=True)
        return layer, True
    except TypeError:
        layer = nn.TransformerEncoderLayer(**kwargs)
        return layer, False


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d_model, heads=4, layers=1, dropout=0.30):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer, self.batch_first = make_encoder_layer(d_model, heads, dropout)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.proj(x).unsqueeze(1)
        cls = self.cls.expand(x.size(0), -1, -1)
        t = torch.cat([cls, z], dim=1)
        if self.batch_first:
            t = self.enc(t)
        else:
            t = self.enc(t.transpose(0, 1)).transpose(0, 1)
        return self.drop(t[:, 0])


class MultiModalEPEModel(nn.Module):
    MODE_SETS = Config.MODE_SETS
    MODALITY_ORDER = Config.MODALITY_ORDER
    KEY_TO_IDX = {"c": 0, "m": 1, "l": 2, "p": 3}

    def __init__(self, config: Config):
        super().__init__()
        a = config.args
        self.config = config
        d = 64
        self.d_model = d

        self.encoders = nn.ModuleDict({
            "c": ModalityEncoder(config.N_CLINICAL, d, 4, 1, a.head_dropout),
            "m": ModalityEncoder(config.N_METAB, d, 4, 1, a.head_dropout),
            "l": ModalityEncoder(config.N_LAB, d, 4, 1, a.head_dropout),
            "p": ModalityEncoder(config.N_PLGF, d, 2, 1, a.head_dropout),
        })
        self.mod_emb = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.prior = nn.Parameter(torch.tensor([
            a.prior_clinical, a.prior_metab, a.prior_lab, a.prior_plgf
        ], dtype=torch.float32))
        self.gate_dyn = nn.ModuleDict({k: nn.Linear(d, 1) for k in self.MODALITY_ORDER})
        self.cross = nn.MultiheadAttention(d, 4, dropout=a.cross_dropout)
        self.cross_norm = nn.LayerNorm(d)
        self.fuse_norm = nn.LayerNorm(d)

        raw_dims = {"c": config.N_CLINICAL, "m": config.N_METAB, "l": config.N_LAB, "p": config.N_PLGF}
        self.mode_raw_dims = {m: sum(raw_dims[k] for k in use) for m, use in self.MODE_SETS.items()}
        self.heads = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(d * 2 + self.mode_raw_dims[m], 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(a.head_dropout),
                nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(a.head_dropout),
                nn.Linear(64, 1),
            ) for m in self.MODE_SETS.keys()
        })

    def _encode(self, x, k):
        return self.encoders[k](x)

    def forward(self, xc, xm, xl, xp, mode="full", return_gate=False):
        if mode not in self.MODE_SETS:
            raise ValueError(mode)
        use = self.MODE_SETS[mode]
        xdict = {"c": xc, "m": xm, "l": xl, "p": xp}

        drop_key = None
        if self.training and len(use) >= 2 and torch.rand(1).item() < self.config.args.modality_dropout:
            drop_key = use[int(torch.randint(len(use), (1,)).item())]

        inp_drop = self.config.args.input_dropout if self.training else 0.0
        feats = {}; raws = []; keys = []
        for k in use:
            x = xdict[k]
            if inp_drop > 0:
                mask = (torch.rand_like(x) > inp_drop).float() / (1.0 - inp_drop)
                x = x * mask
            raws.append(torch.zeros_like(x) if k == drop_key else x)
            if k == drop_key:
                feats[k] = torch.zeros(x.size(0), self.d_model, device=x.device)
            else:
                feats[k] = self._encode(x, k)
            keys.append(k)

        idx = torch.tensor([self.KEY_TO_IDX[k] for k in keys], device=xc.device)
        prior = torch.sigmoid(self.prior[idx]).clamp(1e-3, 1 - 1e-3)
        logits = torch.cat([self.gate_dyn[k](feats[k]) for k in keys], dim=1)
        logits = logits + torch.log(prior / (1 - prior)).unsqueeze(0)
        if drop_key is not None:
            logits[:, keys.index(drop_key)] = float("-inf")
        w = torch.softmax(logits, dim=1)

        stack = torch.stack([feats[k] + self.mod_emb[0, self.KEY_TO_IDX[k]] for k in keys], dim=1)
        s = stack.transpose(0, 1)
        attn_out, _ = self.cross(s, s, s, need_weights=False)
        attn_out = attn_out.transpose(0, 1)
        cross_fused = (w.unsqueeze(-1) * self.cross_norm(attn_out)).sum(dim=1)
        gated = (w.unsqueeze(-1) * torch.stack([feats[k] for k in keys], dim=1)).sum(dim=1)
        fused = self.fuse_norm(gated + 0.5 * cross_fused)
        head_in = torch.cat([fused, cross_fused, torch.cat(raws, dim=1)], dim=1)

        expected = self.d_model * 2 + self.mode_raw_dims[mode]
        if head_in.size(1) != expected:
            raise RuntimeError(f"dim mismatch mode={mode} got={head_in.size(1)} expect={expected}")
        out = self.heads[mode](head_in).squeeze(1)
        if return_gate:
            return out, w, keys
        return out

    def predict_proba(self, *xs, mode="full"):
        return torch.sigmoid(self.forward(*xs, mode=mode))


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

# ============================================================================
# Train / validate
# ============================================================================

def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(y, p)
    return float(auc(fpr, tpr))


def make_criterion(args, pos_weight):
    eps = args.label_smoothing
    gamma = args.focal_gamma

    def crit(logits, y):
        y_smooth = y * (1 - eps) + 0.5 * eps if eps > 0 else y
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y_smooth, pos_weight=pos_weight, reduction="none")
        if gamma > 0:
            p = torch.sigmoid(logits)
            pt = p * y + (1 - p) * (1 - y)
            loss = loss * (1 - pt).pow(gamma)
        return loss.mean()
    return crit


def run_epoch(model, loader, criterion, optimizer, device, mode, train=True, ema=None):
    model.train() if train else model.eval()
    tot = 0.0; preds = []; labels = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xc, xm, xl, xp, y in loader:
            xc, xm, xl, xp, y = [t.to(device) for t in (xc, xm, xl, xp, y)]
            y = y.squeeze(1)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xc, xm, xl, xp, mode=mode)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            tot += loss.item() * y.size(0)
            preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
    return tot / max(1, len(loader.dataset)), preds, labels


@torch.no_grad()
def predict_loader(model, loader, device, mode):
    model.eval()
    preds, labels = [], []
    for xc, xm, xl, xp, y in loader:
        xc, xm, xl, xp = [t.to(device) for t in (xc, xm, xl, xp)]
        logits = model(xc, xm, xl, xp, mode=mode)
        preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(y.squeeze(1).cpu().numpy().tolist())
    return np.array(preds), np.array(labels)


def stratified_split(labels, seed, val_frac=0.2):
    rng = np.random.RandomState(seed)
    y = np.asarray(labels).astype(int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    npos = int(round(val_frac * len(pos))); nneg = int(round(val_frac * len(neg)))
    val = np.concatenate([pos[:npos], neg[:nneg]])
    tr = np.concatenate([pos[npos:], neg[nneg:]])
    return tr, val


def train_one_seed(config, train_sub, internal_loader, val_loader, mode, pos_weight, seed):
    """Train a single seed; returns dict with EMA-selected model and metrics."""
    a = config.args
    torch.manual_seed(seed); np.random.seed(seed)
    criterion = make_criterion(a, pos_weight)
    model = MultiModalEPEModel(config).to(config.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)
    ema = EMA(model, decay=a.ema_decay)
    ema_model = MultiModalEPEModel(config).to(config.DEVICE)   # build once

    best_auc = -1.0; best_state = None; best_epoch = -1; patience = 0
    train_loader = DataLoader(train_sub, batch_size=a.batch_size, shuffle=True, drop_last=False)

    for ep in range(a.num_epochs):
        run_epoch(model, train_loader, criterion, opt, config.DEVICE, mode, train=True, ema=ema)
        ema.copy_to(ema_model)
        _, ip, il = run_epoch(ema_model, internal_loader, criterion, None, config.DEVICE, mode, train=False)
        ia = safe_auc(il, ip)
        sched.step(ia)
        if ia > best_auc:
            best_auc = ia; best_epoch = ep; patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        else:
            patience += 1
            if patience >= a.early_stopping_patience:
                print(f"  seed {seed} early stop @ {ep+1} best_intAUC={best_auc:.4f}")
                break

    final_model = MultiModalEPEModel(config).to(config.DEVICE)
    if best_state is None:
        ema.copy_to(final_model); best_auc = 0.5; best_epoch = a.num_epochs - 1
    else:
        final_model.load_state_dict(best_state)

    _, ip, il = run_epoch(final_model, internal_loader, criterion, None, config.DEVICE, mode, train=False)
    int_auc = safe_auc(il, ip)
    evp, evl = predict_loader(final_model, val_loader, config.DEVICE, mode)
    ev_auc = safe_auc(evl, evp)
    print(f"  seed {seed}: intAUC={int_auc:.4f} evAUC={ev_auc:.4f} epoch={best_epoch+1}")
    return {"model": final_model, "int_auc": int_auc, "ev_auc": ev_auc,
            "seed": seed, "best_epoch": best_epoch, "ev_probs": evp, "ev_labels": evl}

# ============================================================================
# Metrics / plots
# ============================================================================

def evaluate_calibration(y_true, y_prob, name, out, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    try:
        fop, mpv = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    except Exception:
        fop, mpv = calibration_curve(y_true, y_prob, n_bins=max(2, len(y_true)//10), strategy="uniform")
    if len(fop) > 1:
        X = np.vstack([mpv, np.ones(len(mpv))]).T
        slope, intercept = np.linalg.lstsq(X, fop, rcond=None)[0]
    else:
        slope, intercept = 1.0, 0.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].plot(mpv, fop, "s-")
    ax[0].set_title(f"{name}\nBrier={brier:.4f}")
    ax[0].set_xlabel("Mean predicted"); ax[0].set_ylabel("Fraction positives")
    ax[1].hist(y_prob[y_true == 0], bins=20, alpha=.5, label="control")
    ax[1].hist(y_prob[y_true == 1], bins=20, alpha=.5, label="case")
    ax[1].legend(); ax[1].set_title("Probability distribution")
    path = os.path.join(out, f"Calibration_{name}.png")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return {"brier": float(brier), "slope": float(slope), "intercept": float(intercept), "path": path}


def decision_curve(y_true, y_prob, name, out, thresholds=np.arange(0.01, 0.5, 0.01)):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true); prev = y_true.mean()
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        pp = (y_prob >= t).astype(int)
        tp = np.sum((pp == 1) & (y_true == 1)); fp = np.sum((pp == 1) & (y_true == 0))
        nb_model.append(tp/n - fp/n * (t/(1-t)))
        nb_all.append(prev - (1-prev) * (t/(1-t)))
        nb_none.append(0.0)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_model, label="model")
    plt.plot(thresholds, nb_all, "r--", label="treat all")
    plt.plot(thresholds, nb_none, "k:", label="treat none")
    plt.xlabel("Threshold"); plt.ylabel("Net benefit"); plt.title(f"DCA {name}")
    plt.legend(); plt.grid(alpha=.3)
    path = os.path.join(out, f"DCA_{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    i = int(np.argmax(nb_model))
    return {"max_nb": float(nb_model[i]), "thr": float(thresholds[i]), "path": path}


def plot_roc(results, out, disease):
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3", "#F38181", "#AA96DA", "#9B59B6", "#2E86AB"]
    plt.figure(figsize=(11, 8))
    for i, (name, d) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(d["labels"], d["preds"])
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                 label=f"{name} AUC={safe_auc(d['labels'], d['preds']):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, .5)); plt.grid(alpha=.3)
    tag = "FAST_" if os.path.basename(out).startswith("fast") else ""
    path = os.path.join(out, f"{tag}ROC_Curves_{disease}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    return path


def blocks_names(k, dataset):
    return {"c": dataset.clinical_names, "m": dataset.metabolomics_names,
            "l": dataset.lab_names, "p": dataset.plgf_names}[k]


def gradient_importance(model, dataset, mode, device, n=64):
    model.eval()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(dataset), min(n, len(dataset)), replace=False)
    use = MultiModalEPEModel.MODE_SETS[mode]
    chosen_names = [nm for k in ["c", "m", "l", "p"] if k in use for nm in blocks_names(k, dataset)]
    xc = torch.tensor(dataset.clinical[idx], dtype=torch.float32, device=device, requires_grad=True)
    xm = torch.tensor(dataset.metabolomics[idx], dtype=torch.float32, device=device, requires_grad=True)
    xl = torch.tensor(dataset.lab[idx], dtype=torch.float32, device=device, requires_grad=True)
    xp = torch.tensor(dataset.plgf[idx], dtype=torch.float32, device=device, requires_grad=True)
    logits = model(xc, xm, xl, xp, mode=mode)
    torch.sigmoid(logits).sum().backward()
    grads = []
    if "c" in use: grads.append(xc.grad.abs().mean(0).cpu().numpy())
    if "m" in use: grads.append(xm.grad.abs().mean(0).cpu().numpy())
    if "l" in use: grads.append(xl.grad.abs().mean(0).cpu().numpy())
    if "p" in use: grads.append(xp.grad.abs().mean(0).cpu().numpy())
    gi = np.concatenate(grads)
    df = pd.DataFrame({"Feature": chosen_names,
                       "GradImportance": gi[:len(chosen_names)]}).sort_values("GradImportance", ascending=False)
    return df

# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="./output_v5")
    p.add_argument("--num_epochs", type=int, default=220)
    p.add_argument("--seeds", default="5,10,20,42,7,123,2024,3407")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--modality_dropout", type=float, default=0.15)
    p.add_argument("--input_dropout", type=float, default=0.10)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--pos_weight_cap", type=float, default=4.0)
    p.add_argument("--early_stopping_patience", type=int, default=35)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--clip_range", type=float, default=4.0)
    p.add_argument("--winsor_lower", type=float, default=0.01)
    p.add_argument("--winsor_upper", type=float, default=0.99)
    p.add_argument("--rank_metab", type=int, default=1)
    p.add_argument("--prior_clinical", type=float, default=1.0)
    p.add_argument("--prior_metab", type=float, default=0.22)
    p.add_argument("--prior_lab", type=float, default=1.10)
    p.add_argument("--prior_plgf", type=float, default=1.0)
    p.add_argument("--cross_dropout", type=float, default=0.20)
    p.add_argument("--head_dropout", type=float, default=0.30)
    p.add_argument("--temperature", type=float, default=0.025)
    p.add_argument("--keep_margin", type=float, default=0.035)
    p.add_argument("--run_shap", action="store_true")
    p.add_argument("--shap_nsamples", type=int, default=200)
    # fast mode
    p.add_argument("--debug_fast", type=int, default=0)
    p.add_argument("--fast_modes", default="clinical,clinical+plgf,full")
    p.add_argument("--fast_epochs", type=int, default=25)
    p.add_argument("--fast_seeds", default="42,7")
    p.add_argument("--fast_patience", type=int, default=8)
    p.add_argument("--skip_eval", type=int, default=0)
    return Args(**vars(p.parse_args()))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    config = Config(args)

    fast_mode = bool(args.debug_fast) or bool(args.skip_eval)

    print("=" * 70)
    print("EPE/LPE Transformer v5.1 | reliability-prior gate + cross-attn + EMA")
    print("=" * 70)

    if bool(args.debug_fast):
        args.seeds = args.fast_seeds
        args.num_epochs = args.fast_epochs
        args.early_stopping_patience = args.fast_patience
        print(f"[FAST] seeds={args.seeds} epochs={args.num_epochs} patience={args.early_stopping_patience}")
        print("[FAST] 仅用于快速判断方向，不作为最终结论。")

    print(f"device={config.DEVICE} seeds={args.seeds} prior_metab={args.prior_metab}")

    train_pool = EPEDataset(args.data_path, {"Training", "InternalVal"}, config)
    val_ds = EPEDataset(args.data_path, "EV", config)
    disease = train_pool.disease_type
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    modes = {
        "clinical": "Clinical Factors",
        "metab": "Metabolomics",
        "lab": "Lab variables+PIGF",
        "clinical+metab": "Clinical Factors+Metabolomics",
        "clinical+lab": "Clinical Factors+Lab variables+PIGF",
        "metab+lab": "Metabolomics+Lab variables+PIGF",
        "clinical+plgf": "Clinical Factors+PIGF",
        "full": "Multi-Modal Fusion",
    }
    mode_names = {
        "clinical": "Clinical-Factors", "metab": "Metabolomics", "lab": "Lab-data-PIGF",
        "clinical+metab": "Clinical-Factors-Metabolomics", "clinical+lab": "Clinical-Factors-Lab-PIGF",
        "metab+lab": "Metabolomics-Lab-PIGF", "clinical+plgf": "Clinical-Factors-PIGF",
        "full": "Multi-Modal-Fusion",
    }

    if bool(args.debug_fast):
        keep = set(x.strip() for x in args.fast_modes.split(",") if x.strip())
        modes = {k: v for k, v in modes.items() if k in keep}
        print(f"[FAST] modes={list(modes.keys())}")

    y = np.asarray(train_pool.labels).astype(int)
    pos_weight = torch.tensor([min((len(y) - y.sum()) / max(y.sum(), 1), args.pos_weight_cap)],
                              dtype=torch.float32, device=config.DEVICE)

    results = {}; calibration_rows = []; best_models = {}; seed_stats = {}
    for mode, disp in modes.items():
        print("\n" + "=" * 70)
        print(f"TRAIN {disp} | mode={mode}")
        print("=" * 70)
        entries = []
        for seed in [int(s) for s in str(args.seeds).split(",") if str(s).strip()]:
            tr_idx, val_idx = stratified_split(y, seed=seed, val_frac=0.2)
            train_sub = Subset(train_pool, tr_idx.tolist())
            internal_loader = DataLoader(Subset(train_pool, val_idx.tolist()),
                                         batch_size=args.batch_size, shuffle=False)
            entries.append(train_one_seed(config, train_sub, internal_loader, val_loader,
                                          mode, pos_weight, seed))

        ev_labels = entries[0]["ev_labels"]
        ints = np.array([e["int_auc"] for e in entries])
        mx = ints.max()
        keep_mask = ints >= (mx - args.keep_margin)
        if keep_mask.sum() == 0:
            keep_mask = np.ones_like(ints, dtype=bool)
        w = np.exp((ints[keep_mask] - mx) / max(args.temperature, 1e-6)); w /= w.sum()
        ens = np.stack([entries[i]["ev_probs"] for i in np.where(keep_mask)[0]], axis=0)
        ens = (ens * w[:, None]).sum(axis=0)
        auc_ens = safe_auc(ev_labels, ens)
        results[disp] = {"preds": ens.tolist(), "labels": ev_labels.astype(int).tolist(), "auc": auc_ens}
        seed_stats[disp] = (float(np.mean([e["ev_auc"] for e in entries])),
                            float(np.std([e["ev_auc"] for e in entries])))
        best_idx = int(np.argmax(ints))
        best_models[mode] = entries[best_idx]["model"]
        print(f"{disp}: ensemble AUC={auc_ens:.4f} | per-seed {seed_stats[disp][0]:.4f}±"
              f"{seed_stats[disp][1]:.4f} | best int={ints.max():.4f} keep={keep_mask.sum()}/{len(entries)}")

        pred_df = pd.DataFrame({
            "Sample_ID": val_ds.get_sample_ids(),
            "Group": [disease if v == 1 else "control" for v in ev_labels],
            "Prediction_Score": ens, "Model": mode_names[mode],
            "Dataset": "EV", "Disease_Type": disease,
        })
        if fast_mode:
            calibration_rows.append({
                "Model": disp, "AUC": round(auc_ens, 4),
                "PerSeedMean": round(seed_stats[disp][0], 4),
                "PerSeedSD": round(seed_stats[disp][1], 4),
                "Brier": np.nan, "Slope": np.nan, "Intercept": np.nan,
                "MaxNetBenefit": np.nan, "OptimalThreshold": np.nan,
                "Note": "fast_mode_no_calibration_dca",
            })
            pred_df.to_csv(os.path.join(args.output_dir, f"FAST_{mode_names[mode]}_Validation_{disease}.txt"),
                           sep="\t", index=False)
        else:
            cal = evaluate_calibration(ev_labels, ens, mode_names[mode], args.output_dir)
            dca = decision_curve(ev_labels, ens, mode_names[mode], args.output_dir)
            calibration_rows.append({
                "Model": disp, "AUC": round(auc_ens, 4),
                "PerSeedMean": round(seed_stats[disp][0], 4),
                "PerSeedSD": round(seed_stats[disp][1], 4),
                "Brier": round(cal["brier"], 4),
                "Slope": round(cal["slope"], 3), "Intercept": round(cal["intercept"], 3),
                "MaxNetBenefit": round(dca["max_nb"], 4), "OptimalThreshold": round(dca["thr"], 3),
            })
            pred_df.to_csv(os.path.join(args.output_dir,
                                        f"{mode_names[mode]}_Validation_{disease}_Results_v5.txt"),
                           sep="\t", index=False)

    print("\nSummary EV ensemble:")
    for name, d in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        print(f"  {name}: {d['auc']:.4f} ({seed_stats[name][0]:.4f}±{seed_stats[name][1]:.4f})")

    pd.DataFrame(calibration_rows).to_csv(
        os.path.join(args.output_dir, f"Calibration_DCA_Summary_{disease}_v5.txt"), sep="\t", index=False)
    roc_path = plot_roc(results, args.output_dir, disease)

    if not fast_mode:
        rows = []; pvals = []
        names = list(results.keys())
        for nm in names:
            a, lo, hi = delong_auc_ci(results[nm]["labels"], results[nm]["preds"])
            rows.append({"Model": nm, "AUC": round(a, 3), "CI": f"{lo:.3f}-{hi:.3f}"})
        pair_rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a1, a2, pv = delong_roc_test(results[names[i]]["labels"],
                                             results[names[i]]["preds"],
                                             results[names[j]]["preds"])
                pair_rows.append({"Reference": names[i], "Model": names[j],
                                  "dAUC": round(a1 - a2, 3), "P": pv})
                pvals.append(pv)
        q = holm_correction(pvals)
        for r, qq in zip(pair_rows, q):
            r["Holm_q"] = round(float(qq), 4)
        try:
            with pd.ExcelWriter(os.path.join(args.output_dir,
                                             f"Model_AUC_Pairwise_DeLong_{disease}_v5.xlsx"),
                                engine="openpyxl") as wrt:
                pd.DataFrame(rows).to_excel(wrt, sheet_name="AUC_CI", index=False)
                pd.DataFrame(pair_rows).to_excel(wrt, sheet_name="Pairwise", index=False)
                pd.DataFrame(calibration_rows).to_excel(wrt, sheet_name="Calibration_DCA", index=False)
        except Exception as ex:
            pd.DataFrame(pair_rows).to_csv(
                os.path.join(args.output_dir, f"Pairwise_{disease}_v5.csv"), index=False)
            print(f"Excel failed: {ex}")

        imp_dir = os.path.join(args.output_dir, "gradient_importance_v5")
        os.makedirs(imp_dir, exist_ok=True)
        for mode in modes.keys():
            try:
                df = gradient_importance(best_models[mode], train_pool, mode, config.DEVICE, n=64)
                df.to_csv(os.path.join(imp_dir, f"GradImportance_{mode_names[mode]}_{disease}.txt"),
                          sep="\t", index=False)
            except Exception as ex:
                print(f"importance failed {mode}: {ex}")

    print("\nSaved:", roc_path)
    print("Done.")


if __name__ == "__main__":
    main()
