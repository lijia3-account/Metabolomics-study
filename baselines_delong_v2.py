#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reviewer3_baselines_delong_v3.py  (v3, EPE/LPE 通用)
=====================================================
审稿回复分析: 与 v2 相比的修改:
  (1) 剔除 ridge-LR[Clin+PIGF-8] 和 GBM[Clin+PIGF-8] 两个基线模型,
      所有分析仅保留 FULL-62 特征集下的 ridge-LR 与 GBM;
  (2) 所有模型(多模态融合 + Clinical+lab + 2 个基线, 共 4 个)两两
      DeLong 检验, Holm 校正在全部 6 次比较内进行;
  (3) 输出表格式与主流程 Model_AUC_Pairwise_DeLong_*.csv 一致:
      Reference, Ref_AUC, Model, AUC, AUC_95CI, dAUC, P, Holm_q
      按 AUC 降序排列;
  (4) 重写评分文件解析: 不再依赖表头行检测, 跳过所有表头行后按
      Model 列分组拆块; 兼容第二个区块缺失表头/BOM/大小写差异的情况.

包含模型:
  (1) Multi-Modal-Fusion   (主参照, 评分文件)
  (2) Clinical+lab         (评分文件中的另一 Transformer 模态)
  (3) ridge-LR [FULL-62]   (全 62 特征, 惩罚 logistic 回归, C 由 10-fold CV 调优)
  (4) GBM      [FULL-62]   (直方图梯度提升, early stopping)

评分文件兼容两种格式:
  (a) 双区块粘贴格式 (EPE_model_scores.txt / LPE_model_scores.txt): 按表头行切分;
  (b) Transformer 主流程 V2 输出 (*_Results_V2.txt).

用法 (EPE):
  python reviewer3_baselines_delong_v3.py \
      --data_file NewEPE_control_all_data.txt --score_file EPE_model_scores.txt \
      --pos_class EPE --out_prefix reviewer3_EPE
用法 (LPE):
  python reviewer3_baselines_delong_v3.py \
      --data_file NewLPE_control_all_data.txt --score_file LPE_model_scores.txt \
      --pos_class LPE --out_prefix reviewer3_LPE

输出:
  <out_prefix>_delong.csv     4 个模型两两 DeLong 检验表 (Holm 校正, 6 次比较)
  <out_prefix>_EV_scores.csv  每个 EV 样本 x 每个模型的评分宽表
  <out_prefix>_ROC.png        EV 集 ROC 曲线 (参照=黑色粗线, Clinical+lab=红色粗线, 基线细彩线)
"""

import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from io import StringIO
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ap = argparse.ArgumentParser()
ap.add_argument('--data_file', required=True)
ap.add_argument('--score_file', required=True)
ap.add_argument('--pos_class', default='EPE')
ap.add_argument('--ref_model', default=None,
                help="主参照模型名 (默认: 'Multi-Modal-Fusion', 无则取评分文件首个模型)")
ap.add_argument('--out_prefix', default='reviewer3_out')
args = ap.parse_args()

CLIN7 = ['IVF', 'PMH', 'Birth_history', 'RPL', 'age', 'BMI', 'MAP']
METAB_NAMES = ['2-Hydroxy-3-methylbutyric acid', 'Alpha-Linolenic acid',
               'Docosapentaenoic acid (DPA)', 'Homovanillic acid',
               'Hydroxyphenyllactic acid', 'Isovalerylcarnitine',
               'L-Malic acid', 'Palmitelaidic acid', 'Sarcosine']
LAB_START, LAB_END = 'PlGF', 'UA'

d = pd.read_csv(args.data_file, sep='\t')
d.columns = [c.replace('/', '.') for c in d.columns]
assert {'ID', 'DATASET', 'Group'}.issubset(d.columns), "数据文件缺少 ID/DATASET/Group 列"
d = d[d.Group.isin(['control', args.pos_class])].reset_index(drop=True)
tr = d[d.DATASET == 'Training'].reset_index(drop=True)
ev = d[d.DATASET == 'EV'].reset_index(drop=True)
ytr = (tr.Group == args.pos_class).astype(int).values
yev = (ev.Group == args.pos_class).astype(int).values
print(f"Training: n={len(tr)} (阳性={ytr.sum()}) | EV: n={len(ev)} (阳性={yev.sum()})")

def build_feature_sets():
    cols = list(d.columns)
    clin7 = CLIN7
    metab = [c for c in cols if c in METAB_NAMES]
    lab = cols[cols.index(LAB_START): cols.index(LAB_END) + 1]
    # v3: 已剔除 Clin+PIGF-8, 仅保留 FULL-62
    return {'FULL-62': clin7 + metab + lab}

def parse_score_file(path):
    """健壮版评分文件解析: 跳过所有表头行 -> 正文合并 -> 按 Model 列分组.
    兼容: 双区块粘贴(表头可有可无/带BOM/大小写差异) 与 V2 单模型输出."""
    raw = open(path, encoding='utf-8-sig').read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')          # 统一换行符
    all_lines = [l.lstrip('\ufeff') for l in raw.split('\n') if l.strip()]
    # 所有"表头行"(首字段为 sample_id / id 的行)仅取第一个作为列名, 其余跳过
    hdr_idx = [i for i, l in enumerate(all_lines)
               if l.split('\t')[0].strip().lower() in ('sample_id', 'id')]
    if not hdr_idx:
        hdr_idx = [0]
    header = all_lines[hdr_idx[0]]
    skip = set(hdr_idx)
    body = [l for i, l in enumerate(all_lines) if i not in skip]
    df = pd.read_csv(StringIO('\n'.join([header] + body)), sep='\t')
    df.columns = [str(c).strip() for c in df.columns]
    idc = 'Sample_ID' if 'Sample_ID' in df.columns else 'ID'
    dcol = 'Dataset' if 'Dataset' in df.columns else ('DATASET' if 'DATASET' in df.columns else None)
    if dcol:
        df = df[df[dcol].astype(str).str.strip() == 'EV']
    if 'Model' in df.columns and df['Model'].notna().any():
        sc = 'Prediction_Score' if 'Prediction_Score' in df.columns else \
             [c for c in df.columns if c not in (idc, dcol, 'Group', 'Model')
              and not c.startswith('Unnamed')][0]
        out = {}
        for nm, g in df.groupby(df['Model'].astype(str).str.strip()):
            g = g[[idc, 'Group', sc]].rename(columns={idc: 'Sample_ID', sc: 'score'})
            g = g.drop_duplicates(subset='Sample_ID', keep='first')   # 防重复样本
            out[nm] = g.reset_index(drop=True)
        return out
    # 回退: 无 Model 列的单模型文件 (V2 输出等), 名称取唯一值列或文件名
    sc = 'Prediction_Score' if 'Prediction_Score' in df.columns else \
         [c for c in df.columns if c not in (idc, dcol, 'Group', 'Model')
          and not c.startswith('Unnamed')][0]
    nm = args.score_file
    for c in reversed(list(df.columns)):
        vals = df[c].dropna().astype(str).unique()
        if len(vals) == 1 and vals[0] not in (args.pos_class, 'control', 'EPE', 'LPE'):
            nm = vals[0]
            break
    return {nm: df[[idc, 'Group', sc]].rename(columns={idc: 'Sample_ID', sc: 'score'})}

score_blocks = parse_score_file(args.score_file)
print("评分文件解析到模型:", list(score_blocks))

def align(block):
    if set(block.Sample_ID) == set(ev.ID):
        m = ev[['ID', 'Group']].merge(block, left_on='ID', right_on='Sample_ID')
        assert (m.Group_x == m.Group_y).all(), "ID 合并后 Group 不一致"
        assert len(m) == len(ev), f"ID 合并后行数 {len(m)} != EV 样本数 {len(ev)}, 评分文件存在重复/多余样本"
        return m.score.values, 'ID 合并'
    assert len(block) == len(ev), "EV 样本数不一致且 ID 无法匹配"
    b = block.reset_index(drop=True)
    assert (b.Group.values == ev.Group.values).all(), "顺序配对失败: Group 序列不一致"
    return b.score.values, '顺序配对(Group已校验)'

model_scores, how = {}, None
for nm, blk in score_blocks.items():
    s, how = align(blk)
    assert np.std(s) > 1e-6, f"{nm} 评分为常数"
    model_scores[nm] = s
print(f"评分对齐方式: {how}")

disp = lambda nm: str(nm).replace('PLGF', 'PlGF')   # 显示名称: PLGF -> PIGF

# ---- 主参照选择: 其余评分模型自动成为对照 ----
PREF_ORDER = ['Multi-Modal-Fusion', 'Clinical-Factors-Lab-PlGF']   # EPE融合模型 -> LPE最佳模型
if args.ref_model:
    assert args.ref_model in model_scores, f"--ref_model '{args.ref_model}' 不在评分文件中"
    primary = args.ref_model
else:
    primary = next((m for m in PREF_ORDER if m in model_scores), list(model_scores)[0])
    if primary not in PREF_ORDER:
        print(f"提示: 未识别到预设最佳模型, 主参照自动设为首个模型 '{primary}'")
ref_s = model_scores.pop(primary)          # 主参照
extra_cmps = dict(model_scores)            # 例如 Clinical+lab
print(f"参照模型: {disp(primary)}")

# ---- DeLong 工具 ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def _fastdelong(p, m):
    n = p.shape[1] - m; k = p.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(p[r, :m]); ty[r, :] = _midrank(p[r, m:]); tz[r, :] = _midrank(p[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    return aucs, np.cov((tz[:, :m] - tx) / n) / m + np.cov(1 - (tz[:, m:] - ty) / m) / n

_ORDER = np.r_[np.where(yev == 1)[0], np.where(yev == 0)[0]]
_M = int(yev.sum())

def delong_ci(score):
    a, cov = _fastdelong(np.asarray(score, float)[_ORDER][None, :], _M)
    se = np.sqrt(np.asarray(cov).ravel()[0])
    return float(a[0]), float(a[0] - 1.96 * se), float(a[0] + 1.96 * se)

def delong_test(s1, s2):
    p = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])[:, _ORDER]
    a, cov = _fastdelong(p, _M)
    z = abs(a[0] - a[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return 2 * (1 - stats.norm.cdf(z)), float(z)

def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); q = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx]); q[idx] = min(1.0, run)
    return q

# ---- 训练基线 (仅 FULL-62; 已剔除 Clin+PIGF-8 的 ridge-LR/GBM) ----
FEATSETS = build_feature_sets()

score_table = pd.DataFrame({'Sample_ID': ev.ID.values, 'Group': ev.Group.values})
score_table[primary] = ref_s

comparators = []   # (名称, 评分, AUC, CI下限, CI上限, 来源)
for nm, s in extra_cmps.items():
    a, lo, hi = delong_ci(s)
    score_table[nm] = s
    comparators.append((nm, s, a, lo, hi, 'Transformer 模态'))
    print(f"[对照] {disp(nm):24s} AUC={a:.3f} ({lo:.3f}-{hi:.3f})")

for fs_name, feats in FEATSETS.items():
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    Xtr, Xev = ((tr[feats] - mu) / sd).values, ((ev[feats] - mu) / sd).values
    for mname, model in {
        'ridge-LR': LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), penalty='l2',
                                         solver='liblinear', scoring='roc_auc', cv=10),
        'GBM': HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                              validation_fraction=0.15, random_state=42),
    }.items():
        bname = f"{mname}[{fs_name}]"
        model.fit(Xtr, ytr)
        sc = model.predict_proba(Xev)[:, 1]
        a, lo, hi = delong_ci(sc)
        score_table[bname] = sc
        comparators.append((bname, sc, a, lo, hi, '传统基线'))
        print(f"[基线] {bname:24s} AUC={a:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

# 调整顺序: 基线在前, Transformer 模态(Clinical+lab)最后
comparators.sort(key=lambda x: x[5] != '传统基线')

# ---- v3: 所有模型两两 DeLong 检验 (Holm 在全部比较内校正), 输出格式同附件4 ----
entries = [{'name': disp(primary), 'score': ref_s}]
for nm, s, a, lo, hi, src in comparators:
    entries.append({'name': disp(nm), 'score': s})
for e in entries:
    a, lo, hi = delong_ci(e['score'])
    e['auc'], e['lo'], e['hi'] = a, lo, hi

# 按 AUC 降序排列 (与附件4一致)
entries.sort(key=lambda e: -e['auc'])

rows, pvals = [], []
for i in range(len(entries)):
    for j in range(i + 1, len(entries)):
        e1, e2 = entries[i], entries[j]
        p, z = delong_test(e1['score'], e2['score'])
        rows.append(dict(Reference=e1['name'], Ref_AUC=round(e1['auc'], 3),
                         Model=e2['name'], AUC=round(e2['auc'], 3),
                         AUC_95CI=f"({e2['lo']:.3f}-{e2['hi']:.3f})",
                         dAUC=round(e1['auc'] - e2['auc'], 3), P=p))
        pvals.append(p)
qs = holm(pvals)
for r, q in zip(rows, qs):
    r['Holm_q'] = round(q, 4)

res = pd.DataFrame(rows, columns=['Reference', 'Ref_AUC', 'Model', 'AUC',
                                  'AUC_95CI', 'dAUC', 'P', 'Holm_q'])
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 30)
print(f"\n== 所有模型两两 DeLong 比较 (EV 集, Holm 校正, {len(res)} 次比较) ==")
print(res.to_string(index=False))
res.to_csv(f"{args.out_prefix}_delong.csv", index=False)
print(f"DeLong 两两比较表已保存: {args.out_prefix}_delong.csv")

# ---- ROC 曲线 ----
auc_ref = roc_auc_score(yev, ref_s)
fig, ax = plt.subplots(figsize=(9.5, 9.5))
ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1)
fpr, tpr, _ = roc_curve(yev, ref_s)
ax.plot(fpr, tpr, lw=2.8, c='k', label=f"{disp(primary)} (AUC = {auc_ref:.3f})")
palette = plt.cm.tab10.colors
ib = 0
for nm, s, a, lo, hi, src in comparators:
    fpr, tpr, _ = roc_curve(yev, s)
    if src == 'Transformer 模态':
        ax.plot(fpr, tpr, lw=2.4, c='r', label=f"{disp(nm)} (AUC = {a:.3f})")
    else:
        ax.plot(fpr, tpr, lw=1.4, c=palette[ib % 10], alpha=0.9, label=f"{disp(nm)} (AUC = {a:.3f})")
        ib += 1
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.tick_params(axis='both', labelsize=15)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=20)
ax.set_title(f'EV cohort ROC - {args.pos_class}', fontsize=21)
ax.legend(loc='lower right', fontsize=13, framealpha=0.9)
ax.grid(alpha=0.3)
fig.savefig(f"{args.out_prefix}_ROC.png", dpi=300, bbox_inches='tight')
print(f"\nROC 图已保存: {args.out_prefix}_ROC.png")

score_table.columns = [disp(c) for c in score_table.columns]
score_table.to_csv(f"{args.out_prefix}_EV_scores.csv", index=False)
print(f"逐样本 EV 评分已保存: {args.out_prefix}_EV_scores.csv  ({score_table.shape[1]-2} 个模型)")
