# Metabolomics-study

Code for **"Development and validation of multi-modal Transformer models for preeclampsia using clinical, metabolomic, and laboratory data"**.

This repository contains the analysis code for training and externally validating multi-modal Transformer models for early-onset preeclampsia (EPE) and late-onset preeclampsia (LPE) prediction, together with conventional machine-learning baselines and pairwise DeLong comparisons.

## Repository structure

| File | Description |
|---|---|
| `epe_transformer_final.py` | Multi-modal Transformer framework (v5.1): 8 modality-combination models, 8-seed temperature-softmax ensemble, calibration, decision curve analysis, DeLong tests, ROC plotting, gradient-based feature importance. |
| `baselines_delong_v2.py` | Conventional baselines (ridge logistic regression and histogram gradient boosting, FULL-62 feature set) trained on the training pool, plus all pairwise DeLong tests (Holm-corrected) against the Transformer models on the external validation (EV) cohort. |

## Requirements

- Python 3.8+
- numpy, pandas, scipy, scikit-learn, matplotlib
- PyTorch (for `epe_transformer_final.py`)
- Optional: `shap` (SHAP analysis is disabled by default), `openpyxl` (Excel export of DeLong tables)

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch openpyxl
```

## Input data format

Both scripts take a tab-separated file with one row per sample and the following columns:

- `ID` — sample identifier
- `DATASET` — `Training`, `InternalVal` (internal validation subset of the training pool), or `EV` (external validation)
- `Group` — `control`, `EPE`, or `LPE`
- Clinical factors (7): `IVF`, `PMH`, `Birth_history`, `RPL`, `age`, `BMI`, `MAP`
- Targeted metabolites (9): `2-Hydroxy-3-methylbutyric acid`, `Alpha-Linolenic acid`, `Docosapentaenoic acid (DPA)`, `Homovanillic acid`, `Hydroxyphenyllactic acid`, `Isovalerylcarnitine`, `L-Malic acid`, `Palmitelaidic acid`, `Sarcosine`
- Laboratory variables from `PlGF` through `UA` (40 routine laboratory tests + PlGF; 46 columns total including the dedicated PlGF column)

The disease type (EPE or LPE) is detected automatically from the `Group` column.

## Usage

### 1. Multi-modal Transformer (`epe_transformer_final.py`)

```bash
python epe_transformer_final.py --data_path NewEPE_control_all_data.txt --output_dir ./output_EPE
python epe_transformer_final.py --data_path NewLPE_control_all_data.txt --output_dir ./output_LPE
```

Key features:

- **Eight modality combinations**: clinical factors; metabolomics; lab variables+PlGF; clinical+metabolomics; clinical+lab; metabolomics+lab; clinical+PlGF; full fusion (multi-modal fusion).
- **Preprocessing**: winsorization (1st–99th percentiles), robust scaling (median/IQR) fitted on the training pool only; rank-Gaussian transform for metabolomics (training mapping applied to EV).
- **Training**: per seed, the training pool is split 80%/20% (class-stratified) into training and internal-validation subsets; AdamW (lr=3e-4, weight decay=5e-4), batch size 16, up to 220 epochs, early stopping (patience 35) on internal-validation AUC, EMA (decay 0.995), label smoothing (0.05), focal modulation (gamma=1.5, pos_weight capped at 4), modality dropout (0.15), input dropout (0.10), gradient clipping (norm 1.0), ReduceLROnPlateau (x0.5, patience 6).
- **Ensembling**: 8 seeds (5,10,20,42,7,123,2024,3407) combined by temperature-softmax weighting (T=0.025) over seeds within 0.035 of the best internal AUC.
- **Evaluation (EV, assessed once after model selection)**: AUC, calibration (Brier score, slope, intercept), decision curve analysis, pairwise DeLong tests with Holm correction.

Outputs (per run, saved to `--output_dir`):

- `Multi-Modal-Fusion_Validation_<EPE|LPE>_Results_v5.txt` and analogous per-modality score files (EV predictions)
- `Calibration_DCA_Summary_<EPE|LPE>_v5.txt`
- `Model_AUC_Pairwise_DeLong_<EPE|LPE>_v5.xlsx` (AUC/CI, pairwise DeLong, calibration/DCA sheets)
- `ROC_Curves_<EPE|LPE>.png`, `Calibration_<model>.png`, `DCA_<model>.png`
- `gradient_importance_v5/GradImportance_<model>_<EPE|LPE>.txt`

#### Fast smoke-test mode

For quick direction checks only (not for final results):

```bash
python epe_transformer_final.py --data_path <data.txt> --output_dir ./fast_out \
    --debug_fast 1 --fast_modes clinical,clinical+plgf,full --fast_epochs 25 --fast_seeds 42,7
```

### 2. Baselines and pairwise DeLong (`baselines_delong_v2.py`)

Trains ridge logistic regression (L2; C tuned by 10-fold cross-validation, AUC scoring) and histogram gradient boosting (max 300 iterations, early stopping on a 15% validation fraction, random_state=42) on the FULL-62 feature set (7 clinical + 9 metabolites + 46 lab variables including PlGF), standardized with training-pool statistics, then compares all models pairwise on the EV cohort.

```bash
python baselines_delong_v2.py --data_file NewEPE_control_all_data.txt \
    --score_file EPE_model_scores.txt --pos_class EPE --out_prefix reviewer3_EPE
python baselines_delong_v2.py --data_file NewLPE_control_all_data.txt \
    --score_file LPE_model_scores.txt --pos_class LPE --out_prefix reviewer3_LPE
```

`--score_file` accepts Transformer score outputs in two formats: (a) pasted blocks from the `*_Results_v5.txt` files (parsed by `Model` column grouping), or (b) single-model outputs. The primary reference model defaults to `Multi-Modal-Fusion` (EPE) or `Clinical-Factors-Lab-PIGF` when present (override with `--ref_model`).

Outputs:

- `<out_prefix>_delong.csv` — all pairwise DeLong comparisons (Reference, Ref_AUC, Model, AUC, AUC 95% CI, dAUC, raw P, Holm q), sorted by AUC
- `<out_prefix>_EV_scores.csv` — per-sample EV scores for every model
- `<out_prefix>_ROC.png` — EV ROC curves (reference: black; Transformer comparator: red; baselines: thin colored lines)

## Notes

- The external validation cohort is standardized with training-pool statistics only and evaluated exactly once after model selection.
- DeLong confidence intervals use the fast DeLong method; P values are two-sided; Holm correction is applied across all comparisons within each analysis.
- Due to privacy restrictions, the input data files are not publicly available.
