# Is imputed history better than no history at all? Forecasting with implicit neural representations

Code for the paper by Jessy Azizi, Vincent Guigue and Sophie Martin (UMR MIA Paris-Saclay, AgroParisTech).

The proposed model reads the observed prices of the lookback window as an un-imputed set `(t, y_t)`. A Deep Sets encoder with layer-normalized mean and max pooling summarises that set, and an LSTM encodes the exogenous forecasts. A hypernetwork turns both into FiLM parameters. These modulate each layer of an implicit neural representation at every step of the horizon, and each modulated layer is followed by layer normalization and a GELU. Robustness is measured with the history degradation ratio

```
rho(p) = (MAE(p) - MAE(0)) / (MAE_abl - MAE(0))
```

where `MAE_abl` is the same architecture retrained without price history.

## Results

Price history is masked point-wise at random (MCAR) with rate `p`. Every gradient-based model is trained with 5 seeds and evaluated with 5 mask seeds per rate. LEAR is deterministic.

**Table 1.** Nominal MAE at `p = 0`, mean ± std over 5 seeds.

| Model     | NP          | PJM         | BE          | FR          | DE          |
|-----------|-------------|-------------|-------------|-------------|-------------|
| LEAR      | 1.86        | 3.35        | 6.50        | 4.51        | 3.94        |
| DNN       | 2.35 ± 0.09 | 3.61 ± 0.05 | 7.68 ± 0.11 | 6.26 ± 0.08 | 4.50 ± 0.07 |
| EPF-Tr.   | 2.78 ± 0.07 | 3.58 ± 0.11 | 9.01 ± 1.15 | 4.58 ± 0.02 | 5.78 ± 0.11 |
| Tr. imp.  | 3.03 ± 0.25 | 3.39 ± 0.04 | 7.72 ± 0.21 | 5.11 ± 0.22 | 4.93 ± 0.16 |
| Tr. masq. | 2.93 ± 0.38 | 3.35 ± 0.07 | 7.70 ± 0.18 | 5.13 ± 0.29 | 4.90 ± 0.13 |
| INR       | 2.32 ± 0.14 | 3.79 ± 0.10 | 7.63 ± 0.13 | 4.97 ± 0.05 | 4.67 ± 0.21 |

**Table 2.** History degradation ratio `rho(p)` under MCAR, 25 runs per entry (5 for LEAR). Bold: `rho > 1`, i.e. the model fed masked history is less accurate than a price-free model retrained from scratch.

| Market | Model     | p = 0.3 | p = 0.5 | p = 0.7  | p = 0.9  | p = 0.95 |
|--------|-----------|---------|---------|----------|----------|----------|
| NP     | LEAR      | 0.01    | 0.04    | 0.10     | 0.25     | 0.32     |
|        | DNN       | 0.00    | 0.00    | 0.02     | 0.11     | 0.18     |
|        | EPF-Tr.   | 0.00    | 0.01    | 0.02     | 0.07     | 0.11     |
|        | Tr. imp.  | 0.00    | 0.00    | 0.02     | 0.10     | 0.14     |
|        | Tr. masq. | 0.01    | 0.02    | 0.06     | 0.16     | 0.24     |
|        | INR       | 0.00    | 0.00    | 0.01     | 0.08     | 0.18     |
| PJM    | LEAR      | 0.24    | 0.76    | **1.73** | **3.12** | **3.40** |
|        | DNN       | 0.02    | 0.10    | 0.26     | 0.99     | **1.42** |
|        | EPF-Tr.   | 0.02    | 0.07    | 0.19     | 0.55     | 0.73     |
|        | Tr. imp.  | 0.05    | 0.19    | 0.54     | **1.51** | **1.93** |
|        | Tr. masq. | 0.08    | 0.20    | 0.55     | **1.67** | **2.63** |
|        | INR       | 0.02    | 0.05    | 0.10     | 0.49     | **1.38** |
| BE     | LEAR      | 0.04    | 0.11    | 0.23     | 0.58     | 0.78     |
|        | DNN       | 0.01    | 0.02    | 0.04     | 0.18     | 0.29     |
|        | EPF-Tr.   | 0.00    | 0.01    | 0.02     | 0.09     | 0.17     |
|        | Tr. imp.  | 0.02    | 0.05    | 0.10     | 0.25     | 0.33     |
|        | Tr. masq. | 0.01    | 0.04    | 0.09     | 0.20     | 0.27     |
|        | INR       | 0.00    | 0.01    | 0.02     | 0.11     | 0.24     |
| FR     | LEAR      | 0.03    | 0.10    | 0.28     | 0.76     | **1.08** |
|        | DNN       | 0.00    | 0.02    | 0.09     | 0.24     | 0.45     |
|        | EPF-Tr.   | 0.01    | 0.04    | 0.11     | 0.26     | 0.36     |
|        | Tr. imp.  | 0.02    | 0.05    | 0.11     | 0.26     | 0.34     |
|        | Tr. masq. | 0.04    | 0.13    | 0.29     | 0.54     | 0.60     |
|        | INR       | 0.01    | 0.02    | 0.05     | 0.22     | 0.49     |
| DE     | LEAR      | 0.08    | 0.21    | 0.53     | **1.47** | **2.05** |
|        | DNN       | 0.00    | 0.00    | 0.07     | 0.44     | 0.73     |
|        | EPF-Tr.   | -0.06   | -0.13   | -0.21    | -0.17    | -0.06    |
|        | Tr. imp.  | -0.01   | 0.01    | 0.07     | 0.30     | 0.44     |
|        | Tr. masq. | 0.08    | 0.20    | 0.40     | 0.76     | 0.92     |
|        | INR       | -0.01   | -0.01   | -0.01    | 0.13     | 0.40     |

**Table 3.** Ablation of the exogenous conditioning of the INR on DE (MAE, gap to the full model). `Static cond.` fixes the exogenous code over the horizon, `No exog.` removes the exogenous covariates.

| Variant      | p = 0        | p = 0.3      | p = 0.5       | p = 0.7       | p = 0.9        | p = 0.95       |
|--------------|--------------|--------------|---------------|---------------|----------------|----------------|
| Full         | 4.67         | 4.64         | 4.64          | 4.65          | 4.97           | 5.58           |
| Static cond. | 5.07 (+8.5%) | 5.09 (+9.5%) | 5.13 (+10.6%) | 5.25 (+12.8%) | 5.91 (+18.9%)  | 6.85 (+22.7%)  |
| No exog.     | 6.83 (+46%)  | 7.21 (+55%)  | 7.70 (+66%)   | 8.57 (+84%)   | 10.23 (+106%)  | 11.25 (+102%)  |

## Setup

```bash
uv sync
```

## Data

```bash
uv run python -m data.preprocess
```

This downloads the five markets of the EPF benchmark (Lago et al., 2021) into `data/raw/`. Following that benchmark, it keeps the last two years as the test set and holds out the last 42 weeks of the training period for validation. It then scales everything with a scaler fitted on the training split only and writes `data/processed/{market}_series.pkl`.

## Reproducing the paper

```bash
bash run_all.sh
```

This trains every model on every market, evaluates each checkpoint under MCAR masking and prints Tables 1 to 3. LEAR runs on CPU and takes about two hours per market; the other models train in minutes on a GPU. To run the two in parallel:

```bash
MODELS=lear bash run_all.sh train &
MODELS="dnn epf_transformer transformer_imputed transformer_masked inr" bash run_all.sh train
wait
bash run_all.sh eval
```

`MARKETS`, `MODELS` and `SEEDS` restrict any stage, for example `MARKETS=DE MODELS=inr bash run_all.sh`.

## Running a single model

```bash
uv run python train.py inr --market DE --seed 0 --variant full
```

| argument    | values                                                                                   |
|-------------|------------------------------------------------------------------------------------------|
| model       | `lear`, `dnn`, `epf_transformer`, `transformer_imputed`, `transformer_masked`, `inr`    |
| `--market`  | `NP`, `PJM`, `BE`, `FR`, `DE`                                                            |
| `--variant` | `full`, `ablated` (no price history, denominator of rho), `static`, `noexog` (INR only) |

Any configuration value can be overridden in dotted form, for example `train.num_epochs=50 dataset.num_workers=0`. Each run writes its configuration and checkpoint to `runs/{market}/{model}/{variant}/seed{seed}/`.

## Evaluating and reporting

```bash
uv run python evaluate.py
uv run python tables.py
```

`evaluate.py` reloads every checkpoint under `runs/` and evaluates it on the test set without retraining. It covers `p = 0` and `p in {0.3, 0.5, 0.7, 0.9, 0.95}` with five mask seeds each, and writes `metrics.csv` next to the checkpoint. `--market`, `--model` and `--variant` re-evaluate a subset only. The mask is a deterministic function of the mask seed and of the window position, so every model sees the same masks.

`tables.py` prints:

- Table 1: nominal MAE at `p = 0`, mean ± std over training seeds.
- Table 2: `rho(p)` per market and missingness rate.
- Table 3: the ablation of the exogenous conditioning, for every market where it was run.

## Models

| name                  | description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `lear`                | LASSO-estimated autoregressive model, lags D-1, D-2, D-3, D-7, missing prices linearly interpolated; averaged over 84-day, 1092-day and full-history calibration windows, recalibrated weekly |
| `dnn`                 | two-hidden-layer DNN on the same features                                                        |
| `epf_transformer`     | EPF-Transformer (Llorente & Portela), per-market hyperparameters from its authors, 168 h lookback |
| `transformer_imputed` | hourly-token Transformer on the linearly interpolated lookback                                   |
| `transformer_masked`  | same architecture, missing positions excluded from attention                                     |
| `inr`                 | proposed model                                                                                   |

## Layout

```
conf/            base.yaml and one configuration per model
data/            download, preprocessing, windowing and MCAR masking
models/          the six forecasters
training.py      training loop, validation and test MAE
train.py         trains one model and saves its checkpoint
evaluate.py      masking sweep on saved checkpoints
tables.py        Tables 1 to 3
run_all.sh       full campaign
```
