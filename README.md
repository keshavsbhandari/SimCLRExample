# SimCLR Fraud Detection

Self-supervised contrastive pretraining (SimCLR-style) for tabular fraud detection on PaySim.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
cp .env.example .env   # add WANDB_API_KEY
```

## Google Colab

Colab provides a Python runtime — **no virtualenv needed**. Install into the runtime and run from the repo root.

```bash
git clone <your-repo-url>
cd SimCLR

pip install -U pip
pip install -e .

cp .env.example .env   # optional: add WANDB_API_KEY
# or: export WANDB_API_KEY=your_key

# Runtime → Change runtime type → GPU
python main.py experiment=colab
# or: simclr-run experiment=colab
```

Notes:
- Skip `python -m venv .venv`; use `pip install -e .` directly.
- Always run from the repo root so Hydra finds `configs/`.
- Use `experiment=colab` (T4-friendly) or `mini`; avoid `gpu5090` (large batches for local RTX 5090).
- `outputs/`, `data/`, and `wandb/` are lost when the runtime ends — log to W&B or copy to Google Drive.
- Quick smoke test: `WANDB_MODE=disabled python main.py experiment=mini pretrain.max_epochs=1 finetune.max_epochs=1`

## Run

Each experiment is a self-contained YAML under `configs/experiment/`. PaySim is downloaded once to `data/raw/paysim.csv`; preprocessing runs in memory on each pipeline start.

Run from the **repo root** so Hydra finds `configs/`.

### Entry points (after `pip install -e .`)

All training commands below are equivalent — pick whichever you prefer:

```bash
# 1. Direct (default)
python main.py experiment=mini

# 2. Console script (registered by pip install -e .)
simclr-run experiment=mini

# 3. Module invocation (same as simclr-run)
python -m simclr_fraud.train experiment=mini
```

For batch scoring after training:

```bash
simclr-predict --experiment mini --input data/raw/paysim.csv --output outputs/mini/scores.csv
# or: python -m simclr_fraud.predict --experiment mini --input ... --output ...
```

`simclr-run` and `python -m simclr_fraud.train` delegate to `main.py` in the repo root, so **run them from the cloned project directory** (where `configs/` lives).

### Full pipeline (pretrain → finetune → eval)

```bash
# 10% stratified subset (good default for dev)
python main.py experiment=mini

# Full dataset
python main.py experiment=full

# Local RTX 5090 (large batches, bf16)
python main.py experiment=gpu5090

# Google Colab / small GPU
python main.py experiment=colab
```

### Finetune only (load encoder, train classifier)

Skip SimCLR pretrain and load a saved encoder, then train the fraud classifier:

```bash
# Encoder from the same experiment (must exist at outputs/{name}/pretrain/encoder.pt)
python main.py experiment=gpu5090 run_pretrain=false

# Encoder from a different experiment or custom path
python main.py experiment=mini run_pretrain=false \
  finetune.encoder_checkpoint=outputs/gpu5090/pretrain/encoder.pt

# Load encoder but freeze it — train only the classifier head
python main.py experiment=gpu5090 run_pretrain=false finetune.freeze_encoder=true
```

### Pretrain only

```bash
python main.py experiment=mini run_finetune=false
# saves outputs/mini/pretrain/encoder.pt
```

### Baseline (no SimCLR — random encoder)

```bash
python main.py experiment=baseline_mlp
```

### Eval-only (no training)

Re-run plots and test metrics on an existing classifier:

```bash
python main.py experiment=gpu5090 run_eval_only=true
```

Requires `outputs/{name}/finetune/classifier.pt`.

### Batch predict (score a new CSV)

Uses `outputs/{name}/config_resolved.yaml`, `preprocessor.joblib`, and `classifier.pt`:

```bash
simclr-predict --experiment gpu5090 \
  --input data/raw/paysim.csv \
  --output outputs/gpu5090/scores.csv

# equivalent:
python -m simclr_fraud.predict --experiment gpu5090 --input ... --output ...
```

Adds a `fraud_probability` column to the output CSV.

### Overrides and debug runs

```bash
# Force re-download PaySim CSV
python main.py experiment=mini force_download=true

# Disable W&B
WANDB_MODE=disabled python main.py experiment=mini

# Quick smoke test (few batches / epochs)
python main.py experiment=mini \
  pretrain.max_epochs=1 finetune.max_epochs=1 \
  pretrain.limit_train_batches=10 pretrain.limit_val_batches=5 \
  finetune.limit_train_batches=10 finetune.limit_val_batches=5

# Override W&B project per experiment YAML (Hydra wins over .env WANDB_PROJECT)
python main.py experiment=gpu5090 wandb.project=my-project
```

### Where artifacts are saved

All paths are derived from `name` in the experiment YAML (see `src/simclr_fraud/paths.py`):

```text
outputs/{name}/
├── config_resolved.yaml
├── preprocessor.joblib
├── pretrain/
│   ├── encoder.pt              # exported after pretrain (used by finetune)
│   └── checkpoints/best*.ckpt  # Lightning best checkpoint
├── finetune/
│   ├── classifier.pt           # full model for predict / eval-only
│   └── checkpoints/best*.ckpt
└── eval/
    ├── metrics.json
    └── *.png
```

Only **loading** the encoder accepts a custom path: `finetune.encoder_checkpoint=...`. Save locations are fixed under `outputs/{name}/`.

Training outputs also log to W&B when `WANDB_API_KEY` is set. Project comes from `wandb.project` in the experiment YAML; run group/name use `wandb.experiment_name` and `{group}/{stage}`.

## Reproducibility

Global random seed defaults to `seed: 42` in `configs/config.yaml` (PyTorch, NumPy, Python). Data split uses `data.split.seed`; subset sampling uses `data.fraction_seed`.

Each run saves:
- `outputs/{name}/config_resolved.yaml` — full merged config
- `outputs/{name}/preprocessor.joblib` — sklearn preprocessor fit on train split

## Runtime control (stop or cap epochs mid-training)

Each run creates `outputs/{experiment_name}/control.yaml`. Edit it **while training** — changes are picked up at the **end of each epoch** (checkpoint saved, then graceful stop).

```yaml
stop: false           # set true to stop after the current epoch
skip_finetune: false  # set true to skip finetune after pretrain finishes
max_epochs: null      # set e.g. 5 to cap the current stage (lower from 20 mid-run)
```

Examples:

```bash
# Start a long run
python main.py experiment=gpu5090

# In another terminal, cap pretrain at 5 epochs:
# edit outputs/gpu5090/control.yaml → max_epochs: 5

# Or stop immediately after this epoch:
# edit outputs/gpu5090/control.yaml → stop: true

# Skip finetune when pretrain ends:
# edit outputs/gpu5090/control.yaml → skip_finetune: true
```

Do not edit `configs/experiment/*.yaml` mid-run — Hydra only reads those at startup. Use `control.yaml` for live changes.

Pretrain and finetune use **early stopping** when `early_stopping_patience` is set in the experiment YAML. Best weights are reloaded before saving `encoder.pt` / `classifier.pt` and before test/eval.

## Evaluation plots (W&B + local)

After finetune test, evaluation runs automatically and produces:

- ROC curve with threshold markers
- Precision-recall curve with threshold markers
- Threshold sweep (precision / recall / F1 vs threshold)
- Score distribution (fraud vs non-fraud)
- Confusion matrices at θ=0.5 and best val F1 threshold

Artifacts are saved under `outputs/{name}/eval/` and logged to W&B under the `eval/` prefix on the finetune run.

Configure in experiment YAML:

```yaml
eval:
  enabled: true
  log_wandb: true
  save_plots: true
  thresholds: [0.5, 0.7, 0.9, 0.95, 0.99, 0.995]
  find_best_f1_on: val   # also mark best-F1 threshold from validation set
```

Disable with `eval.enabled=false` or `eval.log_wandb=false`.

## CI

GitHub Actions runs `ruff check` and `pytest` on push/PR (see `.github/workflows/ci.yml`). Locally:

```bash
ruff check src tests
pytest tests/ -q
```
