# Experiments

Each experiment is a single YAML file. To add a new experiment, copy the nearest
predecessor and change **only the variable(s) being tested**.

Run an experiment:
```bash
uv run python train.py --exp experiments/expXX_name.yaml
```

---

## Log

| # | Name | Key change | Notes |
|---|------|-----------|-------|
| 01 | `exp01_post_norm` | **Baseline** — Post-LayerNorm (GPT-2 style) | Reference run |
| 02 | `exp02_pre_norm`  | Pre-LayerNorm (GPT-3 / LLaMA style) vs exp01 | |
| 03 | | | |
| … | | | |

---

## What to vary (ideas for future experiments)

- **Normalisation**: Post-norm → Pre-norm → RMSNorm → no-norm
- **Positional encoding**: Learned absolute → RoPE → ALiBi → NoPE
- **Activation**: GELU → SwiGLU → ReLU²
- **Depth vs width**: more layers / fewer heads vs wider hidden dim
- **Context length**: 512 → 1024 → 2048
- **Dropout**: scan attn_dropout and ffn_dropout independently
- **Weight tying**: tie token embedding ↔ output head weights
- **Bias terms**: `bias: true` vs `bias: false`
- **Optimiser**: AdamW → Muon → Lion
- **LR schedule**: cosine → constant → trapezoidal
- **Data size**: 200k → 500k → full dataset
