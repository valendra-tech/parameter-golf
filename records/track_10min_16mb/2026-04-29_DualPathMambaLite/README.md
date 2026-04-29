# DualPathMambaLite

Experimental dual-path variant based on the local `Muon WD + 10 layer` record.
This is staged under `records/track_10min_16mb`, but it is not a completed leaderboard submission until a CUDA/H100 run produces logs and measured `val_bpb`.

## Idea

Keep the proven 10-layer transformer path and add a compact Mamba-inspired branch over the normalized token embeddings. The Mamba-lite branch uses:

- projection from `512d` to `64d`
- 1 gated causal depthwise-conv block by default
- Triton forward kernel for the causal depthwise conv, with PyTorch fallback
- projection back to `512d`
- a learned sigmoid fusion gate added as a residual after the transformer path

This is intentionally smaller than a full parallel Mamba stack. The copied base artifact was already around 15.37 MB, so the default branch is sized to leave room under the 16,000,000 byte cap.

## Defaults

```bash
USE_MAMBA_PATH=1
MAMBA_DIM=64
MAMBA_LAYERS=1
MAMBA_MIXER=triton_conv1d
MAMBA_CONV_KERNEL=7
MAMBA_GATE_INIT=-3.0
ENFORCE_ARTIFACT_BUDGET=1
```

The script raises after quantization if `final_model.int8.ptz + train_gpt.py` exceeds the 16 MB challenge cap.

`MAMBA_MIXER=conv1d` keeps the original PyTorch causal depthwise-conv mixer for comparison, but it was too slow in early 8xH100 smoke runs. Use `triton_conv1d` for the next speed probe.

## Local Checks

The local venv can import PyTorch and run small CPU/MPS smoke tests. A real score requires running the script on the challenge CUDA environment.
