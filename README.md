# PS4: Edge-Preserving Depth Map Upsampling

This project compares nearest, bilinear, and bicubic interpolation; guided and
joint-bilateral filtering; a CNN-only U-Net; and a guided-filter/CNN hybrid on
NYU Depth V2 at 2x, 4x, and 8x upsampling.

## Setup

Put `nyu_depth_v2_labeled.mat` in `data/raw/`, then install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Run

Prepare an evenly spaced subset of up to 500 frames, resized to 240x320:

```powershell
python ps4.py prepare
```

Train both networks for a time budget per model (default: 10 minutes):

```powershell
python ps4.py train --minutes 10
```

Evaluate every method on the contiguous 20% test split:

```powershell
python ps4.py evaluate
```

The evaluation reports pooled RMSE/MAE/abs-rel/PSNR, boundary RMSE/MAE,
per-image-averaged SSIM, and mean end-to-end latency. It writes
`outputs/results.csv` and `outputs/results.md`; adding `--noise` writes
`outputs/results_noisy.csv` and `outputs/results_noisy.md`.

Create qualitative and metric plots after evaluation:

```powershell
python ps4.py figures
```

This writes `outputs/figures/fig_qualitative_4x.png` and
`outputs/figures/fig_bars.png`.

Or run the entire pipeline:

```powershell
python ps4.py all --minutes 10
```

Append `--noise` to training/evaluation to add 2 cm Gaussian noise and 5%
outlier spikes to low-resolution depth. Use `--device cpu` or `--device cuda`
to select a device (`auto` is the default). The optional joint-bilateral
baseline is reported as unavailable when OpenCV was built without
`cv2.ximgproc`.

## Outputs

- `data/processed/nyu_subset.npz` — resized RGB, depth-in-metres arrays, and source indices.
- `checkpoints/ckpt_cnn.pt`, `checkpoints/ckpt_hybrid.pt` — trained models.
- `outputs/results.csv` / `outputs/results.md` — metrics for each method and scale.
- `outputs/results_noisy.csv` / `outputs/results_noisy.md` — noisy-input robustness metrics.
- `outputs/figures/` — qualitative comparisons and metric bar charts.

Depth is linear, clipped at 10 m, and invalid/non-positive depth pixels are
excluded from metrics. Dataset splitting is contiguous (70% train, 10% val,
20% test) to retain NYU scene ordering.
