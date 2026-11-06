# Project Handoff

## Completed

- Recreated `.venv` from `requirements.txt` and selected it as the workspace Python interpreter.
- Prepared 500 evenly sampled NYU Depth V2 frames at 240x320.
- Trained the CNN-only and Hybrid models for 8 minutes each without noisy input. CNN completed 27 steps (validation loss 0.01807); Hybrid completed 46 steps (validation loss 0.00804).
- Ran clean and noisy evaluations and generated the qualitative and bar-chart figures.
- No pretrained weights or external APIs were used.

## Environment and measurements

- Python 3.11.9; PyTorch 2.15.0+cpu; OpenCV 5.0.0 with `cv2.ximgproc`.
- Hardware: Intel Core Ultra 5 125H CPU and Intel Arc graphics; no NVIDIA GPU was detected and `torch.cuda.is_available()` was false. Training and latency measurements therefore used the CPU.
- Latency is the mean per-image end-to-end time over 100 test frames. The CNN and Hybrid paths include bicubic upsampling, guided filtering, and the model forward pass.
- Dataset file: `C:\data\nyu_depth_v2_labeled.mat`, 2,972,037,809 bytes (reported by `ls -lh` as 2.8G; 2.77 GiB). The source file stays outside OneDrive.
- Split: 500 contiguous-order samples: 350 train, 50 validation, 100 test (70/10/20).
- Depth is linear and measured in metres; frames are 240x320 and depths are clipped to 10 m.

## Reproduction

Run the download in a separate Git Bash terminal:

```bash
mkdir -p /c/data
curl -L -o /c/data/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

From the workspace in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python ps4.py prepare --mat /c/data/nyu_depth_v2_labeled.mat --samples 500
python ps4.py train --minutes 8
python ps4.py evaluate
python ps4.py evaluate --noise
python ps4.py figures
```

Results are in `outputs/results.csv`, `outputs/results.md`,
`outputs/results_noisy.csv`, and `outputs/results_noisy.md`. Figures are in
`outputs/figures/`; model checkpoints are in `checkpoints/`.

## Known limitations

- The training/evaluation run was CPU-only; model inference is comparatively slow.
- The experiment uses a 500-frame subset and a contiguous split, so it does not establish generalization to unseen datasets or independent scene splits.
- SSIM is computed over each full frame, including invalid-depth placeholder pixels; RMSE/MAE use only valid ground-truth pixels.
- Hybrid boundary RMSE did not improve over Bicubic at 4x or 8x in this run. Results were not tuned or altered.
- The bar chart uses clean evaluation results; noisy metrics are available in their separate CSV and Markdown files.

## Teammate TODOs

- Build a Gradio app and deploy it to Hugging Face Spaces. No live URL means 0 for UI/UX.
- Create an architecture diagram and a design document.
- Create `eval_P4_<TeamID>.md` from `results.md`.
- Document 3 failure cases.
- Run an unseen-data test on Middlebury or KITTI.
- Explore hole filling, a depth-discontinuity detector, and an adaptive edge-confidence gate.
