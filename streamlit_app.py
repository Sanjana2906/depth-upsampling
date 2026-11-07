"""Streamlit demo for PS4: edge-preserving depth map upsampling.

Run locally:  streamlit run streamlit_app.py
Deploy:       Streamlit Community Cloud (main file = streamlit_app.py)
Needs ps4.py, checkpoints/ckpt_cnn.pt, checkpoints/ckpt_hybrid.pt and samples/ next to this file.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps

import ps4

st.set_page_config(page_title="PS4 Depth Upsampling", layout="wide")

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
torch.set_num_threads(2)
H, W, DMAX = ps4.HEIGHT, ps4.WIDTH, ps4.DMAX
XIMG = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter")

LABELS = {
    "nearest": "Nearest",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "joint_bilateral": "Joint bilateral",
    "guided_filter": "Guided filter",
    "cnn": "CNN-only",
    "hybrid": "Hybrid (guided + CNN)",
}


class UserError(Exception):
    """A problem the user can fix (shown as a clean message, not a traceback)."""


# ------------------------------------------------------------------ models and data
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> dict:
    models: dict = {}
    for family in ("cnn", "hybrid"):
        try:
            models[family] = ps4._load_model(ROOT / "checkpoints" / f"ckpt_{family}.pt", family, DEVICE)
        except Exception as error:  # noqa: BLE001
            print(f"[app] {family} model unavailable: {error}")
    return models


MODELS = load_models()
AVAILABLE = [
    m for m in LABELS
    if (m not in ("cnn", "hybrid") or m in MODELS) and (m != "joint_bilateral" or XIMG)
]
DEFAULT_METHODS = [m for m in ("bicubic", "joint_bilateral", "guided_filter", "hybrid") if m in AVAILABLE]
SAMPLES = sorted(p.stem for p in (ROOT / "samples").glob("*.png"))


@st.cache_data(show_spinner=False)
def load_sample(name: str):
    bgr = cv2.imread(str(ROOT / "samples" / f"{name}.png"))
    if bgr is None:
        raise UserError(f"Sample '{name}' could not be read.")
    depth = np.load(ROOT / "samples" / f"{name}.npy").astype(np.float32)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth


def read_rgb(file) -> np.ndarray:
    raw = np.frombuffer(file.getvalue(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise UserError("Could not read the RGB image. Use a PNG or JPG.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def read_depth(file) -> np.ndarray:
    name = file.name.lower()
    data = file.getvalue()
    if name.endswith(".npy"):
        try:
            depth = np.load(io.BytesIO(data), allow_pickle=False).astype(np.float32)
        except Exception as error:  # noqa: BLE001
            raise UserError(f"Could not read the .npy file: {error}") from error
    else:
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise UserError("Could not read the depth file. Use a .npy (metres) or a PNG.")
        if raw.ndim == 3:
            raw = raw[..., 0]
        # 16-bit PNG = millimetres (Kinect/NYU convention); 8-bit PNG = linear 0..10 m
        depth = raw.astype(np.float32) / 1000.0 if raw.dtype == np.uint16 else raw.astype(np.float32) / 255.0 * DMAX
    if depth.ndim != 2:
        raise UserError("Depth must be a single-channel (H x W) map.")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, DMAX)
    return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ drawing helpers
def colorize(arr: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    x = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    img = (colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
    return cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)


def big(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)


def show_grid(items, cols: int = 3) -> None:
    for start in range(0, len(items), cols):
        row = st.columns(cols)
        for col, (img, caption) in zip(row, items[start:start + cols]):
            col.image(img, caption=caption)


# ------------------------------------------------------------------ core computation
def process(source, sample_name, rgb_file, depth_file, depth_kind, factor, noise, methods):
    if not methods:
        raise UserError("Pick at least one method.")
    rng = np.random.default_rng(0)

    if source == "Sample scene":
        if not sample_name:
            raise UserError("No sample scenes found. Add files to the samples/ folder.")
        rgb, depth = load_sample(sample_name)
    else:
        if rgb_file is None or depth_file is None:
            raise UserError("Upload both an RGB image and a depth map, or switch to a sample scene.")
        rgb, depth = read_rgb(rgb_file), read_depth(depth_file)

    if source != "Sample scene" and depth_kind.startswith("Already"):
        gt = None  # no ground truth: just upsample what the user gave us
        low = cv2.resize(depth, (W // factor, H // factor), interpolation=cv2.INTER_NEAREST)
    else:
        gt = depth
        low = ps4._downsample(gt, factor)
    if noise:
        low = ps4._add_noise(low, rng)

    valid = (gt > 1e-3) if gt is not None else None
    band = (ps4.edge_mask(gt) & valid) if gt is not None else None

    preds, rows = {}, []
    for m in methods:
        t0 = time.perf_counter()
        pred = ps4._predict_method(m, low, rgb, MODELS, DEVICE, XIMG)
        ms = (time.perf_counter() - t0) * 1000
        preds[m] = pred
        if gt is None:
            rows.append([LABELS[m], "n/a", "n/a", "n/a", round(ms, 1)])
        else:
            e = pred - gt
            rows.append([
                LABELS[m],
                round(float(np.sqrt(np.mean(e[valid] ** 2))), 4),
                round(float(np.mean(np.abs(e[valid]))), 4),
                round(float(np.sqrt(np.mean(e[band] ** 2))), 4) if band.any() else "n/a",
                round(ms, 1),
            ])

    ref = gt if gt is not None else np.maximum.reduce(list(preds.values()))
    vmin, vmax = 0.0, float(max(ref.max(), 1e-3))
    depth_items = [(big(rgb), "RGB input")]
    if gt is not None:
        depth_items.append((colorize(gt, vmin, vmax), "Ground truth"))
    depth_items.append((colorize(ps4.upsample(low, rgb.shape[:2], "nearest"), vmin, vmax),
                        f"Low-res input ({factor}x smaller), nearest-upscaled"))
    depth_items += [(colorize(p, vmin, vmax), LABELS[m]) for m, p in preds.items()]

    error_items = []
    if gt is not None:
        error_items.append((cv2.resize((band * 255).astype(np.uint8), (W * 2, H * 2),
                                       interpolation=cv2.INTER_NEAREST), "Boundary band (GT depth edges)"))
        for m, p in preds.items():
            error_items.append((colorize(np.abs(p - gt) * valid, 0.0, 0.5, "inferno"),
                                f"|error| {LABELS[m]} (0-0.5 m)"))

    table = pd.DataFrame(rows, columns=["Method", "RMSE (m)", "MAE (m)", "Boundary RMSE (m)", "Latency (ms)"])
    return depth_items, table, error_items, gt is not None


# ------------------------------------------------------------------ UI
st.title("Edge-preserving depth map upsampling")
st.markdown(
    "Rebuilds a dense depth map from a low-resolution one, using the RGB image as guidance. "
    "Compares **interpolation**, **classical edge-aware filters**, a **CNN**, and a **hybrid** "
    "(bicubic → guided filter → CNN). Depth is linear, in metres, 240×320. "
    "The colour scale is shared across maps, so colours are comparable."
)

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Sample scene", "Upload my own"], label_visibility="collapsed")
    sample_name = rgb_file = depth_file = None
    depth_kind = "Full-resolution (simulate low-res from it)"
    if source == "Sample scene":
        sample_name = st.selectbox("Sample scene (NYU test frame, has ground truth)", SAMPLES) if SAMPLES else None
    else:
        rgb_file = st.file_uploader("RGB image", type=["png", "jpg", "jpeg"])
        depth_file = st.file_uploader("Depth map: .npy (metres), 16-bit PNG (mm) or 8-bit PNG (0-10 m)",
                                      type=["npy", "png"])
        depth_kind = st.radio("The uploaded depth is",
                              ["Full-resolution (simulate low-res from it)", "Already low-resolution"])

    st.header("Settings")
    factor = st.radio("Upsampling factor", [2, 4, 8], index=1, horizontal=True)
    noise = st.checkbox("Add sensor noise (2 cm Gaussian + 5% outlier spikes)")
    methods = st.multiselect("Methods to run", AVAILABLE, default=DEFAULT_METHODS,
                             format_func=lambda m: LABELS[m])
    go = st.button("Upsample", type="primary", use_container_width=True)
    st.caption("The CNN and hybrid models are small and were trained for minutes, "
               "so they are not expected to beat well-tuned classical filters yet.")

if go:
    start = time.perf_counter()
    try:
        with st.spinner("Upsampling..."):
            depth_items, table, error_items, has_gt = process(
                source, sample_name, rgb_file, depth_file, depth_kind, int(factor), noise, methods)
    except UserError as error:
        st.error(str(error))
    except Exception as error:  # noqa: BLE001
        st.error(f"Something went wrong: {error}")
    else:
        note = "" if has_gt else " No ground truth, so no error metrics."
        st.success(f"Done in {time.perf_counter() - start:.1f} s | {factor}x upsampling | "
                   f"noise {'on' if noise else 'off'} | CPU.{note}")
        st.subheader("Depth maps")
        show_grid(depth_items)
        st.subheader("Metrics")
        st.caption("Valid GT pixels only; boundary = band around GT depth edges.")
        st.dataframe(table, hide_index=True, use_container_width=True)
        if error_items:
            st.subheader("Error maps and boundary band")
            show_grid(error_items)
else:
    st.info("Choose a scene in the sidebar and press **Upsample**.")
