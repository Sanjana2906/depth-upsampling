"""Export a few NYU test frames to samples/ for the demo app. Run once from the project root."""
from pathlib import Path
import cv2, numpy as np
import ps4

images, depths = ps4.load_subset(Path("data/processed/nyu_subset.npz"))
_, _, test = ps4.split_indices(len(images))
out = Path("samples"); out.mkdir(exist_ok=True)
for k, idx in enumerate(np.linspace(test[0], test[-1], 5).astype(int), start=1):
    cv2.imwrite(str(out / f"scene_{k}.png"), cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))
    np.save(out / f"scene_{k}.npy", depths[idx].astype(np.float32))   # metres
print("wrote", len(list(out.glob('*.png'))), "scenes to", out)