from pathlib import Path
import pandas as pd
import numpy as np

class NSDReader:
    def __init__(self, images_root: str, fmri_root: str, captions: str, roi_dir: str | None = None, subject: str = "subj01"):
        self.images_root = Path(images_root)
        self.fmri_root = Path(fmri_root)
        self.captions = Path(captions)
        self.roi_dir = Path(roi_dir) if roi_dir else None
        self.subject = subject

    def load(self, n: int | None = None, fmri_dim: int = 2048, seed: int = 1337):
        df = pd.read_csv(self.captions)
        if n is not None:
            df = df.head(n)
        texts = df["caption"].tolist()

        if self.roi_dir is not None:
            roi_file = self.roi_dir / f"{self.subject}_roi.npy"
            if roi_file.exists():
                X = np.load(roi_file)
                if n is not None:
                    X = X[:n]
                return X.astype("float32"), texts

        # Fallback mock if no ROI file exists
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((len(df), fmri_dim), dtype=np.float32)
        return X, texts
