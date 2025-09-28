from pathlib import Path
import pandas as pd
import numpy as np

class NSDReader:
    def __init__(self, images_root: str, fmri_root: str, captions: str):
        self.images_root = Path(images_root)
        self.fmri_root = Path(fmri_root)
        self.captions = Path(captions)

    def load(self, n: int | None = None, fmri_dim: int = 2048, seed: int = 1337):
        rng = np.random.default_rng(seed)
        df = pd.read_csv(self.captions)
        if n is not None:
            df = df.head(n)
        # TODO: înlocuiește cu ROI reale după fMRIPrep (V1/V2/LOC/whole-brain → vector)
        X = rng.standard_normal((len(df), fmri_dim), dtype=np.float32)
        texts = df["caption"].tolist()
        return X, texts
