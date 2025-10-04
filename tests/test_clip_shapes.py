import os
import numpy as np
import pytest

@pytest.mark.parametrize("path", [
    "data/processed/nsd/clip_text.npy",
])
def test_clip_text_shape_exists(path):
    assert os.path.exists(path), f"missing {path}"
    x = np.load(path)
    assert x.ndim == 2 and x.shape[1] >= 256, f"unexpected shape {x.shape}"
