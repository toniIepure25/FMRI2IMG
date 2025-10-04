import os
import numpy as np
from fmri2image.data.datamodule import make_datamodule

def test_datamodule_shapes(tmp_path):
    # Prepare mock ROI and captions
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    N, V = 32, 2048
    np.save(roi_dir / "subj01_roi.npy", np.random.randn(N, V).astype("float32"))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "nsd").mkdir(parents=True, exist_ok=True)
    captions_csv = raw_dir / "nsd" / "captions.csv"
    with open(captions_csv, "w") as f:
        f.write("image_id,caption\n")
        for i in range(N):
            f.write(f"{i},caption {i}\n")

    dm = make_datamodule(
        images_root=str(raw_dir / "nsd" / "images"),
        fmri_root=str(raw_dir / "nsd" / "fmri"),
        captions_csv=str(captions_csv),
        roi_dir=str(roi_dir),
        subject="subj01",
        fmri_dim=V,
        batch_size=4,
        num_workers=0,
    )
    dm.setup(None)
    xb, (idxb, txtb) = next(iter(dm.train_dataloader()))
    assert xb.ndim == 2 and xb.shape[1] == V
    assert len(idxb) == xb.shape[0] == len(txtb)
