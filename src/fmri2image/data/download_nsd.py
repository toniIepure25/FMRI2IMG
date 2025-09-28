from pathlib import Path
import argparse, json

def main(raw_root: str):
    root = Path(raw_root); (root/"nsd"/"images").mkdir(parents=True, exist_ok=True)
    (root/"nsd"/"fmri").mkdir(parents=True, exist_ok=True)
    # MOCK: scriem un captions CSV minim și un README cu pași reali
    captions = root/"nsd"/"captions.csv"
    if not captions.exists():
        captions.write_text("image_id,caption\n0,a dog on grass\n1,a red car\n2,a mountain scene\n")
    readme = root/"nsd"/"README.txt"
    readme.write_text(
        "NSD placeholder. Pentru date reale: configurați AWS CLI și sincronizați din bucketul NSD.\n"
        "Struc.: raw/nsd/images/, raw/nsd/fmri/, raw/nsd/captions.csv\n"
    )
    print(f"[ok] NSD mock at: {root/'nsd'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True)
    args = ap.parse_args()
    main(args.raw_root)
