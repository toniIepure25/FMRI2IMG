import csv
import glob
from pathlib import Path

def last_row_of_csv(path):
    last = None
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    return last or {}

def main():
    rows = []
    for csv_path in glob.glob("outputs/**/metrics.csv", recursive=True):
        row = last_row_of_csv(csv_path)
        run_dir = Path(csv_path).parent
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return float("nan")
        rows.append({
            "run": str(run_dir),
            "loss": to_float(row.get("train/loss_epoch")),
            "top1": to_float(row.get("train/retrieval_zt_top1")),
            "top5": to_float(row.get("train/retrieval_zt_top5")),
            "temp": to_float(row.get("train/temp_epoch")),
        })

    rows.sort(key=lambda x: (-(x["top5"] if x["top5"]==x["top5"] else -1), x["loss"]))

    print("| Run | Loss | Top@1 | Top@5 | Temp |")
    print("|-----|------|-------|-------|------|")
    for r in rows:
        def fmt(v, nd=3):
            return "nan" if v != v else f"{v:.{nd}f}"
        print(f"| {r['run']} | {fmt(r['loss'])} | {fmt(r['top1'])} | {fmt(r['top5'])} | {fmt(r['temp'], 4)} |")

if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True, parents=True)
    main()
