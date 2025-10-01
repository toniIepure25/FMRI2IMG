import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_metrics(log_dir):
    metrics_file = Path(log_dir) / "metrics.csv"
    if not metrics_file.exists():
        print(f"No metrics found at {metrics_file}")
        return

    df = pd.read_csv(metrics_file)
    # păstrează doar coloane utile
    cols = [c for c in df.columns if "train" in c or "val" in c]
    df = df[["step"] + cols]

    df.plot(x="step", y=[c for c in cols if "loss" in c], title="Loss")
    plt.show()

    if any("retrieval" in c for c in cols):
        df.plot(x="step", y=[c for c in cols if "retrieval" in c], title="Top-k Retrieval")
        plt.show()


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    plot_metrics(log_dir)
