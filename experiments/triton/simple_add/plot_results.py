# Load your latest results and visualize them cleanly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot():
    # Read the CSV you just wrote
    path = "./results.csv"
    df = pd.read_csv(path)

    # Basic cleanup
    # Try to infer column names whether they're strings or not
    df.columns = [str(c).strip() for c in df.columns]

    # Coerce types
    for col in ["block_size", "num_warps"]:
        df[col] = df[col].astype(int)
    for col in ["ms", "total_size_in_gb", "gbps"]:
        df[col] = df[col].astype(float)

    # Make dtype labels compact
    df["dtype_clean"] = df["dtype"].astype(str).str.split(".").str[-1]

    # Relative GB/s per dtype (normalizes to each dtype's best)
    df["rel_gbps"] = df.groupby("dtype_clean")["gbps"].transform(lambda s: s / s.max())

    # Best rows per dtype
    best = (
        df.sort_values("gbps", ascending=False)
        .groupby("dtype_clean", as_index=False)
        .head(1)[["dtype_clean", "block_size", "num_warps", "ms", "gbps"]]
        .rename(columns={"dtype_clean": "dtype"})
    )

    # Save an annotated copy
    # annotated_path = "/mnt/data/results_annotated_latest.csv"
    # df.to_csv(annotated_path, index=False)

    # Plot: per dtype, relative GB/s vs block_size, one line per num_warps
    for dtype in sorted(df["dtype_clean"].unique()):
        dsub = (
            df[df["dtype_clean"] == dtype]
            .sort_values(["block_size", "num_warps"])
            .copy()
        )
        plt.figure()
        for nw in sorted(dsub["num_warps"].unique()):
            dd = dsub[dsub["num_warps"] == nw]
            line = dd.groupby("block_size", as_index=False)["rel_gbps"].mean()
            plt.plot(
                line["block_size"],
                line["rel_gbps"],
                marker="o",
                label=f"num_warps={nw}",
            )
        plt.title(f"Relative GB/s vs block_size (dtype={dtype})")
        plt.xlabel("block_size")
        plt.ylabel("relative GB/s (normalized within dtype)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"./{dtype}.png")
