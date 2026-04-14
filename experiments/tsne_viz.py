"""t-SNE visualization: validate that M1 captures geographic semantics.

Applies t-SNE dimensionality reduction to M1 latent representations to
visually validate that the autoencoder successfully captures geographic
semantics and produces distinct clustering patterns by data distribution.

Analyzes both RQ (range query) and SJ (self-join) embeddings:
  - RQ embeddings: AE_S1, AE_C2 (synthetic), AE_S3, AE_S4 (synth+real)
  - SJ embeddings: AE_C2 (synthetic), AE_S4 (synth+real, mixed distributions)

Generates:
  1. tsne_distribution.png      — RQ: distribution clustering (Table 3 & 4)
  2. tsne_selectivity.png       — RQ: distribution vs selectivity comparison
  3. tsne_distribution_sj.png   — SJ: distribution clustering (Table 4 mixed data)
  4. tsne_viz.csv               — quantitative clustering metrics

Metrics (computed in original high-dimensional embedding space):
  - silhouette_dist: silhouette score using distribution type as labels
    (higher = distributions better separated, M1 captures geographic semantics)
  - silhouette_dist_random: same on shuffled embeddings (baseline)
  - silhouette_dist_tsne: silhouette on t-SNE 2D for visual reference

Usage:
    python run_all.py --tables 99
    python -m experiments.tsne_viz --data-dir ./data/downloaded_data
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

N_SAMPLES = 5000
RANDOM_SEED = 42
PERPLEXITY = 30

# RQ AE configurations (Table 3 & 4)
AE_CONFIGS_RQ = [
    ("AE_S1", "Stacked, LD=384, Synthetic"),
    ("AE_C2", "CNN, LD=3072, Synthetic"),
    ("AE_S3", "Stacked, LD=48, Synth+Real"),
    ("AE_S4", "Stacked, LD=384, Synth+Real"),
]

# SJ AE configurations (Table 4 mixed data validation)
AE_CONFIGS_SJ = [
    ("AE_C2", "CNN, LD=3072, Synthetic"),
    ("AE_S4", "Stacked, LD=384, Synth+Real"),
]

# Distribution type -> color mapping for consistent coloring across plots
DIST_COLORS = {
    "bit":       "#e74c3c",
    "diagonal":  "#3498db",
    "gaussian":  "#2ecc71",
    "parcel":    "#f39c12",
    "sierpinski":"#9b59b6",
    "uniform":   "#95a5a6",
    "real":      "#1a1a2e",
    "lake":      "#1a759f",
    "park":      "#76b947",
}

# JN distribution code -> name mapping
# From spatial-embedding/modelsSJ/gen_py/generate_input_JN.py
JN_DIST_CODE_MAP = {
    "0.0": "uniform",
    "1.0": "parcel",
    "2.0": "gaussian",
    "3.0": "bit",
    "4.0": "diagonal",
    "5.0": "sierpinski",
    "6.0": "real",
    # Also handle float/int keys
    0.0: "uniform",
    1.0: "parcel",
    2.0: "gaussian",
    3.0: "bit",
    4.0: "diagonal",
    5.0: "sierpinski",
    6.0: "real",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_ds_file(data_dir, n_samples, task="rq"):
    """Find distribution metadata file matching sample count.

    Searches subdirectories for ds_* files matching the expected sample count.
    For RQ: looks for ds_*_rq*.npy or y_*_distr.npy
    For SJ: looks for ds_*_jn*.npy
    """
    keyword = "rq" if task == "rq" else "jn"
    for subdir in os.listdir(data_dir):
        subpath = os.path.join(data_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for f in os.listdir(subpath):
            if not (f.startswith("ds_") and keyword in f and f.endswith(".npy")):
                continue
            path = os.path.join(subpath, f)
            loaded = np.load(path, allow_pickle=True)
            if loaded.shape[0] == n_samples:
                return loaded
        # Also check y_*_distr.npy (AE_S1 RQ format)
        if task == "rq":
            for f in os.listdir(subpath):
                if "distr" in f and f.endswith(".npy"):
                    path = os.path.join(subpath, f)
                    loaded = np.load(path, allow_pickle=True)
                    if loaded.shape[0] == n_samples and loaded.ndim == 2:
                        return loaded
    return None


def load_embedding_data(data_dir, ae_name, task="rq"):
    """Load embeddings, targets, and distribution metadata for a given task."""
    task_prefix = task
    x = np.load(os.path.join(data_dir, f"x_{task_prefix}_{ae_name}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{task_prefix}_{ae_name}.npy"))
    ds = _find_ds_file(data_dir, x.shape[0], task=task)
    return x, y, ds


def subsample(x, y, ds, n_samples, seed):
    """Randomly subsample for t-SNE."""
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    if n <= n_samples:
        return x, y, ds
    idx = rng.choice(n, size=n_samples, replace=False)
    return x[idx], y[idx], ds[idx] if ds is not None else None


def get_dist_labels(ds, task="rq"):
    """Extract distribution type labels from metadata array.

    For RQ synthetic data: uses column 1 (distribution name strings)
    For RQ real data: derives lake/park from dataset filename
    For JN data: decodes numeric distribution codes using JN_DIST_CODE_MAP

    Returns labels array and sorted unique label names, or (None, None)
    if distribution metadata is unavailable or has only one category.
    """
    if ds is None:
        return None, None

    dist_col = ds[:, 1]

    if task in ("sj", "sj_sel", "sj_mbr", "bj_sel", "bj_mbr"):
        # JN data uses numeric codes — decode to distribution names
        labels = np.array([
            JN_DIST_CODE_MAP.get(code, JN_DIST_CODE_MAP.get(str(code), "unknown"))
            for code in dist_col
        ])
        unique = sorted(set(labels) - {"unknown", ""})
        if len(unique) > 1:
            return labels, unique
        return None, None

    # RQ data
    unique = sorted(set(d for d in dist_col if d))
    if len(unique) > 1:
        return dist_col, unique

    # For real-only RQ data, derive labels from dataset filename
    if unique == ["real"]:
        def _geo_category(path):
            fname = path.split("/")[-1] if "/" in path else path
            if fname.startswith("lake"):
                return "lake"
            if fname.startswith("park"):
                return "park"
            return "other"

        labels = np.array([_geo_category(name) for name in ds[:, 0]])
        derived_unique = sorted(set(labels) - {"other"})
        if len(derived_unique) > 1:
            return labels, derived_unique

    return None, None


def bin_selectivity(y, n_bins=5):
    """Quantile-based selectivity binning."""
    bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    labels = np.clip(np.digitize(y, bins) - 1, 0, n_bins - 1)
    names = [f"[{bins[i]:.3f}, {bins[i+1]:.3f})" for i in range(n_bins)]
    return labels, names


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def run_tsne(x_flat, seed, perplexity=PERPLEXITY):
    """Run t-SNE dimensionality reduction."""
    print(f"    t-SNE: {x_flat.shape[0]} samples, {x_flat.shape[1]} dims ...")
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed,
        n_iter=1000, init="pca", learning_rate="auto",
    )
    return tsne.fit_transform(x_flat)


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def compute_distribution_metrics(x_flat, tsne_2d, dist_labels, unique_dists, seed):
    """Compute distribution clustering metrics in both high-dim and t-SNE space."""
    valid_mask = np.array([d in unique_dists and d != "" for d in dist_labels])
    if valid_mask.sum() < 50:
        return {}

    label_to_int = {name: i for i, name in enumerate(unique_dists)}
    int_labels = np.array([label_to_int.get(d, -1) for d in dist_labels])

    x_valid = x_flat[valid_mask]
    tsne_valid = tsne_2d[valid_mask]
    labels_valid = int_labels[valid_mask]

    sample_size = min(2000, len(labels_valid))

    sil_hd = silhouette_score(x_valid, labels_valid,
                              sample_size=sample_size, random_state=seed)
    sil_tsne = silhouette_score(tsne_valid, labels_valid,
                                sample_size=sample_size, random_state=seed)

    rng = np.random.RandomState(seed + 1)
    shuffled_labels = labels_valid[rng.permutation(len(labels_valid))]
    sil_random = silhouette_score(x_valid, shuffled_labels,
                                  sample_size=sample_size, random_state=seed)

    from sklearn.metrics import silhouette_samples
    sil_samples = silhouette_samples(x_valid, labels_valid)
    per_dist = {}
    for dist_name in unique_dists:
        if dist_name == "":
            continue
        mask = labels_valid == label_to_int[dist_name]
        if mask.sum() > 0:
            per_dist[dist_name] = {
                "silhouette": float(np.mean(sil_samples[mask])),
                "n": int(mask.sum()),
            }

    return {
        "silhouette_dist_hd": sil_hd,
        "silhouette_dist_tsne": sil_tsne,
        "silhouette_dist_random": sil_random,
        "per_dist": per_dist,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_distribution(ax, tsne_2d, dist_labels, unique_dists, title,
                      show_legend=True, point_size=5):
    """Plot t-SNE colored by data distribution type."""
    for dist_name in unique_dists:
        if dist_name == "":
            continue
        mask = dist_labels == dist_name
        if mask.sum() == 0:
            continue
        color = DIST_COLORS.get(dist_name, "#333333")
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=color, s=point_size, alpha=0.5, label=dist_name,
                   rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if show_legend:
        ax.legend(fontsize=8, markerscale=3, loc="best", title="Distribution",
                  framealpha=0.8)


def plot_selectivity(ax, tsne_2d, y, title, point_size=5):
    """Plot t-SNE colored by continuous selectivity value."""
    sc = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                    c=y, cmap="viridis", s=point_size, alpha=0.5,
                    rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


# ---------------------------------------------------------------------------
# Task-specific pipeline
# ---------------------------------------------------------------------------

def process_task(data_dir, ae_configs, task, task_label):
    """Load, subsample, and run t-SNE for a set of AEs on a given task.

    Returns:
        tsne_results: dict ae_name -> tsne_2d
        data_cache: dict ae_name -> (y_sub, ds_sub, x_flat)
        available: list of (ae_name, desc) that were successfully loaded
    """
    tsne_results = {}
    data_cache = {}
    available = []

    task_prefix = "sj_sel" if task == "sj" else task

    for ae_name, desc in ae_configs:
        x_path = os.path.join(data_dir, f"x_{task_prefix}_{ae_name}.npy")
        if not os.path.exists(x_path):
            print(f"\n  {ae_name} ({task_label}): SKIPPED (data not found)")
            continue

        print(f"\n--- {ae_name}: {desc} ({task_label}) ---")
        x = np.load(x_path)
        y_path = os.path.join(data_dir, f"y_{task_prefix}_{ae_name}.npy")
        y = np.load(y_path)
        ds = _find_ds_file(data_dir, x.shape[0], task=task_prefix)
        print(f"  Loaded: x={x.shape}, y={y.shape}, "
              f"ds={'yes' if ds is not None else 'no'}")

        x_sub, y_sub, ds_sub = subsample(x, y, ds, N_SAMPLES, RANDOM_SEED)
        x_flat = x_sub.reshape(x_sub.shape[0], -1)
        print(f"  Subsampled: {x_flat.shape[0]} x {x_flat.shape[1]} features")

        tsne_2d = run_tsne(x_flat, RANDOM_SEED)
        key = f"{ae_name}_{task}"
        tsne_results[key] = tsne_2d
        data_cache[key] = (y_sub, ds_sub, x_flat)
        available.append((ae_name, desc, key))

    return tsne_results, data_cache, available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, output_dir, **kwargs):
    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION: Geographic Semantics in M1 Embeddings")
    print("=" * 60)

    all_rows = []

    # ===================================================================
    # Part 1: RQ embeddings (Table 3 & 4)
    # ===================================================================
    print("\n" + "-" * 40)
    print("PART 1: RQ Embeddings")
    print("-" * 40)

    rq_tsne, rq_cache, rq_available = process_task(
        data_dir, AE_CONFIGS_RQ, "rq", "RQ")

    # Identify AEs with distribution metadata
    rq_with_dist = []
    for ae_name, desc, key in rq_available:
        dist_labels, unique_dists = get_dist_labels(rq_cache[key][1], task="rq")
        if dist_labels is not None:
            rq_with_dist.append((ae_name, desc, key))

    # Figure 1: RQ distribution clustering
    if rq_with_dist:
        n = len(rq_with_dist)
        fig1, axes1 = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes1 = [axes1]
        fig1.suptitle(
            "t-SNE of M1 Latent Space — Colored by Geographic Distribution\n"
            "(Distinct clusters validate that M1 captures geographic semantics)",
            fontsize=13, fontweight="bold")

        for i, (ae_name, desc, key) in enumerate(rq_with_dist):
            ds_sub = rq_cache[key][1]
            x_flat = rq_cache[key][2]
            dist_labels, unique_dists = get_dist_labels(ds_sub, task="rq")
            tsne_2d = rq_tsne[key]

            metrics = compute_distribution_metrics(
                x_flat, tsne_2d, dist_labels, unique_dists, RANDOM_SEED)
            sil_hd = metrics.get("silhouette_dist_hd", float("nan"))

            plot_distribution(
                axes1[i], tsne_2d, dist_labels, unique_dists,
                f"{ae_name} — {desc}\n"
                f"Silhouette (high-dim): {sil_hd:.3f}")

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        path1 = os.path.join(output_dir, "tsne_distribution_rq.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {path1}")
        plt.close(fig1)

    # Figure 2: RQ distribution vs selectivity
    if rq_with_dist:
        n = len(rq_with_dist)
        fig2, axes2 = plt.subplots(2, n, figsize=(7 * n, 11))
        if n == 1:
            axes2 = axes2.reshape(2, 1)
        fig2.suptitle(
            "M1 Latent Space: Distribution Clustering vs. Selectivity Gradient",
            fontsize=13, fontweight="bold")

        for col, (ae_name, desc, key) in enumerate(rq_with_dist):
            ds_sub = rq_cache[key][1]
            y_sub = rq_cache[key][0]
            dist_labels, unique_dists = get_dist_labels(ds_sub, task="rq")
            tsne_2d = rq_tsne[key]

            plot_distribution(axes2[0, col], tsne_2d, dist_labels,
                              unique_dists, f"{ae_name} — Distribution",
                              point_size=4)
            sc = plot_selectivity(axes2[1, col], tsne_2d, y_sub,
                                  f"{ae_name} — RQ Selectivity", point_size=4)

        cbar = fig2.colorbar(sc, ax=axes2[1, :].tolist(), shrink=0.6,
                             pad=0.02)
        cbar.set_label("RQ Selectivity")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path2 = os.path.join(output_dir, "tsne_selectivity.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Saved: {path2}")
        plt.close(fig2)

    # RQ metrics
    for ae_name, desc, key in rq_available:
        tsne_2d = rq_tsne[key]
        y_sub, ds_sub, x_flat = rq_cache[key]
        dist_labels, unique_dists = get_dist_labels(ds_sub, task="rq")
        all_rows.append(_build_metric_row(
            ae_name, desc, "RQ", x_flat, tsne_2d, dist_labels, unique_dists))

    # ===================================================================
    # Part 2: SJ embeddings (Table 4 mixed data validation)
    # ===================================================================
    print("\n" + "-" * 40)
    print("PART 2: SJ Embeddings (Mixed Distributions)")
    print("-" * 40)

    sj_tsne, sj_cache, sj_available = process_task(
        data_dir, AE_CONFIGS_SJ, "sj", "SJ")

    sj_with_dist = []
    for ae_name, desc, key in sj_available:
        dist_labels, unique_dists = get_dist_labels(
            sj_cache[key][1], task="sj_sel")
        if dist_labels is not None:
            sj_with_dist.append((ae_name, desc, key))

    # Figure 3: SJ distribution clustering
    if sj_with_dist:
        n = len(sj_with_dist)
        fig3, axes3 = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes3 = [axes3]
        fig3.suptitle(
            "t-SNE of M1 Latent Space (SJ Embeddings) — "
            "Colored by Geographic Distribution\n"
            "(Validates M1 trained on synthetic+real captures "
            "both synthetic and real geographic semantics)",
            fontsize=13, fontweight="bold")

        for i, (ae_name, desc, key) in enumerate(sj_with_dist):
            ds_sub = sj_cache[key][1]
            x_flat = sj_cache[key][2]
            dist_labels, unique_dists = get_dist_labels(
                ds_sub, task="sj_sel")
            tsne_2d = sj_tsne[key]

            metrics = compute_distribution_metrics(
                x_flat, tsne_2d, dist_labels, unique_dists, RANDOM_SEED)
            sil_hd = metrics.get("silhouette_dist_hd", float("nan"))

            plot_distribution(
                axes3[i], tsne_2d, dist_labels, unique_dists,
                f"{ae_name} — {desc}\n"
                f"Silhouette (high-dim): {sil_hd:.3f}")

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        path3 = os.path.join(output_dir, "tsne_distribution_sj.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {path3}")
        plt.close(fig3)

    # SJ metrics
    for ae_name, desc, key in sj_available:
        tsne_2d = sj_tsne[key]
        y_sub, ds_sub, x_flat = sj_cache[key]
        dist_labels, unique_dists = get_dist_labels(ds_sub, task="sj_sel")
        all_rows.append(_build_metric_row(
            ae_name, desc, "SJ", x_flat, tsne_2d, dist_labels, unique_dists))

    # ===================================================================
    # Output CSV
    # ===================================================================
    print("\n" + "=" * 60)
    print("QUANTITATIVE METRICS — Distribution Clustering")
    print("=" * 60)

    for row in all_rows:
        ae = row["AE"]
        task = row["Task"]
        desc = row["Description"]
        if row.get("silhouette_dist_hd") not in (None, "N/A"):
            print(f"\n  {ae} ({desc}, {task}):")
            print(f"    silhouette (high-dim):  {row['silhouette_dist_hd']}")
            print(f"    silhouette (t-SNE 2D):  {row['silhouette_dist_tsne']}")
            print(f"    silhouette (random):    {row['silhouette_dist_random']}")
        else:
            print(f"\n  {ae} ({desc}, {task}): no distribution metadata")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, "tsne_viz.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    return df


def _build_metric_row(ae_name, desc, task, x_flat, tsne_2d,
                      dist_labels, unique_dists):
    """Build a metrics row dict for one AE + task combination."""
    row = {
        "AE": ae_name,
        "Description": desc,
        "Task": task,
        "N_samples": x_flat.shape[0],
        "Embedding_dim": x_flat.shape[1],
    }

    if dist_labels is not None and unique_dists is not None:
        metrics = compute_distribution_metrics(
            x_flat, tsne_2d, dist_labels, unique_dists, RANDOM_SEED)

        sil_hd = metrics.get("silhouette_dist_hd", float("nan"))
        sil_tsne = metrics.get("silhouette_dist_tsne", float("nan"))
        sil_rand = metrics.get("silhouette_dist_random", float("nan"))

        row["silhouette_dist_hd"] = round(sil_hd, 4)
        row["silhouette_dist_tsne"] = round(sil_tsne, 4)
        row["silhouette_dist_random"] = round(sil_rand, 4)

        per_dist = metrics.get("per_dist", {})
        for dname, dvals in sorted(per_dist.items()):
            row[f"sil_{dname}"] = round(dvals["silhouette"], 4)
    else:
        row["silhouette_dist_hd"] = "N/A"
        row["silhouette_dist_tsne"] = "N/A"
        row["silhouette_dist_random"] = "N/A"

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of M1 embeddings")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_dir, "data",
                                             "downloaded_data")
    output_dir = args.output_dir or os.path.join(project_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    run(data_dir, output_dir)
