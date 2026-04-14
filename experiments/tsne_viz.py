"""t-SNE visualization: validate that M1 captures geographic semantics.

Applies t-SNE dimensionality reduction to M1 latent representations to
visually validate that the autoencoder successfully captures geographic
semantics and produces distinct clustering patterns by data distribution.

Compares AEs from Table 3 (synthetic) and Table 4 (synthetic+real):
  AE_S1: Stacked, LD=384,  Synthetic only
  AE_C2: CNN,     LD=3072, Synthetic only
  AE_S3: Stacked, LD=48,   Synthetic + Real
  AE_S4: Stacked, LD=384,  Synthetic + Real

Generates:
  1. tsne_distribution.png  — primary: distribution clustering (geographic semantics)
  2. tsne_selectivity.png   — secondary: selectivity gradient within clusters
  3. tsne_viz.csv           — quantitative clustering metrics

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

AE_CONFIGS = [
    ("AE_S1", "Stacked, LD=384, Synthetic"),
    ("AE_C2", "CNN, LD=3072, Synthetic"),
    ("AE_S3", "Stacked, LD=48, Synth+Real"),
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
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_ds_file(data_dir, ae_name, n_samples):
    """Find distribution metadata file for a given AE and sample count.

    Searches subdirectories for ds_*_rq*.npy or y_*_distr.npy files
    matching the expected sample count.
    """
    for subdir in os.listdir(data_dir):
        subpath = os.path.join(data_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for f in os.listdir(subpath):
            if not (f.startswith("ds_") and "rq" in f and f.endswith(".npy")):
                continue
            path = os.path.join(subpath, f)
            loaded = np.load(path, allow_pickle=True)
            if loaded.shape[0] == n_samples:
                return loaded
        # Also check y_*_distr.npy (AE_S1 format)
        for f in os.listdir(subpath):
            if "distr" in f and f.endswith(".npy"):
                path = os.path.join(subpath, f)
                loaded = np.load(path, allow_pickle=True)
                if loaded.shape[0] == n_samples and loaded.ndim == 2:
                    return loaded
    return None


def load_rq_data(data_dir, ae_name):
    """Load RQ embeddings, targets, and distribution metadata."""
    x = np.load(os.path.join(data_dir, f"x_rq_{ae_name}.npy"))
    y = np.load(os.path.join(data_dir, f"y_rq_{ae_name}.npy"))
    ds = _find_ds_file(data_dir, ae_name, x.shape[0])
    return x, y, ds


def subsample(x, y, ds, n_samples, seed):
    """Randomly subsample for t-SNE."""
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    if n <= n_samples:
        return x, y, ds
    idx = rng.choice(n, size=n_samples, replace=False)
    return x[idx], y[idx], ds[idx] if ds is not None else None


def get_dist_labels(ds):
    """Extract distribution type labels from metadata array.

    Returns labels array and sorted unique label names, or (None, None)
    if distribution metadata is unavailable or has only one category.
    """
    if ds is None:
        return None, None
    dist_col = ds[:, 1]
    # Filter out empty strings
    unique = sorted(set(d for d in dist_col if d))
    if len(unique) <= 1:
        return None, None
    return dist_col, unique


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
    """Compute distribution clustering metrics in both high-dim and t-SNE space.

    Args:
        x_flat: original high-dimensional embeddings (N, D)
        tsne_2d: t-SNE reduced embeddings (N, 2)
        dist_labels: distribution type label for each sample
        unique_dists: sorted list of unique distribution names
        seed: random seed for reproducibility

    Returns:
        dict with silhouette scores in high-dim, t-SNE, and random baseline
    """
    # Encode string labels to integers, filtering out empty labels
    valid_mask = np.array([d in unique_dists and d != "" for d in dist_labels])
    if valid_mask.sum() < 50:
        return {}

    label_to_int = {name: i for i, name in enumerate(unique_dists)}
    int_labels = np.array([label_to_int.get(d, -1) for d in dist_labels])

    x_valid = x_flat[valid_mask]
    tsne_valid = tsne_2d[valid_mask]
    labels_valid = int_labels[valid_mask]

    sample_size = min(2000, len(labels_valid))

    # Silhouette in original high-dimensional space
    sil_hd = silhouette_score(x_valid, labels_valid,
                              sample_size=sample_size, random_state=seed)

    # Silhouette in t-SNE 2D space
    sil_tsne = silhouette_score(tsne_valid, labels_valid,
                                sample_size=sample_size, random_state=seed)

    # Random baseline: shuffle labels
    rng = np.random.RandomState(seed + 1)
    shuffled_labels = labels_valid[rng.permutation(len(labels_valid))]
    sil_random = silhouette_score(x_valid, shuffled_labels,
                                  sample_size=sample_size, random_state=seed)

    # Per-distribution silhouette (how well each distribution clusters)
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
# Main
# ---------------------------------------------------------------------------

def run(data_dir, output_dir, **kwargs):
    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION: Geographic Semantics in M1 Embeddings")
    print("=" * 60)

    # --- Load, subsample, run t-SNE for each AE ---
    tsne_results = {}
    data_cache = {}  # ae_name -> (y_sub, ds_sub, x_flat)
    available = []

    for ae_name, desc in AE_CONFIGS:
        x_path = os.path.join(data_dir, f"x_rq_{ae_name}.npy")
        if not os.path.exists(x_path):
            print(f"\n  {ae_name}: SKIPPED (data not found)")
            continue

        print(f"\n--- {ae_name}: {desc} ---")
        x, y, ds = load_rq_data(data_dir, ae_name)
        print(f"  Loaded: x={x.shape}, y={y.shape}, "
              f"ds={'yes' if ds is not None else 'no'}")

        x_sub, y_sub, ds_sub = subsample(x, y, ds, N_SAMPLES, RANDOM_SEED)
        x_flat = x_sub.reshape(x_sub.shape[0], -1)
        print(f"  Subsampled: {x_flat.shape[0]} x {x_flat.shape[1]} features")

        tsne_2d = run_tsne(x_flat, RANDOM_SEED)
        tsne_results[ae_name] = tsne_2d
        data_cache[ae_name] = (y_sub, ds_sub, x_flat)
        available.append((ae_name, desc))

    if not available:
        print("ERROR: no data found")
        return pd.DataFrame()

    # Identify AEs with distribution metadata (>1 distribution type)
    ae_with_dist = []
    for ae_name, desc in available:
        dist_labels, unique_dists = get_dist_labels(data_cache[ae_name][1])
        if dist_labels is not None:
            ae_with_dist.append((ae_name, desc))

    # =======================================================================
    # Figure 1 (PRIMARY): Distribution clustering — geographic semantics
    # =======================================================================
    if ae_with_dist:
        n_dist_ae = len(ae_with_dist)
        fig1, axes1 = plt.subplots(1, n_dist_ae,
                                   figsize=(7 * n_dist_ae, 6))
        if n_dist_ae == 1:
            axes1 = [axes1]
        fig1.suptitle(
            "t-SNE of M1 Latent Space — Colored by Geographic Distribution\n"
            "(Distinct clusters validate that M1 captures geographic semantics)",
            fontsize=13, fontweight="bold")

        for i, (ae_name, desc) in enumerate(ae_with_dist):
            ds_sub = data_cache[ae_name][1]
            x_flat = data_cache[ae_name][2]
            dist_labels, unique_dists = get_dist_labels(ds_sub)
            tsne_2d = tsne_results[ae_name]

            # Compute and display silhouette in title
            metrics = compute_distribution_metrics(
                x_flat, tsne_2d, dist_labels, unique_dists, RANDOM_SEED)
            sil_hd = metrics.get("silhouette_dist_hd", float("nan"))

            plot_distribution(
                axes1[i], tsne_2d, dist_labels, unique_dists,
                f"{ae_name} — {desc}\n"
                f"Silhouette (high-dim): {sil_hd:.3f}")

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        path1 = os.path.join(output_dir, "tsne_distribution.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {path1}")
        plt.close(fig1)

    # =======================================================================
    # Figure 2 (SECONDARY): Selectivity gradient overlay
    # Shows that within each distribution cluster, selectivity varies
    # =======================================================================
    if ae_with_dist:
        n_dist_ae = len(ae_with_dist)
        fig2, axes2 = plt.subplots(2, n_dist_ae,
                                   figsize=(7 * n_dist_ae, 11))
        if n_dist_ae == 1:
            axes2 = axes2.reshape(2, 1)
        fig2.suptitle(
            "M1 Latent Space: Distribution Clustering vs. Selectivity Gradient",
            fontsize=13, fontweight="bold")

        for col, (ae_name, desc) in enumerate(ae_with_dist):
            ds_sub = data_cache[ae_name][1]
            y_sub = data_cache[ae_name][0]
            dist_labels, unique_dists = get_dist_labels(ds_sub)
            tsne_2d = tsne_results[ae_name]

            # Row 0: distribution clustering
            plot_distribution(axes2[0, col], tsne_2d, dist_labels,
                              unique_dists, f"{ae_name} — Distribution",
                              point_size=4)

            # Row 1: selectivity gradient
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

    # =======================================================================
    # Quantitative metrics + CSV
    # =======================================================================
    print("\n" + "=" * 60)
    print("QUANTITATIVE METRICS — Distribution Clustering")
    print("=" * 60)

    rows = []
    for ae_name, desc in available:
        tsne_2d = tsne_results[ae_name]
        y_sub, ds_sub, x_flat = data_cache[ae_name]
        dist_labels, unique_dists = get_dist_labels(ds_sub)

        row = {
            "AE": ae_name,
            "Description": desc,
            "N_samples": x_flat.shape[0],
            "Embedding_dim": x_flat.shape[1],
        }

        if dist_labels is not None:
            metrics = compute_distribution_metrics(
                x_flat, tsne_2d, dist_labels, unique_dists, RANDOM_SEED)

            sil_hd = metrics.get("silhouette_dist_hd", float("nan"))
            sil_tsne = metrics.get("silhouette_dist_tsne", float("nan"))
            sil_rand = metrics.get("silhouette_dist_random", float("nan"))

            print(f"\n  {ae_name} ({desc}):")
            print(f"    silhouette (high-dim):  {sil_hd:.4f}")
            print(f"    silhouette (t-SNE 2D):  {sil_tsne:.4f}")
            print(f"    silhouette (random):    {sil_rand:.4f}")

            row["silhouette_dist_hd"] = round(sil_hd, 4)
            row["silhouette_dist_tsne"] = round(sil_tsne, 4)
            row["silhouette_dist_random"] = round(sil_rand, 4)

            per_dist = metrics.get("per_dist", {})
            if per_dist:
                print(f"    per-distribution silhouette (high-dim):")
                for dname, dvals in sorted(per_dist.items()):
                    print(f"      {dname:12s}: {dvals['silhouette']:.4f}  "
                          f"(n={dvals['n']})")
                    row[f"sil_{dname}"] = round(dvals["silhouette"], 4)
        else:
            print(f"\n  {ae_name} ({desc}): no distribution metadata")
            row["silhouette_dist_hd"] = "N/A"
            row["silhouette_dist_tsne"] = "N/A"
            row["silhouette_dist_random"] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "tsne_viz.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    return df


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
