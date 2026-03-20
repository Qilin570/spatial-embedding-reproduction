"""t-SNE visualization: compare M1 embedding quality across autoencoders.

Compares 4 AEs on RQ embeddings:
  AE_S1: Stacked, LD=384,  Synthetic only
  AE_C2: CNN,     LD=3072, Synthetic only
  AE_S3: Stacked, LD=48,   Synthetic + Real
  AE_S4: Stacked, LD=384,  Synthetic + Real

Generates:
  1. tsne_selectivity.png   — 2x4 grid, continuous + binned selectivity
  2. tsne_distribution.png  — per-distribution clustering (synthetic AEs)
  3. tsne_intra_dist.png    — within-distribution selectivity gradient analysis
  4. tsne_viz.csv           — quantitative metrics for all AEs + random baseline

Metrics:
  - dist_corr: Spearman correlation between embedding distance and selectivity
    difference (higher = embedding better encodes selectivity)
  - silhouette_sel: silhouette score using selectivity bins as labels
    (higher = selectivity groups better separated in embedding space)
  - dist_corr_random: same metric on shuffled embeddings (baseline)

Usage:
    python run_all.py --tables 99
    python -m experiments.tsne_viz --data-dir ./data/downloaded_data
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_ds_file(data_dir, n_samples):
    """Search subdirectories for a ds_*_rq*.npy matching n_samples."""
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
    """Load RQ embeddings, targets, and optional distribution metadata."""
    x = np.load(os.path.join(data_dir, f"x_rq_{ae_name}.npy"))
    y = np.load(os.path.join(data_dir, f"y_rq_{ae_name}.npy"))
    ds = _find_ds_file(data_dir, x.shape[0])
    return x, y, ds


def subsample(x, y, ds, n_samples, seed):
    """Randomly subsample for t-SNE."""
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    if n <= n_samples:
        return x, y, ds
    idx = rng.choice(n, size=n_samples, replace=False)
    return x[idx], y[idx], ds[idx] if ds is not None else None


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

def compute_metrics(tsne_2d, y, seed):
    """Compute distance correlation, silhouette score, and random baseline."""
    # Spearman correlation: embedding distance vs selectivity difference
    dists_emb = pdist(tsne_2d)
    dists_y = pdist(y.reshape(-1, 1))
    dist_corr, _ = spearmanr(dists_emb, dists_y)

    # Silhouette score on selectivity quintile bins
    bin_labels, _ = bin_selectivity(y)
    sil = silhouette_score(tsne_2d, bin_labels, sample_size=min(2000, len(y)),
                           random_state=seed)

    # Random baseline: shuffle embedding rows then recompute
    rng = np.random.RandomState(seed + 1)
    shuffled = tsne_2d[rng.permutation(len(tsne_2d))]
    dists_rand = pdist(shuffled)
    dist_corr_rand, _ = spearmanr(dists_rand, dists_y)
    sil_rand = silhouette_score(shuffled, bin_labels,
                                sample_size=min(2000, len(y)),
                                random_state=seed)

    return {
        "dist_corr": dist_corr,
        "silhouette_sel": sil,
        "dist_corr_random": dist_corr_rand,
        "silhouette_random": sil_rand,
    }


def compute_per_dist_corr(tsne_2d, y, ds):
    """Per-distribution selectivity-distance correlation."""
    if ds is None:
        return {}
    dist_col = ds[:, 1]
    results = {}
    for dist_name in sorted(set(dist_col)):
        mask = dist_col == dist_name
        if mask.sum() < 30:
            continue
        sub_emb = tsne_2d[mask]
        sub_y = y[mask]
        if sub_y.std() < 1e-9:
            continue
        d_emb = pdist(sub_emb)
        d_y = pdist(sub_y.reshape(-1, 1))
        corr, _ = spearmanr(d_emb, d_y)
        results[dist_name] = {"corr": corr, "n": int(mask.sum())}
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_continuous(ax, tsne_2d, y, title):
    sc = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                    c=y, cmap="viridis", s=3, alpha=0.5, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def plot_bins(ax, tsne_2d, bin_idx, bin_names, title):
    cmap = plt.cm.get_cmap("tab10", len(bin_names))
    for i, name in enumerate(bin_names):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=[cmap(i)], s=3, alpha=0.5, label=name, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=3, loc="best", title="Selectivity")


def plot_distribution(ax, tsne_2d, ds, title):
    dist_col = ds[:, 1]
    unique_dists = sorted(set(dist_col))
    cmap = plt.cm.get_cmap("Set2", len(unique_dists))
    for i, dist in enumerate(unique_dists):
        mask = dist_col == dist
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=[cmap(i)], s=3, alpha=0.5, label=dist, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=3, loc="best", title="Distribution")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, output_dir, **kwargs):
    print("\n" + "=" * 60)
    print("t-SNE VISUALIZATION: M1 Embedding Quality Comparison")
    print("=" * 60)

    # --- Load, subsample, run t-SNE for each AE ---
    tsne_results = {}
    data_cache = {}
    available = []

    for ae_name, desc in AE_CONFIGS:
        x_path = os.path.join(data_dir, f"x_rq_{ae_name}.npy")
        if not os.path.exists(x_path):
            print(f"\n  {ae_name}: SKIPPED (data not found)")
            continue

        print(f"\n--- {ae_name}: {desc} ---")
        x, y, ds = load_rq_data(data_dir, ae_name)
        print(f"  Loaded: x={x.shape}, y={y.shape}, ds={'yes' if ds is not None else 'no'}")

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

    n_ae = len(available)

    # =======================================================================
    # Figure 1: Selectivity comparison (2 rows x n_ae cols)
    # =======================================================================
    fig, axes = plt.subplots(2, n_ae, figsize=(5 * n_ae, 9))
    if n_ae == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("t-SNE of M1 Embeddings — Colored by RQ Selectivity", fontsize=14)

    for col, (ae_name, desc) in enumerate(available):
        tsne_2d = tsne_results[ae_name]
        y_sub = data_cache[ae_name][0]
        sc = plot_continuous(axes[0, col], tsne_2d, y_sub,
                             f"{ae_name}\n{desc}")
        bin_idx, bin_names = bin_selectivity(y_sub)
        plot_bins(axes[1, col], tsne_2d, bin_idx, bin_names,
                  f"{ae_name} — Selectivity Bins")

    cbar = fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("RQ Selectivity")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(output_dir, "tsne_selectivity.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path1}")
    plt.close(fig)

    # =======================================================================
    # Figure 2: Distribution clustering (only AEs with distribution metadata)
    # =======================================================================
    ae_with_dist = [(n, d) for n, d in available
                    if data_cache[n][1] is not None
                    and len(set(data_cache[n][1][:, 1])) > 1]

    if ae_with_dist:
        n_dist = len(ae_with_dist)
        fig2, axes2 = plt.subplots(1, n_dist, figsize=(7 * n_dist, 5))
        if n_dist == 1:
            axes2 = [axes2]
        fig2.suptitle("t-SNE Colored by Data Distribution", fontsize=14)

        for i, (ae_name, desc) in enumerate(ae_with_dist):
            plot_distribution(axes2[i], tsne_results[ae_name],
                              data_cache[ae_name][1],
                              f"{ae_name} — {desc}")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path2 = os.path.join(output_dir, "tsne_distribution.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Saved: {path2}")
        plt.close(fig2)

    # =======================================================================
    # Figure 3: Within-distribution selectivity gradient (AEs with dist data)
    # =======================================================================
    if ae_with_dist:
        # Pick the first AE with distribution info for detailed analysis
        ae_name, desc = ae_with_dist[0]
        tsne_2d = tsne_results[ae_name]
        y_sub, ds_sub, _ = data_cache[ae_name]
        dists = sorted(set(ds_sub[:, 1]))

        n_dists = len(dists)
        ncols = min(3, n_dists)
        nrows = (n_dists + ncols - 1) // ncols
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes3 = np.atleast_2d(axes3)
        fig3.suptitle(
            f"{ae_name}: Within-Distribution Selectivity Gradient", fontsize=14)

        for idx, dist_name in enumerate(dists):
            r, c = divmod(idx, ncols)
            ax = axes3[r, c]
            mask = ds_sub[:, 1] == dist_name
            sc = ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                            c=y_sub[mask], cmap="viridis", s=6, alpha=0.6,
                            rasterized=True)
            n_pts = mask.sum()
            # Per-dist correlation
            if n_pts >= 30 and y_sub[mask].std() > 1e-9:
                d_emb = pdist(tsne_2d[mask])
                d_y = pdist(y_sub[mask].reshape(-1, 1))
                corr, _ = spearmanr(d_emb, d_y)
                ax.set_title(f"{dist_name} (n={n_pts})\ncorr={corr:.3f}",
                             fontsize=10)
            else:
                ax.set_title(f"{dist_name} (n={n_pts})", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig3.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)

        # Hide unused axes
        for idx in range(n_dists, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes3[r, c].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path3 = os.path.join(output_dir, "tsne_intra_dist.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        print(f"Saved: {path3}")
        plt.close(fig3)

    # =======================================================================
    # Quantitative metrics + CSV
    # =======================================================================
    print("\n" + "=" * 60)
    print("QUANTITATIVE METRICS")
    print("=" * 60)

    rows = []
    for ae_name, desc in available:
        tsne_2d = tsne_results[ae_name]
        y_sub, ds_sub, _ = data_cache[ae_name]

        metrics = compute_metrics(tsne_2d, y_sub, RANDOM_SEED)
        per_dist = compute_per_dist_corr(tsne_2d, y_sub, ds_sub)

        print(f"\n  {ae_name} ({desc}):")
        print(f"    dist_corr (emb-sel):   {metrics['dist_corr']:.4f}  "
              f"(random: {metrics['dist_corr_random']:.4f})")
        print(f"    silhouette (sel bins):  {metrics['silhouette_sel']:.4f}  "
              f"(random: {metrics['silhouette_random']:.4f})")

        if per_dist:
            print(f"    per-distribution corr:")
            for dname, dvals in per_dist.items():
                print(f"      {dname:12s}: corr={dvals['corr']:.4f}  (n={dvals['n']})")

        row = {
            "AE": ae_name,
            "Description": desc,
            "N_samples": N_SAMPLES,
            "dist_corr": round(metrics["dist_corr"], 4),
            "silhouette_sel": round(metrics["silhouette_sel"], 4),
            "dist_corr_random": round(metrics["dist_corr_random"], 4),
            "silhouette_random": round(metrics["silhouette_random"], 4),
        }
        # Add per-distribution correlations as columns
        for dname, dvals in per_dist.items():
            row[f"corr_{dname}"] = round(dvals["corr"], 4)
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
    data_dir = args.data_dir or os.path.join(project_dir, "data", "downloaded_data")
    output_dir = args.output_dir or os.path.join(project_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    run(data_dir, output_dir)
