"""Download data from Mendeley Data for spatial embedding experiments.

Dataset DOI: 10.17632/zp9fh6scw9.2
"""
import os
import sys
import gzip
import shutil
import tarfile
import zipfile

try:
    import requests
except ImportError:
    requests = None


# Mendeley direct download URLs (from READMEs in the original repo)
MENDELEY_URLS = {
    # Pre-trained AE models (.gzip archives)
    "autoencoder_CNN_128-64_emb768_synthetic": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/28377351-f89d-4beb-b1a4-1d7f1caada7f/file_downloaded",
    "autoencoder_CNN3L_128-64_emb1536_real": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/7e2dc6ae-5b5e-4575-a8a6-3658f5cf1d45/file_downloaded",
    "autoencoder_DENSE3L_1024-512_emb384_synthetic": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/c2b239ba-d8bb-4acd-ad92-95466f8decb3/file_downloaded",
    "autoencoder_DENSE3L_1024-512_emb1536_synthetic": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/6a408d97-a21b-4d5e-a530-58699e43a7e1/file_downloaded",
    "autoencoder_DENSE3L_16-32_emb384_real": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/2f7665da-3d3f-470e-b4a6-cd8103614f54/file_downloaded",
    "model_3072_CNNDense_newDatasets_SMLG": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/8af06b8b-d0a6-4b3c-a506-88606eca852e/file_downloaded",

    # RQ training sets
    "RQ_AE_c2": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/88c238a3-32fa-4f93-85ba-436a7b205c0b/file_downloaded",
    "RQ_AE_s1": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/f40872be-feb3-424b-8297-6f7394de2f5e/file_downloaded",
    "RQ_AE_s3": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/2794e9fd-1422-405d-aba3-7490ebd2462d/file_downloaded",
    "RQ_AE_s4": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/c4d74a80-8337-43f0-bbdf-c006f0ebbdf6/file_downloaded",

    # SJ training sets (selectivity)
    "SJ_AE_c2_sel": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/3f2501f3-67c0-4e18-a111-6775fe740f20/file_downloaded",
    "SJ_AE_s4_sel": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/2c2ae0bf-9759-4791-9615-cff4055b09c1/file_downloaded",
    "SJ_AE_c3_sel": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/e8199859-8f22-46f8-998c-0d787aa7989c/file_downloaded",

    # SJ training sets (MBR tests)
    "SJ_AE_c2_mbr": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/6b724714-ea0b-4824-b038-0cdb3a8e989a/file_downloaded",
    "SJ_AE_s4_mbr": "https://data.mendeley.com/public-files/datasets/zp9fh6scw9/files/f75d28f5-c07d-4b92-8624-7ef3a5031df9/file_downloaded",
}

SPATIAL_EMB_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spatial-embedding"
)


def download_file(url, output_path, desc=""):
    """Download a file from URL with progress display."""
    if requests is None:
        print(f"  ERROR: 'requests' module not installed. Run: pip install requests")
        return False

    print(f"  Downloading {desc or os.path.basename(output_path)}...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  [{pct:3d}%] {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR downloading: {e}")
        return False


def extract_archive(archive_path, output_dir):
    """Extract .gzip, .zip, or .tar.gz archive."""
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(output_dir)
        return True
    elif archive_path.endswith('.gzip') or archive_path.endswith('.gz'):
        # Try as tar.gz first
        try:
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(output_dir)
            return True
        except tarfile.TarError:
            pass
        # Try as plain gzip
        try:
            out_file = archive_path.rsplit('.', 1)[0]
            with gzip.open(archive_path, 'rb') as f_in:
                with open(out_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception:
            pass
    return False


def find_existing_data(spatial_emb_dir=None):
    """Search for existing data in the spatial-embedding directory."""
    if spatial_emb_dir is None:
        spatial_emb_dir = SPATIAL_EMB_BASE

    found = {}
    if not os.path.exists(spatial_emb_dir):
        return found

    # Trained models
    trained_model_dir = os.path.join(spatial_emb_dir, "autoEncoders", "trainedModels")
    if os.path.exists(trained_model_dir):
        for item in os.listdir(trained_model_dir):
            full_path = os.path.join(trained_model_dir, item)
            if os.path.isdir(full_path):
                found[f"model:{item}"] = full_path
            elif item.endswith('.gzip'):
                found[f"archive:{item}"] = full_path

    # Histogram .npy files
    hist_tset_dir = os.path.join(spatial_emb_dir, "autoEncoders", "generatedTSet")
    if os.path.exists(hist_tset_dir):
        for item in os.listdir(hist_tset_dir):
            if item.endswith('.npy'):
                found[f"hist:{item}"] = os.path.join(hist_tset_dir, item)

    # Summary CSVs
    summaries_dir = os.path.join(spatial_emb_dir, "summaries")
    if os.path.exists(summaries_dir):
        for item in os.listdir(summaries_dir):
            if item.endswith('.csv'):
                found[f"csv:{item}"] = os.path.join(summaries_dir, item)

    # RQ generated training sets
    rq_tset_dir = os.path.join(spatial_emb_dir, "modelsRQ", "generatedTSet")
    if os.path.exists(rq_tset_dir):
        for root, dirs, files in os.walk(rq_tset_dir):
            for f in files:
                if f.endswith('.npy'):
                    found[f"rq_npy:{f}"] = os.path.join(root, f)
                elif f.endswith('.zip'):
                    found[f"rq_zip:{f}"] = os.path.join(root, f)

    # SJ generated training sets
    sj_tset_dir = os.path.join(spatial_emb_dir, "modelsSJ", "generatedTSet")
    if os.path.exists(sj_tset_dir):
        for root, dirs, files in os.walk(sj_tset_dir):
            for f in files:
                if f.endswith('.npy'):
                    found[f"sj_npy:{f}"] = os.path.join(root, f)
                elif f.endswith('.zip'):
                    found[f"sj_zip:{f}"] = os.path.join(root, f)

    return found


def link_local_data(output_dir, spatial_emb_dir=None):
    """Create symlinks to existing local data."""
    os.makedirs(output_dir, exist_ok=True)
    found = find_existing_data(spatial_emb_dir)
    linked = 0

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    for key, path in found.items():
        category, name = key.split(":", 1)

        if category == "model":
            target = os.path.join(model_dir, name)
        elif category == "archive":
            # Extract gzip archives to model dir
            target = os.path.join(model_dir, name)
            if not os.path.exists(target):
                shutil.copy2(path, target)
                model_name = name.replace('.gzip', '')
                model_target = os.path.join(model_dir, model_name)
                if not os.path.exists(model_target):
                    extract_archive(target, model_dir)
                print(f"  Extracted: {name}")
                linked += 1
            continue
        elif category == "hist":
            # Map histogram files
            if "synt" in name:
                target = os.path.join(output_dir, "histograms_synthetic.npy")
            elif "real" in name:
                target = os.path.join(output_dir, "histograms_real.npy")
            else:
                target = os.path.join(output_dir, name)
        elif category == "csv":
            target = os.path.join(output_dir, name)
        elif category in ("rq_npy", "sj_npy"):
            target = os.path.join(output_dir, name)
        elif category in ("rq_zip", "sj_zip"):
            target = os.path.join(output_dir, name)
        else:
            continue

        if not os.path.exists(target):
            try:
                os.symlink(path, target)
                print(f"  Linked: {name} -> {os.path.basename(target)}")
                linked += 1
            except OSError:
                shutil.copy2(path, target) if not os.path.isdir(path) else shutil.copytree(path, target)
                print(f"  Copied: {name}")
                linked += 1

    print(f"\n  Linked/copied {linked} items to {output_dir}")
    return found


def download_from_mendeley(output_dir):
    """Download data files from Mendeley Data."""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n  Downloading from Mendeley Data (DOI: 10.17632/zp9fh6scw9.2)")
    print(f"  Target directory: {output_dir}\n")

    downloaded = 0
    for name, url in MENDELEY_URLS.items():
        # Determine output path and check if already exists
        if name.startswith("autoencoder_") or name.startswith("model_"):
            archive_path = os.path.join(model_dir, f"{name}.gzip")
            model_path = os.path.join(model_dir, name)
            if os.path.exists(model_path):
                print(f"  Already exists: {name}")
                continue
            if download_file(url, archive_path, name):
                extract_archive(archive_path, model_dir)
                downloaded += 1
        elif name.startswith("RQ_") or name.startswith("SJ_"):
            archive_path = os.path.join(output_dir, f"{name}.zip")
            if os.path.exists(archive_path):
                print(f"  Already exists: {name}")
                continue
            if download_file(url, archive_path, name):
                extract_archive(archive_path, output_dir)
                downloaded += 1

    print(f"\n  Downloaded {downloaded} files")
    if downloaded == 0 and requests is None:
        print("\n  To enable automatic download: pip install requests")
        print("  Or download manually from: https://data.mendeley.com/datasets/zp9fh6scw9/2")


def download_data(output_dir=None, spatial_emb_dir=None):
    """Main entry point for data preparation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")

    print("=" * 60)
    print("Data Preparation")
    print("=" * 60)

    # Step 1: Link local data
    print("\nStep 1: Searching for existing local data...")
    found = find_existing_data(spatial_emb_dir)
    if found:
        print(f"  Found {len(found)} items locally")
        print("\nStep 2: Linking local data...")
        link_local_data(output_dir, spatial_emb_dir)
    else:
        print("  No existing data found locally")

    # Step 2: Download missing from Mendeley
    print("\nStep 3: Checking for missing data to download...")
    download_from_mendeley(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    _print_data_summary(output_dir)

    return output_dir


def _print_data_summary(output_dir):
    """Print summary of available data."""
    categories = {
        "Models": [],
        "Histograms": [],
        "Training sets": [],
        "CSV files": [],
    }

    model_dir = os.path.join(output_dir, "model")
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if not item.endswith('.gzip'):
                categories["Models"].append(item)

    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        if item.startswith("histograms") and item.endswith(".npy"):
            categories["Histograms"].append(item)
        elif item.endswith('.npy'):
            categories["Training sets"].append(item)
        elif item.endswith('.csv'):
            categories["CSV files"].append(item)
        elif os.path.isdir(path) and item not in ("model",):
            n_npy = sum(1 for f in os.listdir(path) if f.endswith('.npy')) if os.path.isdir(path) else 0
            if n_npy > 0:
                categories["Training sets"].append(f"{item}/ ({n_npy} .npy files)")

    for cat, items in categories.items():
        if items:
            print(f"\n  {cat}:")
            for item in sorted(items):
                print(f"    - {item}")
        else:
            print(f"\n  {cat}: (none found)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download/link data for spatial embedding experiments")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--spatial-emb-dir", default=None, help="Path to existing spatial-embedding dir")
    args = parser.parse_args()
    download_data(args.output_dir, args.spatial_emb_dir)
