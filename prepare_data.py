#!/usr/bin/env python3
"""Prepare downloaded data: extract zips and create symlinks with standardized names.

The Mendeley data uses original naming conventions (e.g., x_63410_rq_2_emb1.npy)
but our experiment scripts expect standardized names (e.g., x_rq_AE_S1.npy).

This script:
1. Extracts all .zip archives
2. Creates symlinks mapping original names -> expected names
3. Maps pre-trained model directories to expected names

Usage:
    python3 prepare_data.py [--data-dir ./downloaded_data]
"""
import os
import sys
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")

# Mapping: zip name -> (subfolder in zip, AE name, task type)
# The zip archives contain subdirectories with the actual .npy files
ZIP_MAPPINGS = {
    # RQ training sets
    "RQ_AE_s1": {
        "subfolder": "emb1_bal_synt",
        "ae_name": "AE_S1",
        "task": "rq",
        "x_pattern": "x_*_rq_*_emb*.npy",
        "x1_pattern": "x1_*_rq_*.npy",
        "y_pattern": "y_*_rq_0.npy",
    },
    "RQ_AE_c2": {
        "subfolder": "emb0_bal_synt",
        "ae_name": "AE_C2",
        "task": "rq",
        "x_pattern": "x_*_rq_*_emb*.npy",
        "x1_pattern": "x1_*_rq*.npy",
        "y_pattern": "y_*_rq*_0.npy",
    },
    "RQ_AE_s3": {
        "subfolder": "emb2_real_bal_real",
        "ae_name": "AE_S3",
        "task": "rq",
        "x_pattern": "x_*_rq_*_emb*.npy",
        "x1_pattern": "x1_*_rq_*.npy",
        "y_pattern": "y_*_rq_0.npy",
    },
    "RQ_AE_s4": {
        "subfolder": "emb1_real_bal_real",
        "ae_name": "AE_S4",
        "task": "rq",
        "x_pattern": "x_*_rq_*_emb*.npy",
        "x1_pattern": "x1_*_rq_*.npy",
        "y_pattern": "y_*_rq_0.npy",
    },
    # SJ selectivity training sets
    "SJ_AE_c2_sel": {
        "subfolder": "emb0_bal_synt_sel",
        "ae_name": "AE_C2",
        "task": "sj_sel",
        "x_pattern": "x_*_jn_*_emb*.npy",
        "x1_pattern": "x1_*_jn_*.npy",
        "y_pattern": "y_*_jn_0.npy",
    },
    "SJ_AE_s4_sel": {
        "subfolder": "emb1_real_bal_real_sel",
        "ae_name": "AE_S4",
        "task": "sj_sel",
        "x_pattern": "x_*_jn_*_emb*.npy",
        "x1_pattern": "x1_*_jn_*.npy",
        "y_pattern": "y_*_jn_0*.npy",
    },
    "SJ_AE_c3_sel": {
        "subfolder": "emb3_real_bal_real_sel",
        "ae_name": "AE_C3",
        "task": "sj_sel",
        "x_pattern": "x_*_jn_*_emb*.npy",
        "x1_pattern": "x1_*_jn_*.npy",
        "y_pattern": "y_*_jn_0.npy",
    },
    # SJ MBR test training sets
    "SJ_AE_c2_mbr": {
        "subfolder": "emb0_bal_synt_mbrtest",
        "ae_name": "AE_C2",
        "task": "sj_mbr",
        "x_pattern": "x_*_jn_*_emb*.npy",
        "x1_pattern": "x1_*_jn_*.npy",
        "y_pattern": "y_*_jn_3.npy",
    },
    "SJ_AE_s4_mbr": {
        "subfolder": "emb1_real_bal_real_mbrtest",
        "ae_name": "AE_S4",
        "task": "sj_mbr",
        "x_pattern": "x_*_jn_*_emb*.npy",
        "x1_pattern": "x1_*_jn_*.npy",
        "y_pattern": "y_*_jn_3*.npy",
    },
}

# Model directory mapping: original name -> AE config name
MODEL_DIR_MAPPING = {
    "autoencoder_DENSE3L_1024-512_emb384_synthetic": "AE_S1",
    "autoencoder_DENSE3L_1024-512_emb1536_synthetic": "AE_S2",
    "autoencoder_CNN_128-64_emb768_synthetic": "AE_C1",
    "model_3072_CNNDense_newDatasets_SMLG": "AE_C2",
    "autoencoder_DENSE3L_16-32_emb384_real": "AE_S4",
    "autoencoder_CNN3L_128-64_emb1536_real": "AE_C3",
}


def extract_zips(data_dir):
    """Extract all .zip files that haven't been extracted yet."""
    extracted = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith('.zip'):
            continue
        zip_path = os.path.join(data_dir, fname)
        zip_name = fname.replace('.zip', '')

        # Check if already extracted
        if zip_name in ZIP_MAPPINGS:
            subfolder = ZIP_MAPPINGS[zip_name]["subfolder"]
            extract_dir = os.path.join(data_dir, subfolder)
            if os.path.exists(extract_dir) and os.listdir(extract_dir):
                print(f"  Already extracted: {fname}")
                continue

        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        extracted += 1

    print(f"  Extracted {extracted} zip files")
    return extracted


def find_file(directory, prefix):
    """Find a file in directory matching a prefix pattern."""
    import fnmatch
    if not os.path.exists(directory):
        return None
    for f in os.listdir(directory):
        if fnmatch.fnmatch(f, prefix):
            return os.path.join(directory, f)
    return None


def create_symlinks(data_dir):
    """Create standardized symlinks for experiment scripts."""
    linked = 0

    for zip_name, mapping in ZIP_MAPPINGS.items():
        subfolder = mapping["subfolder"]
        ae_name = mapping["ae_name"]
        task = mapping["task"]
        src_dir = os.path.join(data_dir, subfolder)

        if not os.path.exists(src_dir):
            print(f"  Warning: {subfolder}/ not found, skipping {zip_name}")
            continue

        files_in_dir = os.listdir(src_dir)

        # Find x, x1, y files
        x_src = None
        x1_src = None
        y_src = None

        for f in files_in_dir:
            fpath = os.path.join(src_dir, f)
            if f.startswith('x_') and 'emb' in f and f.endswith('.npy'):
                x_src = fpath
            elif f.startswith('x1_') and f.endswith('.npy'):
                x1_src = fpath
            elif f.startswith('y_') and f.endswith('.npy'):
                y_src = fpath
            elif f.startswith('ds_') and f.endswith('.npy'):
                pass  # ds file, not needed

        # BJ uses same data as SJ (SJ embeddings work for BJ too)
        # Create symlinks for both sj and bj tasks
        tasks_to_link = [task]
        if task == "sj_sel":
            tasks_to_link.append("bj_sel")
        elif task == "sj_mbr":
            tasks_to_link.append("bj_mbr")

        for t in tasks_to_link:
            if x_src:
                target = os.path.join(data_dir, f"x_{t}_{ae_name}.npy")
                if not os.path.exists(target):
                    os.symlink(os.path.abspath(x_src), target)
                    print(f"  {os.path.basename(x_src)} -> x_{t}_{ae_name}.npy")
                    linked += 1

            if x1_src:
                target = os.path.join(data_dir, f"x1_{t}_{ae_name}.npy")
                if not os.path.exists(target):
                    os.symlink(os.path.abspath(x1_src), target)
                    print(f"  {os.path.basename(x1_src)} -> x1_{t}_{ae_name}.npy")
                    linked += 1

            if y_src:
                target = os.path.join(data_dir, f"y_{t}_{ae_name}.npy")
                if not os.path.exists(target):
                    os.symlink(os.path.abspath(y_src), target)
                    print(f"  {os.path.basename(y_src)} -> y_{t}_{ae_name}.npy")
                    linked += 1

    # Also create RQ symlinks from the loose .npy files (already in downloaded_data/)
    # These are from the first RQ_AE_s1 data that was linked directly
    rq_loose = {
        "x_63410_rq_2_emb1.npy": "x_rq_AE_S1.npy",
        "x1_63410_rq_2.npy": "x1_rq_AE_S1.npy",
        "y_63410_rq_0.npy": "y_rq_AE_S1.npy",
    }
    for src_name, target_name in rq_loose.items():
        src = os.path.join(data_dir, src_name)
        target = os.path.join(data_dir, target_name)
        if os.path.exists(src) and not os.path.exists(target):
            os.symlink(os.path.abspath(src), target)
            print(f"  {src_name} -> {target_name}")
            linked += 1

    print(f"\n  Created {linked} symlinks")
    return linked


def verify_models(data_dir):
    """Check available pre-trained models."""
    model_dir = os.path.join(data_dir, "model")
    if not os.path.exists(model_dir):
        print("  No model directory found")
        return

    print("\n  Available pre-trained models:")
    for dirname in sorted(os.listdir(model_dir)):
        dirpath = os.path.join(model_dir, dirname)
        if os.path.isdir(dirpath) and os.path.exists(os.path.join(dirpath, "saved_model.pb")):
            ae_name = MODEL_DIR_MAPPING.get(dirname, "?")
            print(f"    {ae_name}: {dirname}")


def print_data_summary(data_dir):
    """Print summary of available data for each experiment table."""
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 60)

    # Table 3/4: Need histograms
    hist_synth = os.path.exists(os.path.join(data_dir, "histograms_synthetic.npy"))
    hist_real = os.path.exists(os.path.join(data_dir, "histograms_real.npy"))
    print(f"\n  Table 3 (AE CNN synthetic):   {'OK' if hist_synth else 'MISSING histograms_synthetic.npy'}")
    print(f"  Table 4 (AE synth+real):      {'OK' if hist_synth and hist_real else 'MISSING histograms'}")

    # Table 5/15: RQ data
    rq_aes = {"AE_S1": False, "AE_C2": False, "AE_S3": False, "AE_S4": False}
    for ae in rq_aes:
        rq_aes[ae] = os.path.exists(os.path.join(data_dir, f"x_rq_{ae}.npy"))
    rq_ok = sum(rq_aes.values())
    print(f"\n  Table 5/15 (RQ selectivity):  {rq_ok}/4 AE datasets")
    for ae, ok in rq_aes.items():
        print(f"    {ae}: {'OK' if ok else 'MISSING'}")

    # Table 6: SJ selectivity
    sj_sel_aes = {"AE_C2": False, "AE_S4": False, "AE_C3": False}
    for ae in sj_sel_aes:
        sj_sel_aes[ae] = os.path.exists(os.path.join(data_dir, f"x_sj_sel_{ae}.npy"))
    sj_ok = sum(sj_sel_aes.values())
    print(f"\n  Table 6 (SJ selectivity):     {sj_ok}/3 AE datasets")
    for ae, ok in sj_sel_aes.items():
        print(f"    {ae}: {'OK' if ok else 'MISSING'}")

    # Table 7: SJ MBR
    sj_mbr_aes = {"AE_C2": False, "AE_S4": False}
    for ae in sj_mbr_aes:
        sj_mbr_aes[ae] = os.path.exists(os.path.join(data_dir, f"x_sj_mbr_{ae}.npy"))
    mbr_ok = sum(sj_mbr_aes.values())
    print(f"\n  Table 7 (SJ MBR tests):       {mbr_ok}/2 AE datasets")
    for ae, ok in sj_mbr_aes.items():
        print(f"    {ae}: {'OK' if ok else 'MISSING'}")

    # Table 8/18: BJ selectivity
    bj_sel_aes = {"AE_C2": False, "AE_S4": False, "AE_C3": False}
    for ae in bj_sel_aes:
        bj_sel_aes[ae] = os.path.exists(os.path.join(data_dir, f"x_bj_sel_{ae}.npy"))
    bj_ok = sum(bj_sel_aes.values())
    print(f"\n  Table 8/18 (BJ selectivity):  {bj_ok}/3 AE datasets")
    for ae, ok in bj_sel_aes.items():
        print(f"    {ae}: {'OK' if ok else 'MISSING'}")

    # Table 9: BJ MBR
    bj_mbr_aes = {"AE_C2": False, "AE_S4": False}
    for ae in bj_mbr_aes:
        bj_mbr_aes[ae] = os.path.exists(os.path.join(data_dir, f"x_bj_mbr_{ae}.npy"))
    bj_mbr_ok = sum(bj_mbr_aes.values())
    print(f"\n  Table 9 (BJ MBR tests):       {bj_mbr_ok}/2 AE datasets")
    for ae, ok in bj_mbr_aes.items():
        print(f"    {ae}: {'OK' if ok else 'MISSING'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for experiments")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Data directory")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")

    print("Step 1: Extracting zip archives...")
    extract_zips(data_dir)

    print("\nStep 2: Creating standardized symlinks...")
    create_symlinks(data_dir)

    print("\nStep 3: Checking pre-trained models...")
    verify_models(data_dir)

    print_data_summary(data_dir)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  Run experiments:")
    print("    python3 run_all.py --tables 3          # Train AEs (needs histograms)")
    print("    python3 run_all.py --tables 5 15       # RQ selectivity")
    print("    python3 run_all.py --tables 6 7        # SJ experiments")
    print("    python3 run_all.py --tables 8 9 18     # BJ experiments")
    print("    python3 run_all.py --tables all        # Run everything")


if __name__ == "__main__":
    main()
