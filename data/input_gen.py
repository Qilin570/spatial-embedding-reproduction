"""Input generation for M2 models: RQ, SJ, BJ embeddings.

Functions adapted from the authors' code:
  spatial-embedding/modelsRQ/gen_py/generate_input_RQ.py
  spatial-embedding/modelsSJ/gen_py/generate_input_JN.py
"""
import math
import csv
import numpy as np
from tensorflow import keras

from data.histograms import gen_hist_from_file, gen_global_hist, area_intersection
from data.normalization import nor_g_ab
import configs as cfg


# Extracted from the authors' code: generate_input_RQ.py - get_embedding
def get_embedding_rq(local_enc, global_enc, rq_hist, dataset_file, mbr, norm_max):
    """Generate local, global, and RQ embeddings for a range query.

    Args:
        local_enc: local autoencoder model
        global_enc: global autoencoder model
        rq_hist: range query histogram (128x128x1)
        dataset_file: path to histogram CSV file
        mbr: dict with minx, miny, maxx, maxy
        norm_max: normalization max values
    Returns:
        emb_local, emb_global, emb_rq
    """
    hist_local = gen_hist_from_file(cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z, dataset_file)
    hist_local_norm, _, _ = nor_g_ab(
        hist_local.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z)),
        1, cfg.NORM_MIN, norm_max
    )
    emb_local = local_enc.encoder(hist_local_norm.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z)))

    hist_glob = gen_global_hist(hist_local, cfg.DIM_H_X, cfg.DIM_H_Y, mbr)
    hist_glob_norm, _, _ = nor_g_ab(
        hist_glob.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y)),
        1, cfg.NORM_MIN_G, cfg.NORM_MAX_G
    )
    emb_global = global_enc.encoder(hist_glob_norm.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_HG_Z)))

    emb_rq = global_enc.encoder(rq_hist.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_HG_Z)))

    return emb_local, emb_global, emb_rq


# Extracted from the authors' code: generate_input_JN.py - get_embedding
def get_embedding_jn(local_enc, global_enc, dataset_file, mbr, norm_max):
    """Generate local and global embeddings for a join dataset.

    Args:
        local_enc: local autoencoder model
        global_enc: global autoencoder model
        dataset_file: path to histogram CSV file
        mbr: dict with minx, miny, maxx, maxy
        norm_max: normalization max values
    Returns:
        emb_local, emb_global
    """
    hist_local = gen_hist_from_file(cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z, dataset_file)
    hist_local_norm, _, _ = nor_g_ab(
        hist_local.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z)),
        1, cfg.NORM_MIN, norm_max
    )
    emb_local = local_enc.encoder(hist_local_norm.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z)))

    hist_glob = gen_global_hist(hist_local, cfg.DIM_H_X, cfg.DIM_H_Y, mbr)
    hist_glob_norm, _, _ = nor_g_ab(
        hist_glob.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y)),
        1, cfg.NORM_MIN_G, cfg.NORM_MAX_G
    )
    emb_global = global_enc.encoder(hist_glob_norm.reshape((1, cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_HG_Z)))

    return emb_local, emb_global


# Extracted from the authors' code: generate_input_RQ.py - gen_rq_layer
def gen_rq_layer(rq, dimx, dimy,
                 x_min_ref=None, x_max_ref=None,
                 y_min_ref=None, y_max_ref=None):
    """Generate a 128x128x1 histogram representing a range query rectangle.

    Args:
        rq: dict with minx, miny, maxx, maxy
        dimx, dimy: grid dimensions
    """
    if x_min_ref is None:
        x_min_ref = cfg.RQ_X_MIN_REF
    if x_max_ref is None:
        x_max_ref = cfg.RQ_X_MAX_REF
    if y_min_ref is None:
        y_min_ref = cfg.RQ_Y_MIN_REF
    if y_max_ref is None:
        y_max_ref = cfg.RQ_Y_MAX_REF

    rq_layer = np.zeros((dimx, dimy, 1))
    xsizeG = (x_max_ref - x_min_ref) / dimx
    ysizeG = (y_max_ref - y_min_ref) / dimy
    cell_area = xsizeG * ysizeG

    start_cell_row = max(0, math.floor((rq["miny"] - y_min_ref) / ysizeG))
    if start_cell_row > dimy - 1:
        start_cell_row = dimy
    start_cell_col = max(0, math.floor((rq["minx"] - x_min_ref) / xsizeG))
    if start_cell_col > dimx - 1:
        start_cell_col = dimx
    end_cell_row = math.floor((rq["maxy"] - y_min_ref) / ysizeG)
    if end_cell_row < 0:
        end_cell_row = -1
    if end_cell_row > dimy - 1:
        end_cell_row = dimy - 1
    end_cell_col = math.floor((rq["maxx"] - x_min_ref) / xsizeG)
    if end_cell_col < 0:
        end_cell_col = -1
    if end_cell_col > dimx - 1:
        end_cell_col = dimx - 1

    for i in range(start_cell_row, end_cell_row + 1):
        for j in range(start_cell_col, end_cell_col + 1):
            cell_x_min = x_min_ref + j * xsizeG
            cell_x_max = cell_x_min + xsizeG
            cell_y_min = y_min_ref + i * ysizeG
            cell_y_max = cell_y_min + ysizeG
            rq_layer[i, j] = area_intersection(
                (rq['minx'], rq['miny']), (rq['maxx'], rq['maxy']),
                (cell_x_min, cell_y_min), (cell_x_max, cell_y_max)
            ) / cell_area
    return rq_layer


# Adapted from the authors' code: generate_input_RQ.py
# (modified: refactored for config-driven AE selection)
def generate_rq_inputs(ae_config, local_enc, global_enc,
                       result_file, hist_dir, delim=',',
                       flag_sel_card=0, from_x=0, to_x=None, perc=1.0):
    """Generate M2 input data for Range Query experiments.

    Reads a result CSV, computes embeddings, and returns arrays.

    Args:
        ae_config: AutoencoderConfig
        local_enc: local encoder model
        global_enc: global encoder model
        result_file: CSV with RQ results
        hist_dir: directory containing histogram CSV files
        flag_sel_card: 0=selectivity, 1=cardinality, 2=mbrTests, 3=mbrTests_sel
    Returns:
        out_x, out_x1, out_y, out_ds
    """
    norm_max = cfg.get_norm_max(ae_config.trained_on)
    emb_shape = ae_config.emb_shape
    dim_e_x, dim_e_y, dim_e_z = emb_shape

    rows = []
    with open(result_file, mode='r') as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            rows.append(row)

    if to_x is None:
        to_x = len(rows)
    selected = rows[from_x:to_x]
    total = len(selected)

    # mode 0: local emb + global emb + rq emb concatenated along z
    out_x = np.zeros((total, dim_e_x, dim_e_y, dim_e_z + 2 * 2))
    out_x1 = np.zeros((1,))
    out_y = np.zeros((total,))
    out_ds = np.empty((total, 2), dtype='<U40')
    keep = np.ones((total,), dtype=np.int8)

    for idx, row in enumerate(selected):
        if idx % 100 == 0:
            print(f"RQ input gen: {idx}/{total}")

        file_hist = hist_dir + row["dataset"] + "_summary.csv"
        rq0 = dict(minx=float(row["rq_minx"]), miny=float(row["rq_miny"]),
                    maxx=float(row["rq_maxx"]), maxy=float(row["rq_maxy"]))
        mbr0 = dict(minx=float(row["minx"]), miny=float(row["miny"]),
                    maxx=float(row["maxx"]), maxy=float(row["maxy"]))

        if float(row.get("card", 1)) <= 0:
            keep[idx] = 0
            continue

        out_ds[idx] = [row["dataset"], row.get("distr", "")]

        hist_RQ = gen_rq_layer(rq0, cfg.DIM_H_X, cfg.DIM_H_Y)
        embL, embG, embRQ = get_embedding_rq(local_enc, global_enc, hist_RQ, file_hist, mbr0, norm_max)
        embL = embL.numpy().reshape((dim_e_x, dim_e_y, dim_e_z))
        embG = embG.numpy().reshape((32, 32, 2))
        embRQ = embRQ.numpy().reshape((32, 32, 2))
        x = np.concatenate([embL, embG, embRQ], axis=2)
        out_x[idx] = x

        if flag_sel_card == 0:
            out_y[idx] = float(row["rq_sel_real"])
        elif flag_sel_card == 1:
            out_y[idx] = float(row["rq_card_real"])
        elif flag_sel_card == 2:
            out_y[idx] = float(row["mbrTests"])
        else:
            out_y[idx] = float(row["mbrTests"]) / float(row["card"])

    out_x = out_x[keep == 1]
    out_y = out_y[keep == 1]
    out_ds = out_ds[keep == 1]
    return out_x, out_x1, out_y, out_ds


# Adapted from the authors' code: generate_input_JN.py
# (modified: refactored for config-driven AE selection, distribution mapping)
def generate_jn_inputs(ae_config, local_enc, global_enc,
                       result_file, summary_file, hist_dir,
                       delim=',', flag_sel_card=0, max_y=1.0,
                       from_x=0, to_x=None, data_type='synt'):
    """Generate M2 input data for Self-Join or Binary-Join experiments.

    Args:
        ae_config: AutoencoderConfig
        local_enc: local encoder model
        global_enc: global encoder model
        result_file: CSV with join results
        summary_file: CSV with dataset summaries
        hist_dir: directory containing histogram CSV files
        flag_sel_card: 0=selectivity, 1=cardinality, 2=mbrTests, 3=mbrTests_sel
        max_y: maximum acceptable y value
        data_type: 'synt', 'real', or 'real_er'
    Returns:
        out_x, out_x1, out_y, out_ds
    """
    norm_max = cfg.get_norm_max(ae_config.trained_on)
    emb_shape = ae_config.emb_shape
    dim_e_x, dim_e_y, dim_e_z = emb_shape

    # Read summary file
    mbr = {}
    features = {}
    distr = {}
    with open(summary_file, mode='r') as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            name = row["datasetName"]
            if data_type in ("real", "real_er"):
                mbr[name] = dict(minx=float(row["minX"]), miny=float(row["minY"]),
                                 maxx=float(row["maxX"]), maxy=float(row["maxY"]))
            else:
                mbr[name] = dict(minx=float(row["x1"]), miny=float(row["y1"]),
                                 maxx=float(row["x2"]), maxy=float(row["y2"]))
            distr[name] = row.get("distribution", "")
            features[name] = dict(
                card=float(row["num_features"]),
                size=float(row["size"]),
                numPnts=float(row["num_points"]),
                avgArea=float(row["avg_area"]),
                avgLenX=float(row["avg_side_length_0"]),
                avgLenY=float(row["avg_side_length_1"]),
            )

    # Read result file
    result_rows = []
    with open(result_file, mode='r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            result_rows.append(row)

    if to_x is None:
        to_x = len(result_rows)
    selected = result_rows[from_x:to_x]
    total = len(selected)

    out_x = np.zeros((total, dim_e_x, dim_e_y, 2 * dim_e_z))
    out_x1 = np.zeros((total, dim_e_x, dim_e_y, 2 * 2))
    out_y = np.zeros((total,))
    out_ds = np.zeros((total, 2))
    keep = np.ones((total,), dtype=np.int8)

    for idx, row in enumerate(selected):
        if idx % 100 == 0:
            print(f"JN input gen: {idx}/{total}")

        file1 = row["dataset1"]
        file2 = row["dataset2"]

        if data_type == "synt":
            idx_rot = file1.find("_r")
            if idx_rot < 0:
                i_start = file1.find("dataset-")
                i_end = file1.find(".")
                file1 = file1[i_start:i_end]
            fileHist1 = hist_dir + file1 + "_summary.csv"

            idx_rot = file2.find("_r")
            if idx_rot < 0:
                i_start = file2.find("dataset-")
                i_end = file2.find(".")
                file2 = file2[i_start:i_end]
            fileHist2 = hist_dir + file2 + "_summary.csv"
        else:
            fileHist1 = hist_dir + file1 + "_summary.csv"
            if data_type == "real":
                file1 = "lakes_parks/" + file1
            fileHist2 = hist_dir + file2 + "_summary.csv"
            if data_type == "real":
                file2 = "lakes_parks/" + file2

        try:
            embL1, embG1 = get_embedding_jn(local_enc, global_enc, fileHist1, mbr[file1], norm_max)
            embL1 = embL1.numpy().reshape((dim_e_x, dim_e_y, dim_e_z))
            embG1 = embG1.numpy().reshape((32, 32, 2))
            embL2, embG2 = get_embedding_jn(local_enc, global_enc, fileHist2, mbr[file2], norm_max)
            embL2 = embL2.numpy().reshape((dim_e_x, dim_e_y, dim_e_z))
            embG2 = embG2.numpy().reshape((32, 32, 2))
        except (KeyError, FileNotFoundError) as e:
            print(f"Warning: skipping row {idx}: {e}")
            keep[idx] = 0
            continue

        out_x[idx] = np.concatenate([embL1, embL2], axis=2)
        out_x1[idx] = np.concatenate([embG1, embG2], axis=2)

        c1 = features[file1]["card"]
        c2 = features[file2]["card"]

        if flag_sel_card == 0:
            y = float(row["resultSJSize"]) / (c1 * c2)
            if y >= 1.1 * max_y:
                keep[idx] = 0
        elif flag_sel_card == 1:
            y = float(row["resultSJSize"])
        elif flag_sel_card == 2:
            y = float(row["PBSMMBRTests"])
        else:
            y = float(row["PBSMMBRTests"]) / (c1 * c2)
            if y >= 1.1 * max_y:
                keep[idx] = 0

        out_y[idx] = y

        # Distribution encoding
        d_map = {"uniform": 0, "parcel": 1, "gaussian": 2, "bit": 3,
                 "diagonal": 4, "sierpinski": 5}
        out_ds[idx, 0] = d_map.get(distr.get(file1, "").lower(), 6)
        out_ds[idx, 1] = d_map.get(distr.get(file2, "").lower(), 6)

    out_x = out_x[keep == 1]
    out_x1 = out_x1[keep == 1]
    out_y = out_y[keep == 1]
    out_ds = out_ds[keep == 1]
    return out_x, out_x1, out_y, out_ds
