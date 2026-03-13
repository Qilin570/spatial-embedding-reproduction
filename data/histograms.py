"""Histogram generation and loading utilities.

Functions extracted from the authors' code:
  spatial-embedding/autoEncoders/gen_py/generate_histogram.py
"""
import math
import csv
import os
import numpy as np
from os import listdir
from os.path import isfile, join


# Extracted from the authors' code: generate_histogram.py - get_files_path
def get_files_path(path: str):
    """Recursively find all files in a directory."""
    list_files_paths = []
    for structure in listdir(path):
        sub_path = join(path, structure)
        if isfile(sub_path):
            list_files_paths.append(sub_path)
        else:
            list_files_paths = list_files_paths + get_files_path(sub_path)
    return list_files_paths


# Extracted from the authors' code: generate_histogram.py - gen_hist_from_file
def gen_hist_from_file(dimx, dimy, dimz, file):
    """Generate a local histogram (dimx x dimy x dimz) from a CSV summary file."""
    h0 = np.zeros((dimx, dimy, dimz))
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            x = int(row["i0"])
            if x < 0 or x >= dimx:
                continue
            y = int(row["i1"])
            if y < 0 or y >= dimy:
                continue
            if dimz >= 1:
                h0[x, y, 0] = int(row["num_features"])
            if dimz >= 2:
                h0[x, y, 1] = int(row["size"])
            if dimz >= 3:
                h0[x, y, 2] = int(row["num_points"])
            if dimz >= 4:
                avg_area = float(row["avg_area"])
                if avg_area > 1:
                    avg_area = 1.0
                h0[x, y, 3] = avg_area
            if dimz >= 5:
                avg_side_length_0 = float(row["avg_side_length_0"])
                if avg_side_length_0 > 1:
                    avg_side_length_0 = 1.0
                h0[x, y, 4] = avg_side_length_0
            if dimz >= 6:
                avg_side_length_1 = float(row["avg_side_length_1"])
                if avg_side_length_1 > 1:
                    avg_side_length_1 = 1
                h0[x, y, 5] = avg_side_length_1
    return h0


# Extracted from the authors' code: generate_histogram.py - area_intersection
def area_intersection(l1, r1, l2, r2):
    """Calculate the intersection area of two rectangles.

    Args:
        l1, r1: bottom-left and top-right of rectangle 1
        l2, r2: bottom-left and top-right of rectangle 2
    """
    x, y = 0, 1
    x_dist = min(r1[x], r2[x]) - max(l1[x], l2[x])
    y_dist = min(r1[y], r2[y]) - max(l1[y], l2[y])
    if x_dist > 0.0 and y_dist > 0.0:
        return x_dist * y_dist
    return 0.0


# Extracted from the authors' code: generate_histogram.py - gen_global_hist
def gen_global_hist(h0, dimx, dimy, mbr,
                    global_x_min=0, global_x_max=10,
                    global_y_min=0, global_y_max=10):
    """Compute a global histogram from a local histogram using area intersection.

    Args:
        h0: local histogram (dimx, dimy, dimz)
        dimx, dimy: grid dimensions
        mbr: dict with keys minx, miny, maxx, maxy
        global_x_min/max, global_y_min/max: reference space bounds
    """
    xsize = (mbr['maxx'] - mbr['minx']) / dimx
    ysize = (mbr['maxy'] - mbr['miny']) / dimy
    cellArea = xsize * ysize

    xsizeG = (global_x_max - global_x_min) / dimx
    ysizeG = (global_y_max - global_y_min) / dimy

    hg = np.zeros((dimx, dimy))

    for i in range(dimx):
        for j in range(dimy):
            cell = h0[i, j]
            if cell[0] == 0:
                continue
            xC = mbr['minx'] + xsize * j
            yC = mbr['miny'] + ysize * i

            firstCellGcol = math.floor(xC / xsizeG)
            if firstCellGcol == -1:
                firstCellGcol = 0
            if firstCellGcol < 0 or firstCellGcol >= dimx:
                continue
            firstCellGrow = math.floor(yC / ysizeG)
            if firstCellGrow == -1:
                firstCellGrow = 0
            if firstCellGrow < 0 or firstCellGrow >= dimy:
                continue

            hg[firstCellGrow, firstCellGcol] += (
                cell[0] * area_intersection(
                    (xC, yC), (xC + xsize, yC + ysize),
                    (firstCellGcol * xsizeG, firstCellGrow * ysizeG),
                    (firstCellGcol * xsizeG + xsizeG, firstCellGrow * ysizeG + ysizeG)
                ) / cellArea
            )

            secondCellGcol = math.floor((xC + xsize) / xsizeG)
            if secondCellGcol >= dimx:
                secondCellGcol = dimx - 1
            if secondCellGcol > firstCellGcol:
                hg[firstCellGrow, secondCellGcol] += (
                    cell[0] * area_intersection(
                        (xC, yC), (xC + xsize, yC + ysize),
                        (secondCellGcol * xsizeG, firstCellGrow * ysizeG),
                        (secondCellGcol * xsizeG + xsizeG, firstCellGrow * ysizeG + ysizeG)
                    ) / cellArea
                )

            secondCellGrow = math.floor((yC + ysize) / ysizeG)
            if secondCellGrow >= dimy:
                secondCellGrow = dimy - 1
            if secondCellGrow > firstCellGrow:
                hg[secondCellGrow, firstCellGcol] += (
                    cell[0] * area_intersection(
                        (xC, yC), (xC + xsize, yC + ysize),
                        (firstCellGcol * xsizeG, secondCellGrow * ysizeG),
                        (firstCellGcol * xsizeG + xsizeG, secondCellGrow * ysizeG + ysizeG)
                    ) / cellArea
                )

            if secondCellGrow > firstCellGrow and secondCellGcol > firstCellGcol:
                hg[secondCellGrow, secondCellGcol] += (
                    cell[0] * area_intersection(
                        (xC, yC), (xC + xsize, yC + ysize),
                        (secondCellGcol * xsizeG, secondCellGrow * ysizeG),
                        (secondCellGcol * xsizeG + xsizeG, secondCellGrow * ysizeG + ysizeG)
                    ) / cellArea
                )
    return hg


# Adapted from the authors' code: generate_histogram.py - gen_input_from_file
def gen_input_from_file(dimx, dimy, dimz, path, mbrFile, fieldName, suffix):
    """Load local histograms from files and generate corresponding global histograms.

    Args:
        dimx, dimy, dimz: histogram dimensions
        path: directory containing histogram CSV files
        mbrFile: CSV file with dataset MBRs
        fieldName: 0 for Alberto's format, 1 for Ahmed's format
        suffix: suffix to append to derive dataset name
    Returns:
        hh: array of local histograms
        hg: array of global histograms
    """
    mbr = {}
    with open(mbrFile, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            if fieldName == 0:
                name = row["datasetName"]
                mbr[name] = dict(minx=float(row["minX"]), miny=float(row["minY"]),
                                 maxx=float(row["maxX"]), maxy=float(row["maxY"]))
            elif fieldName == 1:
                name = row["dataset"]
                mbr[name] = dict(minx=float(row["x1"]), miny=float(row["y1"]),
                                 maxx=float(row["x2"]), maxy=float(row["y2"]))

    files = get_files_path(path)
    print(f'Found {len(files)} files')
    hh = np.zeros((len(files), dimx, dimy, dimz))
    hg = np.zeros((len(files), dimx, dimy))
    count = 0
    for ff in files:
        h0 = gen_hist_from_file(dimx, dimy, dimz, ff)
        hh[count] = h0
        name = (ff.rpartition("/")[2]).rpartition("_summary")[0] + suffix
        if name in mbr.keys():
            mbr0 = mbr[name]
        else:
            mbr0 = dict(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0)
        hg[count] = gen_global_hist(h0, dimx, dimy, mbr0)
        count += 1
    return hh, hg


def load_all_histograms(data_dir):
    """Load all histogram .npy files from directory.

    Returns dict mapping category names to histogram arrays.
    """
    histograms = {}
    if not os.path.exists(data_dir):
        print(f"Warning: histogram directory {data_dir} does not exist")
        return histograms
    for f in os.listdir(data_dir):
        if f.endswith('.npy') or f.endswith('.npy.gz'):
            key = f.replace('.npy.gz', '').replace('.npy', '')
            histograms[key] = np.load(os.path.join(data_dir, f))
    return histograms
