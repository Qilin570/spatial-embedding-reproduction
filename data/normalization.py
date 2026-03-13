"""Normalization and denormalization functions for histograms and target variables.

All functions extracted from the authors' code:
  spatial-embedding/autoEncoders/gen_py/generate_histogram.py
"""
import math
import numpy as np


# Extracted from the authors' code: generate_histogram.py - nor_g_ab
def nor_g_ab(hist, c, min_val, max_val):
    """Normalize histogram with min-max and optional log(1+cx) transform.

    Args:
        hist: input histogram array (3D or 4D)
        c: if > 0, apply log(1+c*x) transform before min-max
        min_val: minimum value(s) for normalization (scalar or array)
        max_val: maximum value(s) for normalization (scalar or array)
    Returns:
        normalized histogram, original min, original max
    """
    minimum_nolog = np.amin(hist, axis=tuple(range(len(hist.shape) - 1))) if hist.ndim > 1 else np.amin(hist)
    maximum_nolog = np.amax(hist, axis=tuple(range(len(hist.shape) - 1))) if hist.ndim > 1 else np.amax(hist)

    hist = hist.copy().astype(np.float64)

    if c:
        hist = np.log(1 + c * hist)
        if isinstance(min_val, (list, np.ndarray)):
            min_val = np.log(1 + c * np.array(min_val))
        elif min_val != -1:
            min_val = np.log(1 + c * min_val)
        if isinstance(max_val, (list, np.ndarray)):
            max_val = np.log(1 + c * np.array(max_val))
        elif max_val != -1:
            max_val = np.log(1 + c * max_val)

    if isinstance(min_val, (int, float)) and min_val == -1:
        minimum = np.amin(hist, axis=tuple(range(len(hist.shape) - 1))) if hist.ndim > 1 else np.amin(hist)
    else:
        minimum = min_val

    if isinstance(max_val, (int, float)) and max_val == -1:
        maximum = np.amax(hist, axis=tuple(range(len(hist.shape) - 1))) if hist.ndim > 1 else np.amax(hist)
    else:
        maximum = max_val

    if len(hist.shape) == 3:
        denom = maximum - minimum
        if isinstance(denom, np.ndarray):
            denom[denom == 0] = 1
        elif denom == 0:
            denom = 1
        return (hist - minimum) / denom, minimum_nolog, maximum_nolog

    for z_dim in range(hist.shape[-1]):
        denom = maximum[z_dim] - minimum[z_dim]
        if denom == 0:
            denom = 1
        hist[..., z_dim] = (hist[..., z_dim] - minimum[z_dim]) / denom

    return hist, minimum_nolog, maximum_nolog


# Extracted from the authors' code: generate_histogram.py - denorm_g_ab
def denorm_g_ab(hist, c, min_val, max_val):
    """Denormalize histogram (inverse of nor_g_ab).

    Args:
        hist: normalized histogram
        c: log transform constant (same as used in normalization)
        min_val: original min value(s)
        max_val: original max value(s)
    Returns:
        denormalized histogram
    """
    hist = hist.copy().astype(np.float64)

    if c:
        min_log = np.log(1 + c * np.array(min_val))
        max_log = np.log(1 + c * np.array(max_val))
    else:
        min_log = np.array(min_val) if isinstance(min_val, (list, np.ndarray)) else min_val
        max_log = np.array(max_val) if isinstance(max_val, (list, np.ndarray)) else max_val

    delta = max_log - min_log

    if hist.ndim == 4:
        norm_h = np.zeros_like(hist)
        for z_dim in range(hist.shape[3]):
            norm_h[:, :, :, z_dim] = np.exp(hist[:, :, :, z_dim] * delta[z_dim] + min_log[z_dim])
            norm_h[:, :, :, z_dim] = (norm_h[:, :, :, z_dim] - 1) / c
        return norm_h
    elif hist.ndim == 3:
        norm_h = np.zeros_like(hist)
        if isinstance(delta, np.ndarray) and len(delta) > 1:
            for z_dim in range(hist.shape[2]):
                norm_h[:, :, z_dim] = np.exp(hist[:, :, z_dim] * delta[z_dim] + min_log[z_dim])
                norm_h[:, :, z_dim] = (norm_h[:, :, z_dim] - 1) / c
        else:
            norm_h = np.exp(hist * delta + min_log)
            norm_h = (norm_h - 1) / c
        return norm_h
    else:
        norm_h = np.exp(hist * delta + min_log)
        norm_h = (norm_h - 1) / c
        return norm_h


# Extracted from the authors' code: generate_histogram.py - nor_y_ab
def nor_y_ab(y, c, min_val, max_val):
    """Normalize target variable with optional log transform and min-max scaling.

    Args:
        y: target values
        c: if > 0, apply log(1+c*y) transform
        min_val: minimum (-1 to compute from data)
        max_val: maximum (-1 to compute from data)
    Returns:
        normalized y
    """
    y = np.double(y)
    if c > 0:
        y_norm = np.log(1 + c * y)
    else:
        y_norm = y.copy()

    if min_val == -1:
        minimum = np.amin(y_norm, axis=0)
    else:
        if c > 0:
            minimum = math.log(1 + c * min_val)
        else:
            minimum = min_val

    if max_val == -1:
        maximum = np.amax(y_norm, axis=0)
    else:
        if c > 0:
            maximum = math.log(1 + c * max_val)
        else:
            maximum = max_val

    denom = maximum - minimum
    if denom == 0:
        denom = 1
    y_norm = (y_norm - minimum) / denom
    return y_norm


# Extracted from the authors' code: generate_histogram.py - denorm_y_ab
def denorm_y_ab(y_nor, c, min_val, max_val):
    """Denormalize target variable (inverse of nor_y_ab).

    Args:
        y_nor: normalized values
        c: log transform constant
        min_val: original minimum
        max_val: original maximum
    Returns:
        denormalized y
    """
    y_nor = np.double(y_nor)
    min_val = np.double(min_val)
    max_val = np.double(max_val)
    if c > 0:
        min_log = math.log(1 + c * min_val)
        max_log = math.log(1 + c * max_val)
        delta = max_log - min_log
        y = np.exp(y_nor * delta + min_log)
        y = (y - 1) / c
    else:
        delta = max_val - min_val
        y = y_nor * delta + min_val
    return y


# Extracted from the authors' code: generate_histogram.py - nor_a_ab
def nor_a_ab(a, c, min_val, max_val):
    """Normalize local histogram array with optional log transform.

    Args:
        a: histogram array (N, dimx, dimy, dimz)
        c: log constant
        min_val: per-feature min values
        max_val: per-feature max values
    Returns:
        normalized array
    """
    min_arr = np.array(min_val)
    max_arr = np.array(max_val)
    if c > 0:
        a_norm = np.log(1 + c * a)
        minimum = np.log(1 + c * min_arr)
        maximum = np.log(1 + c * max_arr)
    else:
        a_norm = a.copy()
        minimum = min_arr
        maximum = max_arr
    for z_dim in range(a_norm.shape[3]):
        denom = maximum[z_dim] - minimum[z_dim]
        if denom == 0:
            denom = 1
        a_norm[:, :, :, z_dim] = (a_norm[:, :, :, z_dim] - minimum[z_dim]) / denom
    return a_norm
