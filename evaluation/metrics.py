"""Evaluation metrics: WMAPE, MAPE, RMA, MAE for M1 and M2.

Metric functions adapted from the authors' code:
  spatial-embedding/autoEncoders/code_py/run_autoenc.py (wmape)
  spatial-embedding/modelsRQ/code_py/run_model_all.py (mape_error_zero)
"""
import numpy as np


# Adapted from the authors' code: run_autoenc.py - wmape
# (modified: vectorized with numpy instead of nested loops)
def wmape_autoencoder(orig, decoded):
    """Compute WMAPE for autoencoder reconstruction.

    Uses per-feature equal-weight averaging as in the original paper:
    wmape_f[i] = sum|orig_f - dec_f| / sum(orig_f) for each feature
    wmape = mean(wmape_f) across all features

    Args:
        orig: original data (N, dimx, dimy, dimz)
        decoded: reconstructed data
    Returns:
        overall_wmape: arithmetic mean of per-feature WMAPEs
        per_feature_wmape: list of per-feature WMAPE
    """
    n_features = orig.shape[-1] if orig.ndim > 3 else 1
    per_feature = []

    for f in range(n_features):
        if orig.ndim > 3:
            actual = orig[..., f].flatten()
            pred = decoded[..., f].flatten()
        else:
            actual = orig.flatten()
            pred = decoded.flatten()
        abs_err = np.sum(np.abs(actual - pred))
        act_sum = np.sum(np.abs(actual))
        wmape_f = abs_err / act_sum if act_sum > 0 else 0.0
        per_feature.append(wmape_f)

    overall = np.mean(per_feature)
    return overall, per_feature


# Extracted from the authors' code: run_model_all.py - mape_error_zero
def mape_error_zero(y, predict):
    """Compute comprehensive error metrics including handling of zero values.

    This is a direct port of the original mape_error_zero function.

    Args:
        y: actual values (1D array or column vector)
        predict: predicted values (1D array or column vector)
    Returns:
        dict with keys: rma, mape, wmape, freq, non_zero, mae_zero,
                        freq_zero, zero, wmape_tot, outliers, outliers_zero
    """
    y = np.asarray(y).flatten()
    predict = np.asarray(predict).flatten()

    delta_zero = 0.0
    zero = 0
    non_zero = 0
    outliers = 0
    outliers_zero = 0
    delta = 0.0
    delta_w = 0.0
    den_w = 0.0
    rma = 0.0
    freq_zero = np.zeros(7, dtype=int)
    freq = np.zeros(7, dtype=int)

    for i in range(len(y)):
        val = predict[i]
        if val < 0.0:
            val = 0.0
        if y[i] == 0.0:
            zero += 1
            delta_zero += val
            if val == 0.0:
                freq_zero[6] += 1
            elif val < 0.000001:
                freq_zero[5] += 1
            elif val < 0.00001:
                freq_zero[4] += 1
            elif val < 0.0001:
                freq_zero[3] += 1
            elif val < 0.001:
                freq_zero[2] += 1
            elif val < 0.01:
                freq_zero[1] += 1
            elif val < 0.1:
                freq_zero[0] += 1
            else:
                outliers_zero += 1
        else:
            non_zero += 1
            delta += abs(y[i] - val) / y[i]
            delta_w += abs(y[i] - val)
            a = abs(predict[i] / y[i])
            if a < 1.0:
                a = 1 / a
            rma += a
            den_w += y[i]

            rel_err = abs(y[i] - predict[i]) / y[i]
            if rel_err < 0.00001:
                freq[6] += 1
            elif rel_err < 0.0001:
                freq[5] += 1
            elif rel_err < 0.001:
                freq[4] += 1
            elif rel_err < 0.01:
                freq[3] += 1
            elif rel_err < 0.1:
                freq[2] += 1
            elif rel_err < 1:
                freq[1] += 1
            elif rel_err < 10:
                freq[0] += 1
            else:
                outliers += 1

    if zero == 0:
        zero_denom = 1  # avoid division by zero
    else:
        zero_denom = zero

    if non_zero == 0:
        non_zero_denom = 1
    else:
        non_zero_denom = non_zero

    if den_w == 0:
        den_w = 1

    return {
        'rma': rma / non_zero_denom,
        'mape': delta / non_zero_denom,
        'wmape': delta_w / den_w,
        'freq': freq.tolist(),
        'non_zero': non_zero,
        'mae_zero': delta_zero / zero_denom,
        'freq_zero': freq_zero.tolist(),
        'zero': zero,
        'wmape_tot': (delta_w + delta_zero) / den_w,
        'outliers': outliers,
        'outliers_zero': outliers_zero,
    }


def compute_baseline_rq(y):
    """Compute baseline WMAPE using mean prediction for RQ.

    Args:
        y: actual selectivity values
    Returns:
        baseline WMAPE
    """
    y = np.asarray(y).flatten()
    mean_pred = np.mean(y)
    abs_error = np.sum(np.abs(y - mean_pred))
    actual_sum = np.sum(np.abs(y))
    return abs_error / actual_sum if actual_sum > 0 else 0.0


def compute_baseline_jn(y):
    """Compute baseline WMAPE using mean prediction for JN.

    Args:
        y: actual selectivity values
    Returns:
        baseline WMAPE
    """
    return compute_baseline_rq(y)
