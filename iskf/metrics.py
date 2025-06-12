"""Metrics for evaluating the performance of a filter."""

import numpy as np


def rmse(x_pred, x_true, axis=0):
    """Root mean squared error. Input arrays have shape (n, T).

    If x_true contains NaN values in any column, that column is ignored.
    """
    # Create a mask for columns that don't have NaN values in x_true
    valid_mask = ~np.isnan(x_true).any(axis=0)

    # Apply mask to both arrays
    x_true_valid = x_true[:, valid_mask]
    x_pred_valid = x_pred[:, valid_mask]

    # Check if there's any valid data left
    if x_true_valid.size == 0:
        return np.nan

    return np.sqrt(np.mean(np.linalg.norm(x_pred_valid - x_true_valid, axis=axis) ** 2))


def rmedse(x_pred, x_true, axis=0):
    """Median squared norm error. Input arrays have shape (n, T).

    If x_true contains NaN values in any column, that column is ignored.
    """
    # Create a mask for columns that don't have NaN values in x_true
    valid_mask = ~np.isnan(x_true).any(axis=0)

    # Apply mask to both arrays
    x_true_valid = x_true[:, valid_mask]
    x_pred_valid = x_pred[:, valid_mask]

    # Check if there's any valid data left
    if x_true_valid.size == 0:
        return np.nan

    return np.sqrt(
        np.median(np.linalg.norm(x_pred_valid - x_true_valid, axis=axis) ** 2)
    )


def mne(x_pred, x_true, axis=0):
    """Mean norm error. Input arrays have shape (n, T).

    If x_true contains NaN values in any column, that column is ignored.
    """
    # Create a mask for columns that don't have NaN values in x_true
    valid_mask = ~np.isnan(x_true).any(axis=0)

    # Apply mask to both arrays
    x_true_valid = x_true[:, valid_mask]
    x_pred_valid = x_pred[:, valid_mask]

    # Check if there's any valid data left
    if x_true_valid.size == 0:
        return np.nan

    return np.mean(np.linalg.norm(x_pred_valid - x_true_valid, axis=axis))


def median_ne(x_pred, x_true, axis=0):
    """Median norm error. Input arrays have shape (n, T).

    If x_true contains NaN values in any column, that column is ignored.
    """
    # Create a mask for columns that don't have NaN values in x_true
    valid_mask = ~np.isnan(x_true).any(axis=0)

    # Apply mask to both arrays
    x_true_valid = x_true[:, valid_mask]
    x_pred_valid = x_pred[:, valid_mask]

    # Check if there's any valid data left
    if x_true_valid.size == 0:
        return np.nan

    return np.median(np.linalg.norm(x_pred_valid - x_true_valid, axis=axis))


def max_ne(x_pred, x_true, axis=0):
    """Maximum norm error. Input arrays have shape (n, T).

    If x_true contains NaN values in any column, that column is ignored.
    """
    # Create a mask for columns that don't have NaN values in x_true
    valid_mask = ~np.isnan(x_true).any(axis=0)

    # Apply mask to both arrays
    x_true_valid = x_true[:, valid_mask]
    x_pred_valid = x_pred[:, valid_mask]

    # Check if there's any valid data left
    if x_true_valid.size == 0:
        return np.nan

    return np.max(np.linalg.norm(x_pred_valid - x_true_valid, axis=axis))


METRIC_REGISTRY = {
    "rmse": rmse,
    "rmedse": rmedse,
    "mne": mne,
    "median_ne": median_ne,
    "max_ne": max_ne,
}
