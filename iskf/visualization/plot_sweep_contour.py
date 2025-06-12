"""
Plotting utility for visualizing HuberizedSSKalmanFilter grid search results.

This script provides a function to generate a contour plot of a performance
metric (e.g., RMSE) over a grid of `coef_s` and `coef_o` hyperparameters.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from frozendict import frozendict  # For type hinting the input dict keys


def setup_matplotlib_style():
    """Sets up a consistent matplotlib style for plots."""
    font = {"family": "normal", "weight": "bold", "size": 16}
    matplotlib.rc("font", **font)
    plt.rcParams["text.usetex"] = True
    sns.set_style("whitegrid")


setup_matplotlib_style()  # Apply style when module is imported


def plot_contour_results(
    sweep_results: Dict[frozendict, float],
    x_param_name: str,  # Key for the x-axis data in sweep_results
    y_param_name: str,  # Key for the y-axis data in sweep_results
    x_display_name: str,  # Display name for the x-axis
    y_display_name: str,  # Display name for the y-axis
    metric_name: str = "RMSE",  # Name of the metric for labeling
    save_path: Optional[str] = None,
    levels: Optional[
        Union[int, List[float], np.ndarray]
    ] = 20,  # Number of contour levels or array of levels
    title: Optional[str] = None,
    use_log_scale: bool = True,  # Whether to use log scale for axes
):
    """
    Plots a contour map of the metric values from hyperparameter sweep results.

    Args:
        sweep_results: Dictionary mapping hyperparameter sets (frozendict containing
                       x_param_name and y_param_name) to metric scores.
        x_param_name: The dictionary key for the x-axis hyperparameter (e.g., "coef_s").
        y_param_name: The dictionary key for the y-axis hyperparameter (e.g., "coef_o").
        x_display_name: The label for the x-axis (e.g., r"$\\delta_s$").
        y_display_name: The label for the y-axis (e.g., r"$\\delta_o$").
        metric_name: Name of the metric for the colorbar label (e.g., "RMSE").
        save_path (optional): Path to save the plot image. If None, plot is shown.
        levels (optional): Number of contour levels or a list/array of specific levels.
        title (optional): Title for the plot.
        use_log_scale (optional): Whether to use log scale for x and y axes.
    """
    if not sweep_results:
        print("Sweep results dictionary is empty. Cannot generate plot.")
        return

    # Extract unique sorted x and y parameter values
    try:
        x_values = [dict(key)[x_param_name] for key in sweep_results.keys()]
        y_values = [dict(key)[y_param_name] for key in sweep_results.keys()]

        # Filter out infinity values for determining axis limits
        x_finite = [x for x in x_values if np.isfinite(x)]
        y_finite = [y for y in y_values if np.isfinite(y)]

        if not x_finite or not y_finite:
            print(
                "Warning: No finite parameter values found. Check your parameter grid."
            )
            return

        x_unique_values = sorted(list(set(x_finite)))
        y_unique_values = sorted(list(set(y_finite)))
    except KeyError as e:
        print(
            f"Error: Parameter name {e} not found in sweep_results keys. Cannot generate plot."
        )
        return

    if not x_unique_values or not y_unique_values:
        print(
            f"Could not extract {x_param_name} or {y_param_name} values. Ensure they exist in all sweep_results keys. Cannot generate plot."
        )
        return

    # Create a 2D grid for metric values, initialized with NaN
    metric_grid = np.full((len(y_unique_values), len(x_unique_values)), np.nan)

    # Create mappings from parameter values to indices
    x_value_to_idx = {val: idx for idx, val in enumerate(x_unique_values)}
    y_value_to_idx = {val: idx for idx, val in enumerate(y_unique_values)}

    # Populate the metric_grid
    for param_set_frozen, metric_value in sweep_results.items():
        param_set = dict(param_set_frozen)
        try:
            x_val = param_set[x_param_name]
            y_val = param_set[y_param_name]

            # Skip infinity values
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue

            # Find indices in the sorted lists
            idx_x = x_value_to_idx.get(x_val)
            idx_y = y_value_to_idx.get(y_val)

            if idx_x is not None and idx_y is not None:
                metric_grid[idx_y, idx_x] = metric_value
        except (KeyError, ValueError) as e:
            print(
                f"Skipping param_set {param_set} due to missing key ({x_param_name} or {y_param_name}) or value not in sweep axes: {e}"
            )
            continue

    if np.all(np.isnan(metric_grid)):
        print(
            f"Metric grid is all NaNs. Check if {x_param_name}/{y_param_name} in sweep_results match generated axes."
        )
        return

    X_GRID, Y_GRID = np.meshgrid(x_unique_values, y_unique_values)

    plt.figure(figsize=(8, 6))
    if title:
        plt.title(title)

    contour_kwargs = {"colors": "black", "linewidths": 0.7}
    contourf_kwargs = {"cmap": "Greys", "alpha": 0.85}

    if levels is not None:
        contour_kwargs["levels"] = levels
        contourf_kwargs["levels"] = levels

    plt.contour(X_GRID, Y_GRID, metric_grid, **contour_kwargs)
    contour_fill = plt.contourf(X_GRID, Y_GRID, metric_grid, **contourf_kwargs)

    cbar = plt.colorbar(contour_fill)
    cbar.set_label(metric_name)
    plt.xlabel(x_display_name)
    plt.ylabel(y_display_name)

    # Use logarithmic scale if requested and appropriate
    if use_log_scale:
        # Check if all values are positive
        if all(x > 0 for x in x_unique_values) and all(y > 0 for y in y_unique_values):
            plt.xscale("log")
            plt.yscale("log")
            print("Using logarithmic scale for both axes")
        else:
            print("Cannot use logarithmic scale: some values are non-positive")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Contour plot saved to {save_path}")
    else:
        plt.show()
