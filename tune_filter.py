"""
Unified script to tune hyperparameters for various Kalman filter implementations.

This script performs the following steps based on command-line arguments:
1. Loads pre-simulated data from a pickle file.
2. Reconstructs the system model from model_type and model_params.
3. Based on the specified filter_type:
    a. Defines filter-specific hyperparameter grids.
    b. Instantiates the chosen filter.
    c. Runs grid search using the loaded simulation data.
    d. Prints the best hyperparameters and performance metric.
    e. Saves detailed results to a .pkl file.
    f. Generates and saves an appropriate plot (contour or line).
"""

import os
import pickle
import argparse
import pathlib
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project-specific imports
from iskf.models.vehicle import vehicle_ss
from iskf.models.cstr import cascaded_cstr_ss
from iskf.grid_search import grid_search_filter_hyperparams
from iskf.filters.util import chi_squared_quantile
from iskf.metrics import METRIC_REGISTRY

# Filter classes
from iskf.filters.huber_kalman_filter import HuberKalmanFilter
from iskf.filters.circular_huber_kalman_filter import (
    CircularHuberKalmanFilter,
)
from iskf.filters.steady_huber_kalman_filter import SteadyHuberKalmanFilter
from iskf.filters.steady_circular_huber_kalman_filter import (
    SteadyCircularHuberKalmanFilter,
)
from iskf.filters.steady_regularized_kalman_filter import (
    SteadyRegularizedKalmanFilter,
)
from iskf.filters.weighted_likelihood_filter import WeightedLikelihoodFilter
from iskf.filters.steady_one_step_huber_filter import SteadyOneStepHuberFilter
from iskf.filters.steady_two_step_huber_filter import SteadyTwoStepHuberFilter
from iskf.filters.steady_three_term_huber import SteadyThreeTermHuberFilter
from iskf.filters.kalman_filter import KalmanFilter
from iskf.filters.steady_kalman_filter import SteadyKalmanFilter

# Plotting functions
from iskf.visualization.plot_sweep_contour import plot_contour_results

# --- Global Configuration ---
SWEEP_RESOLUTION = 20
DEFAULT_METRIC = "rmse"
# STEP_SIZE_VALUES = np.geomspace(0.1, 100.0, SWEEP_RESOLUTION)
STEP_SIZE_VALUES = [1.0]

coef_s_min = 1e-1
coef_s_max = 10.0
coef_o_min = 1e-1
coef_o_max = 10.0

# # vehicle seed 0, optimistic for contour plot
# coef_s_min = 1e-1
# coef_s_max = 1.5
# coef_o_min = 0.9
# coef_o_max = 5.0

# # cstr seed 0, optimistic for contour plot
# coef_s_min = 1e-1
# coef_s_max = 1.5
# coef_o_min = 0.5
# coef_o_max = 10.0

coef_s_sweep = np.geomspace(coef_s_min, coef_s_max, SWEEP_RESOLUTION)
coef_o_sweep = np.geomspace(coef_o_min, coef_o_max, SWEEP_RESOLUTION)
coef_s_sweep = np.append(coef_s_sweep, np.inf)
coef_o_sweep = np.append(coef_o_sweep, np.inf)

NUM_PARALLEL_JOBS_RUN = -1
PARAMETER_SEARCH_DIR = os.path.join("results", "parameter_search_data")
PARAMETER_SEARCH_PLOTS = "figures"
EVALUATION_RESULTS_DIR = os.path.join("results", "evaluations")
SUPPORTED_FILTER_TYPES = [
    "huber",
    "circular_huber",
    "steady_huber",
    "steady_circular_huber",
    "steady_regularized",
    "wolf",
    "steady_one_step_huber",
    "steady_two_step_huber",
    "steady_three_term_huber",
]

# --- Filter Tuning Helper Functions ---


def _tune_circular_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes HuberizedSSKalmanFilter."""
    print("\n  Step 2: Setting up HSSKF grid search parameters...")
    param_grid = [
        {"coef_s": cs, "coef_o": co} for cs in coef_s_sweep for co in coef_o_sweep
    ]
    initial_filter = CircularHuberKalmanFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )
    base_filename = f"circular_huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for CircularHuberKalmanFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )
    print(
        f"\n--- CircularHuberKalmanFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    if "coef_s" in best_params:
        print(f"  coef_s: {best_params['coef_s']:.4f}")
    if "coef_o" in best_params:
        print(f"  coef_o: {best_params['coef_o']:.4f}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata - store grid search details and consolidated info
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "circular_huber",
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
        "test_data_path": None,  # Will be updated if test evaluation is performed
    }

    # Save both the detailed grid search results and the consolidated results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All CircularHuberKalmanFilter grid search results saved to {results_filename}"
    )

    print(f"\n  Step 4: Generating contour plot for CircularHuberKalmanFilter...")
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"{base_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "coef_s",
        "y_param_name": "coef_o",
        "x_display_name": r"$\lambda_p$",
        "y_display_name": r"$\lambda_m$",
        "levels": 25,
        "use_log_scale": False,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"CircularHuberKalmanFilter contour plot saved to {plot_save_path}")
    return best_params, initial_filter, best_score


def _tune_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes HuberRegressionFilter."""
    print("\n  Step 2: Setting up HRF grid search parameters...")
    scale_cov_update_setting = False

    param_grid = [
        {"coef_s": cs, "coef_o": co} for cs in coef_s_sweep for co in coef_o_sweep
    ]
    initial_filter = HuberKalmanFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )
    base_filename = f"huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for HuberKalmanFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )
    print(
        f"\n--- HuberKalmanFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    if "coef_s" in best_params:
        print(f"  coef_s: {best_params['coef_s']:.4f}")
    if "coef_o" in best_params:
        print(f"  coef_o: {best_params['coef_o']:.4f}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata - store grid search details and consolidated info
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "huber",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save both the detailed grid search results and the consolidated results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(f"All HuberKalmanFilter grid search results saved to {results_filename}")

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"huber_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "huber",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(f"Consolidated HuberKalmanFilter results saved to {consolidated_filename}")

    print(f"\n  Step 4: Generating contour plot for HuberKalmanFilter...")
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"{base_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "coef_s",
        "y_param_name": "coef_o",
        "x_display_name": r"$\lambda_p$",
        "y_display_name": r"$\lambda_m$",
        "levels": 25,
        "use_log_scale": True,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"HuberKalmanFilter contour plot saved to {plot_save_path}")
    return best_params, initial_filter, best_score


def _plot_wolf_results_lines(
    all_sweeps_data: dict,
    metric_name: str,
    save_path: str = None,
    x_axis_label: str = "Coefficient (coef)",
    y_axis_label: str = None,
    title: str = None,
):
    """Plots metric vs. coef for different WoLF weighting types on a single graph."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    if y_axis_label is None:
        y_axis_label = metric_name
    for weighting_type, data_dict in all_sweeps_data.items():
        if not data_dict:
            print(f"Warning: No data for weighting type {weighting_type}, skipping.")
            continue

        finite_coeffs_scores = {}
        inf_coeffs_scores = {}  # Store original key and score for inf

        for c_key, score in data_dict.items():
            try:
                c_float = float(c_key)
                if np.isinf(c_float):
                    inf_coeffs_scores[c_key] = score
                elif not np.isnan(c_float):  # Ignore NaN coefficients
                    finite_coeffs_scores[c_float] = score
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert coefficient '{c_key}' to float for '{weighting_type}'. Skipping point."
                )

        if not finite_coeffs_scores:
            if inf_coeffs_scores:  # only inf scores exist for this weighting_type
                print(
                    f"Info: Weighting type '{weighting_type}' only has data for infinite coefficient(s). No line plotted."
                )
                # Attempt to add to legend even if no line, so all types are listed
                plt.plot([], [], linestyle="None", label=str(weighting_type))
            else:  # no valid data at all for this weighting_type
                print(
                    f"Warning: No valid finite coefficient data for weighting type '{weighting_type}'. Skipping plot for this type."
                )
            # In either case of no finite data, print any inf scores and continue
            for c_key_inf, score_inf in inf_coeffs_scores.items():
                print(
                    f"    Score for '{str(weighting_type)}' at coef {c_key_inf}: {score_inf:.4f}"
                )
            continue  # Skip to next weighting_type

        # Proceed to plot if there is finite data
        sorted_coefs = sorted(finite_coeffs_scores.keys())
        sorted_scores = [finite_coeffs_scores[k] for k in sorted_coefs]

        plt.plot(
            sorted_coefs,
            sorted_scores,
            marker="o",
            linestyle="-",
            label=str(weighting_type),
        )

        # After plotting the line for finite coeffs, print info about inf coeffs for this type
        if inf_coeffs_scores:
            print(
                f"    Note for '{str(weighting_type)}':"
            )  # Clarify which weighting type the note is for
            for c_key_inf, score_inf in inf_coeffs_scores.items():
                print(
                    f"      Score at coefficient {c_key_inf} is {score_inf:.4f} (not shown on line plot)."
                )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title:
        plt.title(title)
    plt.legend(title="Weighting Type")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Line plot saved to {save_path}")


def _tune_wolf(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes WoLF filter with one-dimensional grid search over coefficient."""
    print("\n  Step 2: Setting up WoLF grid search parameters...")
    # Convert single sweep_res value to match size of coef sweep we want to run
    if sweep_res != 21:  # If default is changed
        coef_sweep_res = sweep_res
    else:
        coef_sweep_res = 40  # More fine-grained for WoLF

    # Wolf needs its own specialized range - from literature
    # Extend range of coefficients for thorough exploration
    coef_min = 1e-1
    coef_max = 20.0  # upper practical limit

    # Create the parameter grid with a logarithmic spacing for the coefficient
    coef_values = np.geomspace(coef_min, coef_max, coef_sweep_res)

    # Add infinity to test the unweighted case
    coef_values = np.append(coef_values, np.inf)

    param_grid = [{"coef": c} for c in coef_values]
    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = WeightedLikelihoodFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    # File name base
    base_filename = f"wolf_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for WoLF... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )

    # Filter for only valid results (in case some combinations failed)
    valid_results = {}
    for params, score in all_results.items():
        if not np.isnan(score) and not np.isinf(score):
            valid_results[params] = score

    # Find the best parameters
    best_score_valid = float("inf")
    best_params_valid = None
    for params, score in valid_results.items():
        if score < best_score_valid:
            best_score_valid = score
            best_params_valid = dict(params)

    # If we found valid results, update the best parameters
    if best_params_valid is not None:
        best_params = best_params_valid
        best_score = best_score_valid

    print(
        f"\n--- Weighted Likelihood Filter (WoLF) Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "wolf",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save the grid search results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"wolf_{sim_data_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(f"All WoLF grid search results saved to {results_filename}")

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"wolf_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "wolf",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(f"Consolidated WoLF results saved to {consolidated_filename}")

    # Create a line plot of coefficient vs. error metric
    print(
        f"\n  Step 4: Generating line plot for WoLF coefficient vs. {metric.upper()}..."
    )
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"wolf_{sim_data_filename}{opt_suffix}{metric_suffix}_line_plot.pdf",
    )

    # Process all_results to prepare for plotting
    coef_to_error = {}
    for params, score in valid_results.items():
        params_dict = dict(params)
        coef = params_dict.get("coef", None)
        if coef is not None:
            coef_to_error[coef] = score

    # Sort by coefficient for line plot
    # sorted_coef_error = sorted(coef_to_error.items()) # This sorting is handled by the plot function
    # coefs, errors = zip(*sorted_coef_error) if sorted_coef_error else ([], []) # Also handled by plot function

    # Plot the coefficient vs. error curve
    # _plot_wolf_results_lines expects all_sweeps_data to be a dict of dicts,
    # e.g., {'WeightingType1': {coef1: score1,...}, 'WeightingType2': {coefA: scoreA,...}}
    # For WoLF, we have a single series of coef vs. error.
    plot_data_structured = {"WoLF": coef_to_error}

    _plot_wolf_results_lines(
        all_sweeps_data=plot_data_structured,  # Pass the correctly structured data
        metric_name=metric.upper(),
        save_path=plot_save_path,
        x_axis_label=r"$\\lambda$ Coefficient",
        y_axis_label=f"{metric.upper()} Error",
        title=f"WoLF {metric.upper()} vs. $\\lambda$ Coefficient",
    )
    print(f"WoLF line plot saved to {plot_save_path}")

    return best_params, initial_filter, best_score


def _tune_steady_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """
    Tunes SteadyHuberKalmanFilter with grid search over coef_s, coef_o.
    """
    print("\n  Step 2: Setting up SteadyHuberKalmanFilter grid search parameters...")
    param_grid = [
        {"coef_s": cs, "coef_o": co} for cs in coef_s_sweep for co in coef_o_sweep
    ]
    initial_filter = SteadyHuberKalmanFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )
    base_filename = f"steady_huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for SteadyHuberKalmanFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )
    print(
        f"\n--- SteadyHuberKalmanFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    if "coef_s" in best_params:
        print(f"  coef_s: {best_params['coef_s']:.4f}")
    if "coef_o" in best_params:
        print(f"  coef_o: {best_params['coef_o']:.4f}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_huber",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyHuberKalmanFilter grid search results saved to {results_filename}"
    )

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "steady_huber",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(
        f"Consolidated SteadyHuberKalmanFilter results saved to {consolidated_filename}"
    )

    print(f"\n  Step 4: Generating contour plot for SteadyHuberKalmanFilter...")
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"{base_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "coef_s",
        "y_param_name": "coef_o",
        "x_display_name": r"$\lambda_p$",  # Or a more specific name like r"$c_s$"
        "y_display_name": r"$\lambda_m$",  # Or a more specific name like r"$c_o$"
        "levels": sweep_res,  # Use sweep_res for contour levels
        "use_log_scale": True,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"SteadyHuberKalmanFilter contour plot saved to {plot_save_path}")
    return best_params, initial_filter, best_score


def _tune_steady_circular_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    print(
        f"\n  Step 2: Setting up SteadyCircularHuberKalmanFilter grid search parameters..."
    )
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    # Create a single param_grid with all combinations
    param_grid = []

    # Add finite iterations parameters with different step sizes
    for step_size_current in STEP_SIZE_VALUES:
        for cs in coef_s_sweep:
            for co in coef_o_sweep:
                param_grid.append(
                    {"coef_s": cs, "coef_o": co, "step_size": step_size_current}
                )

    # # Add infinite num_iters case
    # for cs in coef_s_sweep:
    #     for co in coef_o_sweep:
    #         param_grid.append({"coef_s": cs, "coef_o": co, "num_iters": np.inf})

    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = SteadyCircularHuberKalmanFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    print(
        f"\n  Step 3: Running grid search for SteadyCircularHuberKalmanFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )
    print(
        f"\n--- SteadyCircularHuberKalmanFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_circular_huber",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save the results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_circular_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyCircularHuberKalmanFilter grid search results saved to {results_filename}"
    )

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_circular_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "steady_circular_huber",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(
        f"Consolidated SteadyCircularHuberKalmanFilter results saved to {consolidated_filename}"
    )

    # Create filtered views of results by parameter configurations
    # This helps with visualizing the 2D slices for each configuration
    step_size_filtered_results = {}
    inf_iters_results = {}

    for params, score in all_results.items():
        params_dict = dict(params)
        if "num_iters" in params_dict and params_dict["num_iters"] == np.inf:
            inf_iters_filtered = {
                "coef_s": params_dict["coef_s"],
                "coef_o": params_dict["coef_o"],
            }
            inf_iters_results[tuple(inf_iters_filtered.items())] = score
        elif "step_size" in params_dict:
            step_size = params_dict["step_size"]
            if step_size not in step_size_filtered_results:
                step_size_filtered_results[step_size] = {}
            filtered_params = {
                "coef_s": params_dict["coef_s"],
                "coef_o": params_dict["coef_o"],
            }
            step_size_filtered_results[step_size][
                tuple(filtered_params.items())
            ] = score

    # Plot contours for infinite iterations
    if inf_iters_results:
        plot_save_path = os.path.join(
            PARAMETER_SEARCH_PLOTS,
            f"steady_circular_huber_inf_iters_{sim_data_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
        )
        plot_args = {
            "x_param_name": "coef_s",
            "y_param_name": "coef_o",
            "x_display_name": r"$\lambda_p$",
            "y_display_name": r"$\lambda_m$",
            "levels": 20,
            "use_log_scale": True,  # Use log scale for better visualization
        }
        plot_contour_results(
            sweep_results=inf_iters_results,
            metric_name=metric.upper(),
            save_path=plot_save_path,
            **plot_args,
        )
        print(
            f"SteadyCircularHuberKalmanFilter contour plot for infinite iterations saved to {plot_save_path}"
        )

    # Plot contours for each step_size for visualization
    for step_size, results in step_size_filtered_results.items():
        plot_save_path = os.path.join(
            PARAMETER_SEARCH_PLOTS,
            f"steady_circular_huber_step{step_size}_{sim_data_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
        )
        plot_args = {
            "x_param_name": "coef_s",
            "y_param_name": "coef_o",
            "x_display_name": r"$\lambda_p$",
            "y_display_name": r"$\lambda_m$",
            "levels": 20,
            "use_log_scale": True,  # Use log scale for better visualization
        }
        plot_contour_results(
            sweep_results=results,
            metric_name=metric.upper(),
            save_path=plot_save_path,
            **plot_args,
        )
        print(
            f"SteadyCircularHuberKalmanFilter contour plot for step_size={step_size} saved to {plot_save_path}"
        )

    return best_params, initial_filter, best_score


def _tune_steady_regularized(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes SteadyRegularizedKalmanFilter with a 2D grid search over regularization parameters."""
    print(
        f"\n  Step 2: Setting up SteadyRegularizedKalmanFilter grid search parameters..."
    )

    # For regularized filter, we'll use a slightly different range that makes sense for the regularization parameters
    alpha_min = 1e-5
    alpha_max = 1.0
    beta_min = 1e-5
    beta_max = 1.0

    # Create logarithmically spaced grid of regularization parameters
    alpha_sweep = np.geomspace(alpha_min, alpha_max, sweep_res)
    beta_sweep = np.geomspace(beta_min, beta_max, sweep_res)

    # Include infinity for comparison
    alpha_sweep = np.append(alpha_sweep, np.inf)
    beta_sweep = np.append(beta_sweep, np.inf)

    param_grid = [{"alpha": a, "beta": b} for a in alpha_sweep for b in beta_sweep]
    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = SteadyRegularizedKalmanFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    # File name base
    base_filename = f"steady_regularized_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for SteadyRegularizedKalmanFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )

    # Print the best results
    print(
        f"\n--- SteadyRegularizedKalmanFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_regularized",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save the grid search results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_regularized_{sim_data_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyRegularizedKalmanFilter grid search results saved to {results_filename}"
    )

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_regularized_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "steady_regularized",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(
        f"Consolidated SteadyRegularizedKalmanFilter results saved to {consolidated_filename}"
    )

    # Generate a contour plot of the results
    print(f"\n  Step 4: Generating contour plot for SteadyRegularizedKalmanFilter...")
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"steady_regularized_{sim_data_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "alpha",
        "y_param_name": "beta",
        "x_display_name": r"$\alpha$",
        "y_display_name": r"$\beta$",
        "levels": 25,
        "use_log_scale": True,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"SteadyRegularizedKalmanFilter contour plot saved to {plot_save_path}")

    return best_params, initial_filter, best_score


def _plot_steady_one_step_huber_results_lines(
    all_sweeps_data: dict,  # Expected: {series_name_1: {coef1: score1, ...}, series_name_2: {coefA: scoreA, ...}}
    metric_name: str,
    save_path: str = None,
    x_axis_label: str = r"$\lambda$",
    y_axis_label: str = None,
    title: str = None,
):
    """Plots metric vs. coef for different SteadyOneStepHuberFilter configurations (e.g., different series labels)."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    if y_axis_label is None:
        y_axis_label = metric_name

    # The loop iterates over series (e.g., different step_sizes or other configurations if they were present)
    # In the current _tune_steady_one_step_huber, there will be only one series named "SteadyOneStepHuber"
    for series_label, coef_score_map in all_sweeps_data.items():
        if not coef_score_map:
            print(f"Warning: No data for series '{series_label}', skipping.")
            continue

        finite_coeffs_scores = {}
        inf_coeffs_scores = {}  # Store original key and score for inf

        for (
            c_key,
            score,
        ) in (
            coef_score_map.items()
        ):  # Iterate over coef:score pairs for the current series
            try:
                c_float = float(c_key)
                if np.isinf(c_float):
                    inf_coeffs_scores[c_key] = score
                elif not np.isnan(c_float):  # Ignore NaN coefficients
                    finite_coeffs_scores[c_float] = score
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert coefficient '{c_key}' to float for series '{series_label}'. Skipping point."
                )

        if not finite_coeffs_scores:
            if inf_coeffs_scores:  # only inf scores exist for this series
                print(
                    f"Info: Series '{series_label}' only has data for infinite coefficient(s). No line plotted."
                )
                # Attempt to add to legend even if no line, so all series are listed
                plt.plot([], [], linestyle="None", label=str(series_label))
            else:  # no valid data at all for this series
                print(
                    f"Warning: No valid finite coefficient data for series '{series_label}'. Skipping plot for this series."
                )
            # In either case of no finite data, print any inf scores and continue
            for c_key_inf, score_inf in inf_coeffs_scores.items():
                print(
                    f"    Score for '{str(series_label)}' at coef {c_key_inf}: {score_inf:.4f}"
                )
            continue  # Skip to next series

        # Proceed to plot if there is finite data for the current series
        sorted_coefs = sorted(finite_coeffs_scores.keys())
        sorted_scores = [finite_coeffs_scores[k] for k in sorted_coefs]

        plt.plot(
            sorted_coefs,
            sorted_scores,
            marker="o",
            linestyle="-",
            label=str(
                series_label
            ),  # Use series_label for the legend (e.g., "SteadyOneStepHuber")
        )

        # After plotting the line for finite coeffs, print info about inf coeffs for this series
        if inf_coeffs_scores:
            print(f"    Note for series '{str(series_label)}':")
            for c_key_inf, score_inf in inf_coeffs_scores.items():
                print(
                    f"      Score at coefficient {c_key_inf} is {score_inf:.4f} (not shown on line plot)."
                )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title:
        plt.title(title)
    plt.legend(title="Series")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Line plot saved to {save_path}")


def _tune_steady_one_step_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,  # state dimension, not directly used for coef range here, but ny is.
    ny,  # measurement dimension, used for coef range like WoLF's Huber.
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes SteadyOneStepHuberFilter with a one-dimensional grid search over 'coef'."""
    print(f"\n  Step 2: Setting up SteadyOneStepHuberFilter grid search parameters...")

    # One-step Huber like WoLF needs a specialized range.
    coef_min = 1e-1
    coef_max = 20.0  # upper practical limit based on WoLF experience

    # Create finer resolution for 1D parameter sweep
    coef_sweep_res = sweep_res * 2 if sweep_res < 30 else sweep_res

    # Create the parameter grid with a logarithmic spacing
    coef_values = np.geomspace(coef_min, coef_max, coef_sweep_res)

    # Add infinity to test the unweighted case
    coef_values = np.append(coef_values, np.inf)

    param_grid = [
        {"coef": c, "step_size": step_size}
        for c in coef_values
        for step_size in STEP_SIZE_VALUES
    ]
    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = SteadyOneStepHuberFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    # File name base
    base_filename = f"steady_one_step_huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for SteadyOneStepHuberFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )

    # Filter out invalid results (NaN or inf scores)
    valid_results = {
        params: score
        for params, score in all_results.items()
        if not np.isnan(score) and not np.isinf(score)
    }

    # Recompute best parameters based on valid results if needed
    if not valid_results:
        print("Warning: No valid results found. Using original grid search results.")
    else:
        best_score_valid = float("inf")
        best_params_valid = None
        for params, score in valid_results.items():
            if score < best_score_valid:
                best_score_valid = score
                best_params_valid = dict(params)

        if best_params_valid:
            best_params = best_params_valid
            best_score = best_score_valid

    # Print the best results
    print(
        f"\n--- SteadyOneStepHuberFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_one_step_huber",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save the grid search results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_one_step_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyOneStepHuberFilter grid search results saved to {results_filename}"
    )

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_one_step_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "steady_one_step_huber",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(
        f"Consolidated SteadyOneStepHuberFilter results saved to {consolidated_filename}"
    )

    # Generate a line plot of the coefficient vs. error metric
    print(
        f"\n  Step 4: Generating line plot for SteadyOneStepHuberFilter coefficient vs. {metric.upper()}..."
    )
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"steady_one_step_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_line_plot.pdf",
    )

    # Process for plotting
    coef_to_error = {}
    for params, score in valid_results.items():
        params_dict = dict(params)
        coef = params_dict.get("coef", None)
        if coef is not None:
            coef_to_error[coef] = score

    # Structure data for plotting: {series_name: {coef: score}}
    # For this filter, we only have one series of results.
    plot_data_structured = {"SteadyOneStepHuber": coef_to_error}

    # Create the line plot
    _plot_steady_one_step_huber_results_lines(
        all_sweeps_data=plot_data_structured,  # Pass the correctly structured data
        metric_name=metric.upper(),
        save_path=plot_save_path,
        x_axis_label=r"$\lambda$ Coefficient",
        y_axis_label=f"{metric.upper()} Error",
        title=f"SteadyOneStepHuberFilter {metric.upper()} vs. $\\lambda$ Coefficient",
    )
    print(f"SteadyOneStepHuberFilter line plot saved to {plot_save_path}")

    return best_params, initial_filter, best_score


def _plot_steady_two_step_huber_results_lines(
    all_sweeps_data: dict,
    metric_name: str,
    save_path: str = None,
    x_axis_label: str = r"$\lambda$",
    y_axis_label: str = None,
    title: str = None,
):
    """Plots metric vs. coef_s/coef_o for different step_size values on a single graph."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    if y_axis_label is None:
        y_axis_label = metric_name
    for step_size_val, data_dict in all_sweeps_data.items():
        if not data_dict:
            print(f"Warning: No data for step_size {step_size_val}, skipping.")
            continue
        sorted_coefs = sorted(data_dict.keys())
        sorted_scores = [data_dict[c] for c in sorted_coefs]
        plt.plot(
            sorted_coefs,
            sorted_scores,
            marker="o",
            linestyle="-",
            label=f"step_size={step_size_val:.2f}",
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title:
        plt.title(title)
    plt.legend(title="Step Size")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Line plot saved to {save_path}")


def _tune_steady_two_step_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes SteadyTwoStepHuberFilter with a two-dimensional grid search over coef_s and coef_o."""
    print(f"\n  Step 2: Setting up SteadyTwoStepHuberFilter grid search parameters...")
    param_grid = [
        {"coef_s": cs, "coef_o": co, "step_size": step_size}
        for cs in coef_s_sweep
        for co in coef_o_sweep
        for step_size in STEP_SIZE_VALUES
    ]
    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = SteadyTwoStepHuberFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    # File name base
    base_filename = f"steady_two_step_huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for SteadyTwoStepHuberFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )

    # Filter out invalid results (NaN or inf scores)
    valid_results = {
        params: score
        for params, score in all_results.items()
        if not np.isnan(score) and not np.isinf(score)
    }

    # Recompute best parameters based on valid results if needed
    if not valid_results:
        print("Warning: No valid results found. Using original grid search results.")
    else:
        best_score_valid = float("inf")
        best_params_valid = None
        for params, score in valid_results.items():
            if score < best_score_valid:
                best_score_valid = score
                best_params_valid = dict(params)

        if best_params_valid:
            best_params = best_params_valid
            best_score = best_score_valid

    # Print the best results
    print(
        f"\n--- SteadyTwoStepHuberFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    if "coef_s" in best_params:
        print(f"  coef_s: {best_params['coef_s']:.4f}")
    if "coef_o" in best_params:
        print(f"  coef_o: {best_params['coef_o']:.4f}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_two_step_huber",
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
        "test_data_path": None,  # Will be updated if test evaluation is performed
    }

    # Save the grid search results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyTwoStepHuberFilter grid search results saved to {results_filename}"
    )

    # Generate a contour plot of the coefficient combinations
    print(
        f"\n  Step 4: Generating contour plot for SteadyTwoStepHuberFilter coefficients..."
    )
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"{base_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "coef_s",
        "y_param_name": "coef_o",
        "x_display_name": r"$\lambda_x$",
        "y_display_name": r"$\lambda_y$",
        "levels": SWEEP_RESOLUTION,
        "use_log_scale": False,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"SteadyTwoStepHuberFilter contour plot saved to {plot_save_path}")

    return best_params, initial_filter, best_score


def _tune_steady_three_term_huber(
    system_model_obj,
    cov_input,
    cov_measurement,
    T_out,
    Y_meas,
    X_states,
    x0_est,
    P0_init,
    nx,
    ny,
    sweep_res,
    metric,
    n_jobs,
    optimistic,
    sim_data_filename,
    sim_data_path,
):
    """Tunes SteadyThreeTermHuberFilter with a two-dimensional grid search over coef_s, coef_o."""
    print(
        f"\n  Step 2: Setting up SteadyThreeTermHuberFilter grid search parameters..."
    )

    # Create parameter grid for coef_s and coef_o as in the other Huber filters
    param_grid = [
        {"coef_s": cs, "coef_o": co} for cs in coef_s_sweep for co in coef_o_sweep
    ]
    print(f"    Generated param_grid with {len(param_grid)} parameter combinations")

    # Create initial filter with default values
    initial_filter = SteadyThreeTermHuberFilter(
        system_model=system_model_obj,
        cov_input=cov_input,
        cov_measurement=cov_measurement,
    )

    # File name base
    base_filename = f"steady_three_term_huber_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    print(
        f"\n  Step 3: Running grid search for SteadyThreeTermHuberFilter... (Optimistic mode: {optimistic}, Metric: {metric.upper()})"
    )
    best_params, best_score, all_results = grid_search_filter_hyperparams(
        initial_filter,
        param_grid,
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs,
    )

    # Print the best results
    print(
        f"\n--- SteadyThreeTermHuberFilter Grid Search Results (Optimistic: {optimistic}, Metric: {metric.upper()}) ---"
    )
    print(f"Best Hyperparameters ({metric.upper()}): {best_params}")
    if "coef_s" in best_params:
        print(f"  coef_s: {best_params['coef_s']:.4f}")
    if "coef_o" in best_params:
        print(f"  coef_o: {best_params['coef_o']:.4f}")
    print(f"Best Score: {best_score:.4f}")

    # Create an extended results dictionary with metadata
    extended_results = {
        "grid_search_results": all_results,
        "best_params": best_params,
        "best_score": best_score,
        "filter_type": "steady_three_term_huber",
        "sim_data_path": sim_data_path,
        "metric": metric,
        "optimistic": optimistic,
        "evaluated_on_test": False,  # This will be updated if test evaluation is performed
    }

    # Save the grid search results
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_three_term_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_grid_search_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(extended_results, f)
    print(
        f"All SteadyThreeTermHuberFilter grid search results saved to {results_filename}"
    )

    # Also save the consolidated results
    consolidated_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"steady_three_term_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    consolidated_results = {
        "filter_type": "steady_three_term_huber",
        "best_params": best_params,
        "best_score": best_score,
        "sim_data_path": os.path.abspath(sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": None,  # Will be updated if test evaluation is performed
        "evaluated_on_test": False,
    }
    with open(consolidated_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(
        f"Consolidated SteadyThreeTermHuberFilter results saved to {consolidated_filename}"
    )

    # Generate a contour plot of the coefficient combinations
    print(
        f"\n  Step 4: Generating contour plot for SteadyThreeTermHuberFilter coefficients..."
    )
    plot_save_path = os.path.join(
        PARAMETER_SEARCH_PLOTS,
        f"steady_three_term_huber_{sim_data_filename}{opt_suffix}{metric_suffix}_contour_plot.pdf",
    )
    plot_args = {
        "x_param_name": "coef_s",
        "y_param_name": "coef_o",
        "x_display_name": r"$\lambda_p$",
        "y_display_name": r"$\lambda_m$",
        "levels": 25,
        "use_log_scale": True,  # Use log scale for better visualization
    }
    plot_contour_results(
        sweep_results=all_results,
        metric_name=metric.upper(),
        save_path=plot_save_path,
        **plot_args,
    )
    print(f"SteadyThreeTermHuberFilter contour plot saved to {plot_save_path}")

    return best_params, initial_filter, best_score


def _evaluate_filter_on_test_data(
    filter_instance,
    T_out_test,
    Y_measurements_test,
    X_true_states_test,
    x0_estimate_filter_test,
    P0_initial_val_test,
):
    """
    Helper function to evaluate a filter on test data and compute metrics.

    Args:
        filter_instance: The filter to evaluate
        T_out_test: Time values for test data
        Y_measurements_test: Measurements from test data
        X_true_states_test: True states from test data
        x0_estimate_filter_test: Initial state estimate for test
        P0_initial_val_test: Initial covariance for test

    Returns:
        tuple: (X_predicted_test, evaluated_metrics_dict)
    """
    print(f"    Running {filter_instance.__class__.__name__} on test data...")

    # Reset the filter with initial conditions for test data
    filter_instance.reset(x0_estimate_filter_test, P0_initial_val_test)

    # Run the filter on test data
    X_predicted_test = filter_instance.estimate(
        T_out_test,
        Y_measurements_test,
        x_initial_estimate=x0_estimate_filter_test,
        P_initial=P0_initial_val_test,
    )

    # Compute metrics
    # Always compare with X_true_states_test[:, 1:] to match grid_search.py logic
    x_true_for_metric_eval = X_true_states_test[:, 1:]
    print(
        f"    Using X_true_states_test[:, 1:] ({x_true_for_metric_eval.shape}) for consistent metric calculation with grid_search.py"
    )

    # Verify shapes are compatible
    if X_predicted_test.shape != x_true_for_metric_eval.shape:
        print(
            f"    Error: Shape mismatch for metrics. X_predicted_test: {X_predicted_test.shape}, X_true_for_metric_eval: {x_true_for_metric_eval.shape}."
        )
        return X_predicted_test, {}

    evaluated_metrics_dict = {}
    for metric_name_eval, metric_func_eval in METRIC_REGISTRY.items():
        try:
            score_eval = metric_func_eval(X_predicted_test, x_true_for_metric_eval)
            print(f"      {metric_name_eval.upper()}: {score_eval:.4f}")
            evaluated_metrics_dict[metric_name_eval] = score_eval
        except Exception as e:
            print(f"      Error calculating {metric_name_eval.upper()}: {e}")
            evaluated_metrics_dict[metric_name_eval] = None

    return X_predicted_test, evaluated_metrics_dict


# --- Main Dispatch Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for a specified filter."
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        required=True,
        choices=SUPPORTED_FILTER_TYPES,
        help=f"Type of filter to tune. Choices: {SUPPORTED_FILTER_TYPES}",
    )
    parser.add_argument(
        "--sim_data_path",
        type=str,
        required=True,
        help="Path to the pickled simulation data file generated by vehicle_example.py.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to pickled test data for evaluation. If None, no evaluation is performed.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help=f"Random seed for simulation. Default: {42}",
    )
    parser.add_argument(
        "--optimistic",
        action="store_true",
        help="If set, use true states for metric calculation (optimistic mode). Otherwise, use predicted vs actual measurements (realistic mode).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=list(METRIC_REGISTRY.keys()),
        help=f"Metric to use for hyperparameter evaluation. Default: {DEFAULT_METRIC}. Choices: {list(METRIC_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--sweep_resolution",
        type=int,
        default=20,  # Default value set here
        help="Number of points for each hyperparameter dimension in grid search. Default: 20",
    )
    args = parser.parse_args()
    filter_type_to_tune = args.filter_type.lower()
    np.random.seed(args.random_seed)
    optimistic = args.optimistic
    metric = args.metric.lower()
    sweep_resolution_arg = args.sweep_resolution  # Use the parsed argument
    test_data_path = args.test_data_path

    # Extract the base filename without extension
    sim_data_filename = pathlib.Path(args.sim_data_path).stem

    print(f"Starting hyperparameter tuning for: {filter_type_to_tune.upper()}...")
    print(
        f"Optimistic mode: {optimistic} (Using {'true states' if optimistic else 'predicted measurements'} for metrics)"
    )
    print(f"Metric: {metric.upper()}")
    print(f"Simulation data: {sim_data_filename}")

    # Create directories for saving results
    os.makedirs(PARAMETER_SEARCH_DIR, exist_ok=True)
    os.makedirs(PARAMETER_SEARCH_PLOTS, exist_ok=True)
    os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)

    # Load simulation data from the provided pickle file
    print(f"  Step 1: Loading simulation data from {args.sim_data_path}...")
    try:
        with open(args.sim_data_path, "rb") as f:
            loaded_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Simulation data file not found at {args.sim_data_path}")
        return
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        return

    # Handle different ways of specifying the process noise covariance
    try:
        # Unpack loaded_data into the variables previously returned by _setup_simulation()
        try:
            model_type_loaded = loaded_data["model_type"]
            model_params_loaded = loaded_data["model_params"]

            # Reconstruct the system model object
            if model_type_loaded == "vehicle_ss":
                system_model_obj = vehicle_ss(**model_params_loaded)
                print(
                    f"    System model object '{model_type_loaded}' reconstructed successfully."
                )
            elif model_type_loaded == "cascaded_cstr_ss":
                system_model_obj = cascaded_cstr_ss(**model_params_loaded)
                print(
                    f"    System model object '{model_type_loaded}' reconstructed successfully."
                )
            else:
                print(
                    f"Error: Unknown model_type '{model_type_loaded}' in simulation data."
                )
                return

            # Load covariance matrices using the new keys
            cov_input_loaded = loaded_data["cov_input"]
            cov_measurement_loaded = loaded_data["cov_measurement"]
            print("    Loaded cov_input and cov_measurement from simulation data.")

            T_out = loaded_data["T_out"]
            Y_measurements = loaded_data["Y_measurements"]
            X_true_states = loaded_data["X_true_states"]
            x0_estimate_filter = loaded_data["x0_estimate_filter"]
            P0_initial_val = loaded_data["P0_initial_val"]
            nx_sim_val = loaded_data["nx_sim_val"]
            ny_sim_val = loaded_data["ny_sim_val"]
            print("    Simulation data loaded successfully.")
        except KeyError as e:
            print(
                f"Error: Missing key {e} in the loaded simulation data. Ensure the data file is correctly generated."
            )
            return

    except KeyError as e:
        print(
            f"Error: Missing key {e} in the loaded simulation data. Ensure the data file is correctly generated."
        )
        return

    # Store common arguments for tuning functions
    common_tune_args = {
        "system_model_obj": system_model_obj,
        "cov_input": cov_input_loaded,
        "cov_measurement": cov_measurement_loaded,
        "T_out": T_out,
        "Y_meas": Y_measurements,
        "X_states": X_true_states,
        "x0_est": x0_estimate_filter,
        "P0_init": P0_initial_val,
        "nx": nx_sim_val,
        "ny": ny_sim_val,
        "sweep_res": sweep_resolution_arg,
        "metric": metric,
        "n_jobs": NUM_PARALLEL_JOBS_RUN,
        "optimistic": optimistic,
        "sim_data_filename": sim_data_filename,
        "sim_data_path": os.path.abspath(args.sim_data_path),
    }

    tune_function_dispatch = {
        "circular_huber": _tune_circular_huber,
        "huber": _tune_huber,
        "wolf": _tune_wolf,
        "steady_huber": _tune_steady_huber,
        "steady_circular_huber": _tune_steady_circular_huber,
        "steady_regularized": _tune_steady_regularized,
        "steady_one_step_huber": _tune_steady_one_step_huber,
        "steady_two_step_huber": _tune_steady_two_step_huber,
        "steady_three_term_huber": _tune_steady_three_term_huber,
    }

    if filter_type_to_tune not in tune_function_dispatch:
        print(f"Error: Unknown filter type '{filter_type_to_tune}' specified.")
        return

    tune_function = tune_function_dispatch[filter_type_to_tune]

    # Call the appropriate tuning function
    # These functions now return: best_overall_params, initial_filter_template, best_overall_score
    best_hyperparams, initial_filter_template, best_tune_score = tune_function(
        **common_tune_args
    )

    # Generate a consolidated results pickle file
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    # Define a single consolidated results file path
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{filter_type_to_tune}_{sim_data_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )

    # Create a consolidated results dictionary
    consolidated_results = {
        "filter_type": filter_type_to_tune,
        "best_params": best_hyperparams,
        "best_score": best_tune_score,
        "sim_data_path": os.path.abspath(args.sim_data_path),
        "metric": metric,
        "optimistic": optimistic,
        "test_data_path": os.path.abspath(test_data_path) if test_data_path else None,
        "evaluated_on_test": False,  # Will be updated if test evaluation is performed
    }

    # Save consolidated results
    with open(results_filename, "wb") as f:
        pickle.dump(consolidated_results, f)
    print(f"\nConsolidated tuning results saved to: {results_filename}")

    if test_data_path:
        print(f"\n--- Evaluating tuned filter on test data: {test_data_path} ---")
        try:
            with open(test_data_path, "rb") as f_test:
                test_data_loaded = pickle.load(f_test)

            Y_measurements_test = test_data_loaded["Y_measurements"]
            X_true_states_test = test_data_loaded["X_true_states"]
            # Use initial conditions from the test data file for evaluation
            x0_estimate_filter_test = test_data_loaded.get(
                "x0_estimate_filter", x0_estimate_filter
            )  # Fallback to tuning x0
            P0_initial_val_test = test_data_loaded.get(
                "P0_initial_val", P0_initial_val
            )  # Fallback to tuning P0
            T_out_test = test_data_loaded[
                "T_out"
            ]  # Ensure T_out for test data is loaded

            print("    Test data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Test data file not found at {test_data_path}")
            return
        except KeyError as e:
            print(
                f"Error: Missing key {e} in the loaded test data. Ensure the data file is correctly generated."
            )
            return
        except Exception as e:
            print(f"Error loading test data: {e}")
            return

        # Evaluate the tuned filter
        print(f"    Reconstructing best filter with params: {best_hyperparams}")
        best_tuned_filter = initial_filter_template.new_with_kwargs_like(
            **best_hyperparams
        )

        X_predicted_test, evaluated_metrics_dict = _evaluate_filter_on_test_data(
            best_tuned_filter,
            T_out_test,
            Y_measurements_test,
            X_true_states_test,
            x0_estimate_filter_test,
            P0_initial_val_test,
        )

        # Evaluate standard KalmanFilter on the same test data
        print("\n--- Evaluating standard KalmanFilter for comparison ---")
        standard_kf = KalmanFilter(
            system_model=system_model_obj,
            cov_input=cov_input_loaded,
            cov_measurement=cov_measurement_loaded,
        )

        _, standard_kf_metrics = _evaluate_filter_on_test_data(
            standard_kf,
            T_out_test,
            Y_measurements_test,
            X_true_states_test,
            x0_estimate_filter_test,
            P0_initial_val_test,
        )

        # Evaluate SteadyKalmanFilter on the same test data
        print("\n--- Evaluating SteadyKalmanFilter for comparison ---")
        steady_kf = SteadyKalmanFilter(
            system_model=system_model_obj,
            cov_input=cov_input_loaded,
            cov_measurement=cov_measurement_loaded,
        )

        _, steady_kf_metrics = _evaluate_filter_on_test_data(
            steady_kf,
            T_out_test,
            Y_measurements_test,
            X_true_states_test,
            x0_estimate_filter_test,
            P0_initial_val_test,
        )

        # Update the consolidated results with test evaluation metrics
        consolidated_results.update(
            {
                "test_metrics": evaluated_metrics_dict,
                "standard_kf_metrics": standard_kf_metrics,
                "steady_kf_metrics": steady_kf_metrics,
                "evaluated_on_test": True,
                "test_data_path": os.path.abspath(test_data_path),
            }
        )

        # Save the updated consolidated results
        with open(results_filename, "wb") as f:
            pickle.dump(consolidated_results, f)
        print(f"\nUpdated consolidated results with test metrics: {results_filename}")

        # Save evaluation results to a JSON file
        if evaluated_metrics_dict:
            sim_data_basename = pathlib.Path(args.sim_data_path).stem
            test_data_basename = pathlib.Path(test_data_path).stem
            tuning_mode_suffix_str = (
                "_opt" if optimistic else "_real"
            )  # Suffix for optimistic/realistic tuning
            eval_filename = f"eval_{filter_type_to_tune}_test_{test_data_basename}_tuned_{sim_data_basename}{tuning_mode_suffix_str}.json"
            eval_save_path = os.path.join(EVALUATION_RESULTS_DIR, eval_filename)

            # Prepare data for JSON serialization - convert numpy types if any in best_hyperparams
            serializable_best_hyperparams = {}
            if best_hyperparams:
                for k, v in best_hyperparams.items():
                    if isinstance(v, np.generic):
                        serializable_best_hyperparams[k] = v.item()
                    else:
                        serializable_best_hyperparams[k] = v

            results_to_save = {
                "filter_type": filter_type_to_tune,
                "best_hyperparameters_from_tuning": serializable_best_hyperparams,
                "tuning_metric": metric,
                "best_tuning_score": best_tune_score,
                "tuning_mode_optimistic": optimistic,
                "simulation_data_path": os.path.abspath(args.sim_data_path),
                "test_data_path": (
                    os.path.abspath(test_data_path) if test_data_path else None
                ),
                "evaluation_metrics_on_test_data": evaluated_metrics_dict,
                "standard_kalman_filter_metrics": standard_kf_metrics,
                "steady_kalman_filter_metrics": steady_kf_metrics,
            }

            try:
                with open(eval_save_path, "w") as f_eval_json:
                    json.dump(results_to_save, f_eval_json, indent=4)
                print(f"    Evaluation results saved to: {eval_save_path}")
            except Exception as e:
                print(f"    Error saving evaluation results to JSON: {e}")
        else:
            print("    Skipping metric calculation due to alignment issues.")

    print(f"\n{filter_type_to_tune.upper()} hyperparameter tuning finished.")
    print(f"Mode: {'Optimistic' if optimistic else 'Realistic'}")
    print(f"Metric: {metric.upper()}")
    print(f"Simulation data: {sim_data_filename}")
    print(f"Results saved to: {results_filename}")


if __name__ == "__main__":
    main()
