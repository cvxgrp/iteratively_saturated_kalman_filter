#!/usr/bin/env python3
"""
Plot vehicle results from tuned filter data or iteration sweep results.

This script has two main functionalities:
1. Plot vehicle trajectory: Loads a pickle file created by tune_filter.py, reconstructs both the
   steady-state Kalman filter and the tuned filter, and plots the vehicle trajectory
   showing the true path, measurements, and estimates from both filters.

2. Plot iteration sweep results: Loads a pickle file created by tune_num_iters.py and plots
   the performance metric vs. number of iterations for different step sizes.

Usage:
    python plot_vehicle_results.py --tune_filter_results <pickle_file_path>
    python plot_vehicle_results.py --sweep_iters_results <pickle_file_path>

Example:
    python plot_vehicle_results.py --tune_filter_results results/parameter_search_data/steady_huber_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl
    python plot_vehicle_results.py --sweep_iters_results results/parameter_search_data/hsskf_iter_count_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl
"""

import os
import sys
import pickle
import argparse
import pathlib
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns

# Project-specific imports
from iskf.models.vehicle import vehicle_ss
from iskf.models.cstr import cascaded_cstr_ss
from iskf.metrics import METRIC_REGISTRY

# Import filter classes
from iskf.filters.kalman_filter import KalmanFilter
from iskf.filters.steady_kalman_filter import SteadyKalmanFilter
from iskf.filters.huber_kalman_filter import HuberKalmanFilter
from iskf.filters.circular_huber_kalman_filter import CircularHuberKalmanFilter
from iskf.filters.steady_huber_kalman_filter import SteadyHuberKalmanFilter
from iskf.filters.steady_circular_huber_kalman_filter import (
    SteadyCircularHuberKalmanFilter,
)
from iskf.filters.steady_regularized_kalman_filter import SteadyRegularizedKalmanFilter
from iskf.filters.weighted_likelihood_filter import WeightedLikelihoodFilter
from iskf.filters.steady_one_step_huber_filter import SteadyOneStepHuberFilter
from iskf.filters.steady_two_step_huber_filter import SteadyTwoStepHuberFilter
from iskf.filters.steady_three_term_huber import SteadyThreeTermHuberFilter


def setup_matplotlib_for_latex():
    """Configure matplotlib to use LaTeX formatting."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 18,
            "font.size": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "figure.titlesize": 18,
        }
    )


def reconstruct_model_and_filters(
    data_file_path: str,
    force_simulation: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
    """
    Load tuning results, determine whether to use test or simulation data, and reconstruct the system model.

    Args:
        data_file_path: Path to the pickle file containing the tuning results
        force_simulation: If True, use simulation data even if test data is available

    Returns:
        Tuple of (data, best_params, reconstructed_model, using_test_data)
    """
    print(f"Loading tuning results from {data_file_path}")
    with open(data_file_path, "rb") as f:
        results_data = pickle.load(f)

    # Extract data based on the file format
    using_test_data = False

    if (
        isinstance(results_data, dict)
        and "filter_type" in results_data
        and "sim_data_path" in results_data
    ):
        # New consolidated format
        filter_type = results_data["filter_type"]
        best_params = results_data["best_params"]
        sim_data_path = results_data["sim_data_path"]
        test_data_path = results_data.get("test_data_path")
        best_score = results_data.get("best_score")
        evaluated_on_test = results_data.get("evaluated_on_test", False)

        print(f"Found filter type '{filter_type}'")
        print(f"Best parameters: {best_params}")
        if best_score is not None:
            print(f"Best score: {best_score:.4f}")

        # Check if test data is available and was evaluated, and not forcing simulation
        if (
            test_data_path
            and evaluated_on_test
            and os.path.exists(test_data_path)
            and not force_simulation
        ):
            print(f"Test data is available, will use it for plotting: {test_data_path}")
            data_path = test_data_path
            using_test_data = True
        else:
            if force_simulation and test_data_path and evaluated_on_test:
                print(f"Forcing use of simulation data as requested.")
            print(f"Using simulation data for plotting: {sim_data_path}")
            data_path = sim_data_path

    elif isinstance(results_data, dict) and "grid_search_results" in results_data:
        # Extended results format from individual filter tuner
        filter_type = results_data["filter_type"]
        best_params = results_data["best_params"]
        sim_data_path = results_data["sim_data_path"]
        test_data_path = results_data.get("test_data_path")
        best_score = results_data.get("best_score")
        evaluated_on_test = results_data.get("evaluated_on_test", False)

        print(f"Found filter type '{filter_type}' in extended results format")
        print(f"Best parameters: {best_params}")
        if best_score is not None:
            print(f"Best score: {best_score:.4f}")

        # Check if test data is available and was evaluated, and not forcing simulation
        if (
            test_data_path
            and evaluated_on_test
            and os.path.exists(test_data_path)
            and not force_simulation
        ):
            print(f"Test data is available, will use it for plotting: {test_data_path}")
            data_path = test_data_path
            using_test_data = True
        else:
            if force_simulation and test_data_path and evaluated_on_test:
                print(f"Forcing use of simulation data as requested.")
            print(f"Using simulation data for plotting: {sim_data_path}")
            data_path = sim_data_path

    else:
        # Legacy format - need to extract information from the filename
        all_results = results_data

        # Extract best parameters from grid search results
        best_score = float("inf")
        best_params = None
        for params, score in all_results.items():
            if score < best_score:
                best_score = score
                best_params = dict(params)

        # Extract filter type from the filename
        pickle_filename = os.path.basename(data_file_path)
        filter_type = None
        for filter_name in [
            "huber",
            "circular_huber",
            "steady_huber",
            "steady_circular_huber",
            "steady_regularized",
            "wolf",
            "steady_one_step_huber",
            "steady_two_step_huber",
            "steady_three_term_huber",
        ]:
            if filter_name in pickle_filename:
                filter_type = filter_name
                break

        if filter_type is None:
            raise ValueError(
                f"Could not determine filter type from filename: {pickle_filename}"
            )

        # Extract simulation data filename from the grid search results filename
        parts = pickle_filename.split("_")
        sim_data_filename = None

        # Look for "vehicle" in filename parts to identify simulation data filename
        for i, part in enumerate(parts):
            if part == "vehicle":
                # Reconstruct simulation data filename
                sim_parts = []
                # Add filter type prefix if it exists
                if filter_type:
                    filter_parts = filter_type.split("_")
                    for fp in filter_parts:
                        parts.remove(fp) if fp in parts else None

                # Find continuous sequence starting with "vehicle"
                j = i
                while j < len(parts) and not (
                    parts[j] == "optimistic"
                    or parts[j] == "realistic"
                    or parts[j] in ["rmse", "mne"]
                ):
                    sim_parts.append(parts[j])
                    j += 1

                sim_data_filename = "_".join(sim_parts)
                break

        if not sim_data_filename:
            raise ValueError(
                f"Could not extract simulation data filename from: {pickle_filename}"
            )

        print(f"Extracted simulation data filename: {sim_data_filename}")

        # Construct path to simulation data - legacy format only uses simulation data
        data_path = os.path.join(
            "results", "simulation_data", f"{sim_data_filename}.pkl"
        )

    # Verify the chosen data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        loaded_data = pickle.load(f)

    # Reconstruct system model
    model_type = loaded_data["model_type"]
    model_params = loaded_data["model_params"]

    if model_type == "vehicle_ss":
        system_model = vehicle_ss(**model_params)
    elif model_type == "cstr_ss":
        system_model = cascaded_cstr_ss(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    reconstructed_model = {
        "system_model": system_model,
        "cov_input": loaded_data["cov_input"],
        "cov_measurement": loaded_data["cov_measurement"],
        "filter_type": filter_type,
    }

    return loaded_data, best_params, reconstructed_model, using_test_data


def run_filters_and_plot(
    sim_data: Dict[str, Any],
    best_params: Dict[str, Any],
    model_info: Dict[str, Any],
    output_path: str,
    using_test_data: bool = False,
):
    """
    Run the steady Kalman filter and the tuned filter on the simulation data and generate plots.

    Args:
        sim_data: Simulation or test data dictionary
        best_params: Best parameters from grid search
        model_info: Dictionary containing the reconstructed model and filter information
        output_path: Path to save the output plot
        using_test_data: Whether test data is being used
    """
    # Extract necessary data from sim_data
    T_out = sim_data["T_out"]
    Y_measurements = sim_data["Y_measurements"]
    X_true_states = sim_data["X_true_states"]
    x0_estimate = sim_data["x0_estimate_filter"]
    P0_initial = sim_data["P0_initial_val"]

    # Extract model information
    system_model = model_info["system_model"]
    cov_input = model_info["cov_input"]
    cov_measurement = model_info["cov_measurement"]
    filter_type = model_info["filter_type"]

    # Common filter arguments
    common_filter_args = {
        "system_model": system_model,
        "cov_input": cov_input,
        "cov_measurement": cov_measurement,
    }

    # Create the steady Kalman filter
    steady_kf = SteadyKalmanFilter(**common_filter_args)

    # Create the tuned filter based on filter_type
    tuned_filter = None

    if filter_type == "huber":
        tuned_filter = HuberKalmanFilter(**common_filter_args, **best_params)
    elif filter_type == "circular_huber":
        tuned_filter = CircularHuberKalmanFilter(**common_filter_args, **best_params)
    elif filter_type == "steady_huber":
        tuned_filter = SteadyHuberKalmanFilter(**common_filter_args, **best_params)
    elif filter_type == "steady_circular_huber":
        tuned_filter = SteadyCircularHuberKalmanFilter(
            **common_filter_args, **best_params
        )
    elif filter_type == "steady_regularized":
        tuned_filter = SteadyRegularizedKalmanFilter(
            **common_filter_args, **best_params
        )
    elif filter_type == "wolf":
        tuned_filter = WeightedLikelihoodFilter(**common_filter_args, **best_params)
    elif filter_type == "steady_one_step_huber":
        tuned_filter = SteadyOneStepHuberFilter(**common_filter_args, **best_params)
    elif filter_type == "steady_two_step_huber":
        tuned_filter = SteadyTwoStepHuberFilter(**common_filter_args, **best_params)
    elif filter_type == "steady_three_term_huber":
        tuned_filter = SteadyThreeTermHuberFilter(**common_filter_args, **best_params)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Run the steady Kalman filter
    print("Running steady Kalman filter...")
    steady_kf_estimates = steady_kf.estimate(
        T_out, Y_measurements, x_initial_estimate=x0_estimate, P_initial=P0_initial
    )

    # Run the tuned filter
    print(f"Running tuned {filter_type} filter...")
    tuned_filter_estimates = tuned_filter.estimate(
        T_out, Y_measurements, x_initial_estimate=x0_estimate, P_initial=P0_initial
    )

    # Calculate metrics for position estimates (first two dimensions of the state)
    def calculate_position_rmse(estimates, true_states):
        """Calculate RMSE for position estimates."""
        if estimates.shape[1] == true_states.shape[1] - 1:
            true_states = true_states[:, 1:]  # Align if necessary

        position_diff = estimates[:2, :] - true_states[:2, :]
        return np.sqrt(np.mean(np.sum(position_diff**2, axis=0)))

    # Align true states with estimates if necessary
    true_states_aligned = X_true_states
    if steady_kf_estimates.shape[1] == X_true_states.shape[1] - 1:
        true_states_aligned = X_true_states[:, 1:]

    steady_kf_rmse = calculate_position_rmse(steady_kf_estimates, true_states_aligned)
    tuned_filter_rmse = calculate_position_rmse(
        tuned_filter_estimates, true_states_aligned
    )

    # Get pretty names for the filters
    filter_display_names = {
        "huber": "Huber Kalman Filter",
        "circular_huber": "Circular Huber Kalman Filter",
        "steady_huber": "Steady-State Huber KF",
        "steady_circular_huber": "Steady-State Circular Huber KF",
        "steady_regularized": "Steady-State Regularized KF",
        "wolf": "Weighted Likelihood Filter",
        "steady_one_step_huber": "Steady One-Step Huber KF",
        "steady_two_step_huber": r"ISKF ($\tilde k=2$)",
        "steady_three_term_huber": "Steady Three-Term Huber KF",
    }

    tuned_filter_name = filter_display_names.get(
        filter_type, filter_type.replace("_", " ").title()
    )

    # Create the trajectory plot
    plt.figure(figsize=(8, 6))

    # Plot measurements with low alpha for clarity
    plt.scatter(
        Y_measurements[0, :],
        Y_measurements[1, :],
        color="gray",
        alpha=0.4,
        marker="x",
        s=40,
        label="Measurements",
    )

    # Plot true trajectory
    plt.plot(
        X_true_states[0, :],
        X_true_states[1, :],
        "k-",
        linewidth=2,
        label="True Position",
    )

    # Plot Steady Kalman filter estimates
    plt.plot(
        steady_kf_estimates[0, :],
        steady_kf_estimates[1, :],
        "b:",
        linewidth=2,
        markersize=2,
        # label=f"SSKF (RMSE: {steady_kf_rmse:.2f})",
        label=f"KF",
    )

    # Plot tuned filter estimates
    plt.plot(
        tuned_filter_estimates[0, :],
        tuned_filter_estimates[1, :],
        "r--",
        linewidth=2,
        # label=f"{tuned_filter_name} (RMSE: {tuned_filter_rmse:.2f})",
        label=f"{tuned_filter_name}",
    )

    # Set plot limits with some padding
    x_min, x_max = np.min(X_true_states[0, :]), np.max(X_true_states[0, :])
    y_min, y_max = np.min(X_true_states[1, :]), np.max(X_true_states[1, :])
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

    # Add labels and title
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create custom legend with better formatting
    handles, labels = plt.gca().get_legend_handles_labels()

    # Add text with filter parameters
    # param_text = r"$\textbf{Tuned Filter Parameters}$" + "\n"
    # for param_name, param_value in best_params.items():
    #     param_latex = param_name
    #     if param_name == "coef_s":
    #         param_latex = r"$\lambda_s$"
    #     elif param_name == "coef_o":
    #         param_latex = r"$\lambda_o$"
    #     elif param_name == "coef":
    #         param_latex = r"$\lambda$"
    #     elif param_name == "step_size":
    #         param_latex = r"$\alpha$"

    #     # Format the value, handling special cases like infinity
    #     if param_value == np.inf:
    #         param_value_str = r"$\infty$"
    #     else:
    #         param_value_str = f"{param_value:.4f}"

    #     param_text += f"{param_latex}: {param_value_str}\n"

    # # Add text box with parameters
    # plt.figtext(0.02, 0.02, param_text, bbox=dict(facecolor="white", alpha=0.8))

    # Add data source indicator
    # data_source = "Test Data" if using_test_data else "Simulation Data"
    # plt.figtext(
    #     0.98, 0.02, data_source, ha="right", bbox=dict(facecolor="white", alpha=0.8)
    # )

    plt.legend(handles, labels, loc="best", framealpha=0.9)
    plt.tight_layout()

    # Save the plot
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # Also display the plot if running interactively
    plt.show()

    # --- Add logic for SteadyTwoStepHuberFilter estimates and state trajectory plot ---
    s2sh_estimates = None
    if filter_type == "steady_two_step_huber":
        s2sh_estimates = tuned_filter_estimates
        print(
            "Using estimates from the tuned SteadyTwoStepHuberFilter for state plots."
        )
    else:
        print(
            "Tuned filter is not SteadyTwoStepHuberFilter. "
            "Running a new SteadyTwoStepHuberFilter with default params for state plots."
        )
        default_s2sh_params = {
            "coef_s": 1.345,  # Huber parameter for state
            "coef_o": 1.345,  # Huber parameter for observation
            "step_size": 1.0,  # Step size for the internal iterations
        }
        s2sh_filter = SteadyTwoStepHuberFilter(
            **common_filter_args, **default_s2sh_params
        )
        print(
            f"Running new SteadyTwoStepHuberFilter with params: {default_s2sh_params}"
        )
        s2sh_estimates = s2sh_filter.estimate(
            T_out, Y_measurements, x_initial_estimate=x0_estimate, P_initial=P0_initial
        )

    # Determine output path for the state trajectories plot
    output_path_dir = os.path.dirname(output_path)
    output_path_basename = os.path.basename(output_path)
    output_path_name, output_path_ext = os.path.splitext(output_path_basename)

    # Construct a clean name for the states plot
    # Remove common suffixes like '_trajectory_plot' or '_plot' before adding '_states_plot'
    if output_path_name.endswith("_trajectory_plot"):
        base_name_for_states_plot = output_path_name[: -len("_trajectory_plot")]
    elif output_path_name.endswith("_plot"):
        base_name_for_states_plot = output_path_name[: -len("_plot")]
    else:
        base_name_for_states_plot = output_path_name

    output_filename_states_plot = os.path.join(
        output_path_dir, f"{base_name_for_states_plot}_states_plot{output_path_ext}"
    )

    # Ensure the time vector for plotting matches the estimate length
    # T_out from sim_data is usually np.arange(num_steps)
    # estimates have num_steps columns, true_states_aligned also has num_steps columns
    time_vector_for_plot = T_out
    if len(T_out) != steady_kf_estimates.shape[1]:
        print(
            f"Warning: T_out length ({len(T_out)}) differs from estimate length ({steady_kf_estimates.shape[1]}). Adjusting time vector."
        )
        time_vector_for_plot = np.arange(steady_kf_estimates.shape[1])

    # Plot state trajectories
    plot_state_trajectories(
        time_vector_for_plot,  # Use T_out from sim_data, should match estimate columns
        true_states_aligned,  # Already aligned
        steady_kf_estimates,
        s2sh_estimates,
        output_filename_states_plot,
    )


def plot_state_trajectories(
    time_vector: np.ndarray,
    true_states: np.ndarray,
    kf_estimates: np.ndarray,
    s2sh_estimates: np.ndarray,
    output_filepath: str,
):
    """
    Plot individual state trajectories (positions and velocities).

    Args:
        time_vector: Array of time steps.
        true_states: True state trajectories (num_states, num_steps).
        kf_estimates: Kalman Filter state estimates (num_states, num_steps).
        s2sh_estimates: Steady Two-Step Huber Filter state estimates (num_states, num_steps).
        output_filepath: Path to save the output plot.
    """
    if true_states.shape[0] < 4:
        print(
            f"Warning: True states have {true_states.shape[0]} dimensions, expected at least 4 for vehicle plot. Plotting available states."
        )
        num_states_to_plot = true_states.shape[0]
    else:
        num_states_to_plot = 4

    state_labels_pretty = [r"$\xi_1$", r"$\xi_2$", r"$\nu_1$", r"$\nu_2$"]
    # plot_titles = [
    #     "Position $\\xi_1$ Trajectory",
    #     "Position $\\xi_2$ Trajectory",
    #     "Velocity $\\nu_1$ Trajectory",
    #     "Velocity $\\nu_2$ Trajectory",
    # ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 6))  # Adjusted figsize for better layout
    axs = axs.ravel()  # Flatten axes array for easy iteration

    # Store handles and labels for the main figure legend
    handles_list = []
    labels_list = []

    for i in range(num_states_to_plot):
        ax = axs[i]
        # ax.plot(time_vector, true_states[i, :], "k-", linewidth=2, label="True")
        (line1,) = ax.plot(
            time_vector,
            np.abs(kf_estimates[i, :] - true_states[i, :]),
            "b:",
            linewidth=2,
            label="KF",
        )
        (line2,) = ax.plot(
            time_vector,
            np.abs(s2sh_estimates[i, :] - true_states[i, :]),
            "g--",  # Green dashed line for S2SHF
            linewidth=2,
            label=r"ISKF ($\tilde k=2$)",  # Label from filter_display_names
        )

        # Store handles and labels from the first subplot for the main legend
        if i == 0:
            handles_list.extend([line1, line2])
            # Get labels from lines to avoid duplicates if ax.legend() was used before
            current_labels = [h.get_label() for h in handles_list]
            labels_list = current_labels

        # ax.set_title(plot_titles[i])
        ax.set_ylabel(state_labels_pretty[i] + " error")
        ax.set_xlabel(r"$t$")
        # ax.legend(loc="best") # Removed individual legends
        ax.grid(True, linestyle="--", alpha=0.7)

    # If fewer than 4 states, hide unused subplots
    for j in range(num_states_to_plot, 4):
        fig.delaxes(axs[j])

    # Add a single legend to the figure
    # Adjust ncol and bbox_to_anchor for desired placement (e.g., above subplots)
    fig.legend(
        handles_list,
        labels_list,
        loc="upper center",
        ncol=len(labels_list),
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(
        pad=2.0, rect=[0, 0, 1, 0.95]
    )  # Adjust rect to make space for fig.legend
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    print(f"State trajectories plot saved to {output_filepath}")
    plt.show()


def plot_iteration_sweep_results(
    data_file_path: str,
    output_path: str = None,
):
    """
    Plot the results of an iteration sweep from tune_num_iters.py.

    Args:
        data_file_path: Path to the pickle file containing iteration sweep results
        output_path: Path to save the output plot (if None, auto-generated)
    """
    print(f"Loading iteration sweep results from {data_file_path}")

    # Load the data
    with open(data_file_path, "rb") as f:
        results_data = pickle.load(f)

    # Extract relevant data
    model_info = results_data.get("model_info", {})
    metric = model_info.get("metric", "rmse")
    optimistic = model_info.get("optimistic", False)
    iter_count_sweep = model_info.get("iter_count_sweep", [])
    exact_hsskf_data = results_data.get("exact_hsskf", {})
    baseline_score = exact_hsskf_data.get("score")

    # Get the SSKF and KF scores
    sskf_data = results_data.get("sskf", {})
    sskf_score = sskf_data.get("score")
    kf_data = results_data.get("kf", {})
    kf_score = kf_data.get("score")

    # Debug information about the scores
    if sskf_score is None:
        print(
            "Warning: SSKF score is None. Keys in results_data:",
            list(results_data.keys()),
        )
        print("SSKF data:", sskf_data)

    if kf_score is None:
        print(
            "Warning: KF score is None. Keys in results_data:",
            list(results_data.keys()),
        )
        print("KF data:", kf_data)

    all_best_scores = results_data.get("all_best_scores", {})
    best_global_params = results_data.get("best_global_params", {})

    # Set up the plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    # Create a colormap for different step sizes
    colormap = plt.get_cmap("viridis")

    # Sort step sizes for consistent plotting
    step_sizes = sorted([float(k) for k in all_best_scores.keys()])
    num_step_sizes = len(step_sizes)

    # Create a list to store best points for annotations
    best_points = []
    best_overall_point = None

    # Plot each step size as a separate line
    for i, step_size in enumerate(step_sizes):
        step_size_str = str(step_size)
        if step_size_str not in all_best_scores:
            step_size_str = float(step_size)  # Try as float key

        step_data = all_best_scores.get(step_size_str, {})
        if not step_data:
            continue

        # Convert step_data to a sorted list of (iter_count, score) pairs
        iter_scores = [(int(k), v) for k, v in step_data.items()]
        iter_scores.sort()  # Sort by iteration count

        if not iter_scores:
            continue

        iter_counts, scores = zip(*iter_scores)

        # Find best score for this step size
        best_idx = np.argmin(scores)
        best_iter = iter_counts[best_idx]
        best_score = scores[best_idx]
        best_points.append((best_iter, best_score, step_size))

        # Check if this is the global best
        if (
            best_global_params
            and step_size == best_global_params.get("step_size")
            and best_iter == best_global_params.get("iter_count")
        ):
            best_overall_point = (best_iter, best_score)

        # Plot the line
        (line,) = plt.plot(
            iter_counts,
            scores,
            "o-",
            # color=colormap(i / max(1, num_step_sizes - 1)),
            linewidth=2,
            markersize=6,
            # label=f"Step size = {step_size:.4f}",
        )

    # # Add horizontal line for the SSKF score
    # if sskf_score is not None:
    #     plt.axhline(
    #         y=sskf_score,
    #         color="blue",
    #         linestyle="--",
    #         linewidth=2,
    #         label=f"SSKF {metric.upper()} ({sskf_score:.4f})",
    #     )

    # # Add horizontal line for the KF score
    # if kf_score is not None:
    #     plt.axhline(
    #         y=kf_score,
    #         color="green",
    #         linestyle="--",
    #         linewidth=2,
    #         label=f"KF {metric.upper()} ({kf_score:.4f})",
    #     )

    # # Add horizontal line for the baseline score (Exact HSSKF)
    # if baseline_score is not None:
    #     plt.axhline(
    #         y=baseline_score,
    #         color="red",
    #         linestyle="--",
    #         linewidth=2,
    #         label=f"Exact HSSKF {metric.upper()} ({baseline_score:.4f})",
    #     )

    # Set labels and title
    plt.xlabel(r"$\tilde k$", fontsize=20)
    plt.ylabel(f"{metric.upper()}", fontsize=20)

    # # Add legend
    # plt.legend(loc="best", fontsize=9)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Use logarithmic scale for x-axis if appropriate
    if len(iter_count_sweep) > 0 and all(i > 0 for i in iter_count_sweep):
        # plt.xscale("log")
        plt.xticks(sorted(iter_count_sweep), [str(x) for x in sorted(iter_count_sweep)])

    plt.tight_layout()

    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Iteration sweep plot saved to {output_path}")
    else:
        # Auto-generate output filename
        input_basename = os.path.basename(data_file_path)
        input_name = os.path.splitext(input_basename)[0]
        output_filename = f"{input_name}_iter_sweep_plot.pdf"
        output_path = os.path.join("figures", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Iteration sweep plot saved to {output_path}")

    # Also display the plot if running interactively
    plt.show()

    return output_path


def main():
    """Main function to handle command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Plot vehicle trajectory with filter estimates or iteration sweep results"
    )

    # Create mutually exclusive group for the two input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tune_filter_results",
        type=str,
        help="Path to the pickle file containing tuning results from tune_filter.py",
    )
    input_group.add_argument(
        "--sweep_iters_results",
        type=str,
        help="Path to the pickle file containing iteration sweep results from tune_num_iters.py",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output plot (default: auto-generated in figures/ folder)",
    )
    parser.add_argument(
        "--force-simulation",
        action="store_true",
        help="Force using simulation data even if test data is available (only for --tune_filter_results)",
    )

    args = parser.parse_args()

    # Determine which mode to run in
    if args.tune_filter_results:
        data_file_path = args.tune_filter_results
        plot_mode = "vehicle_trajectory"
    else:
        data_file_path = args.sweep_iters_results
        plot_mode = "iteration_sweep"

    # Check if the input file exists
    if not os.path.exists(data_file_path):
        print(f"Error: The specified file '{data_file_path}' does not exist.")
        sys.exit(1)

    # Configure matplotlib for LaTeX
    setup_matplotlib_for_latex()

    # Process based on the selected mode
    if plot_mode == "vehicle_trajectory":
        # Load data and reconstruct model and filters
        try:
            data, best_params, model_info, using_test_data = (
                reconstruct_model_and_filters(data_file_path, args.force_simulation)
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output filename based on input filename
            input_basename = os.path.basename(data_file_path)
            input_name = os.path.splitext(input_basename)[0]
            data_type = "test" if using_test_data else "sim"
            output_filename = f"{input_name}_{data_type}_trajectory_plot.pdf"
            output_path = os.path.join("figures", output_filename)

        # Create the figures directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run filters and create plot
        run_filters_and_plot(
            data, best_params, model_info, output_path, using_test_data
        )

    else:  # plot_mode == "iteration_sweep"
        # Plot iteration sweep results
        plot_iteration_sweep_results(data_file_path, args.output)


if __name__ == "__main__":
    main()
