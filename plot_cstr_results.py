#!/usr/bin/env python3
"""
Plot CSTR results from tuned filter data.

This script loads a pickle file created by tune_filter.py for a CSTR model,
reconstructs the Steady-State Kalman filter and the tuned filter, runs them
on the original simulation data, and plots the CSTR state trajectories
(concentrations and temperatures) showing true values, measurements (where applicable),
and estimates from both filters.

Usage:
    python plot_cstr_results.py --tune_filter_results <pickle_file_path>

Example:
    python plot_cstr_results.py --tune_filter_results results/parameter_search_data/steady_iskf_cstr_n4_p10_sp10_sm10_steps1000_seed0_optimistic_rmse_results.pkl
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
import seaborn as sns

# Project-specific imports
from iskf.models.cstr import cascaded_cstr_ss
from iskf.metrics import METRIC_REGISTRY

# Import filter classes
from iskf.filters.kalman_filter import (
    KalmanFilter,
)  # For completeness, though SKF is primary
from iskf.filters.steady_kalman_filter import SteadyKalmanFilter
from iskf.filters.huber_kalman_filter import HuberKalmanFilter
from iskf.filters.iskf import IterSatKalmanFilter
from iskf.filters.steady_huber_kalman_filter import SteadyHuberKalmanFilter
from iskf.filters.steady_iskf import (
    SteadyIterSatKalmanFilter,
)
from iskf.filters.steady_regularized_kalman_filter import SteadyRegularizedKalmanFilter
from iskf.filters.weighted_likelihood_filter import WeightedLikelihoodFilter
from iskf.filters.steady_one_step_iskf import SteadyOneStepIterSatFilter
from iskf.filters.steady_two_step_iskf import SteadyTwoStepIterSatFilter
from iskf.filters.steady_three_term_iskf import SteadyThreeStepIterSatFilter


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


def reconstruct_cstr_model_and_filters(
    tune_results_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, int]:
    """
    Load tuning results, associated simulation data, and reconstruct the CSTR system model.

    Args:
        tune_results_path: Path to the pickle file containing the tuning results.

    Returns:
        Tuple of (loaded_sim_data, best_tuned_filter_params, reconstructed_model_info, filter_type_name, num_reactors)
    """
    print(f"Loading tuning results from {tune_results_path}")
    with open(tune_results_path, "rb") as f:
        tuning_results_data = pickle.load(f)

    filter_type_name = tuning_results_data["filter_type"]
    best_tuned_filter_params = tuning_results_data["best_params"]
    sim_data_path = tuning_results_data["sim_data_path"]

    print(f"  Tuned Filter Type: {filter_type_name}")
    print(f"  Best Tuned Parameters: {best_tuned_filter_params}")
    print(f"  Original simulation data path: {sim_data_path}")

    if not os.path.exists(sim_data_path):
        raise FileNotFoundError(
            f"Original simulation data file not found: {sim_data_path}"
        )

    print(f"Loading original simulation data from {sim_data_path}...")
    with open(sim_data_path, "rb") as f:
        loaded_sim_data = pickle.load(f)

    model_type_sim = loaded_sim_data["model_type"]
    model_params_sim = loaded_sim_data["model_params"]
    num_reactors = loaded_sim_data.get(
        "n_reactors_sim", model_params_sim.get("n")
    )  # Get n_reactors

    if model_type_sim != "cascaded_cstr_ss":
        raise ValueError(
            f"Expected model_type 'cascaded_cstr_ss', but got '{model_type_sim}' from {sim_data_path}"
        )

    if num_reactors is None:
        raise ValueError(
            f"Could not determine number of reactors from simulation data at {sim_data_path}"
        )

    system_model_obj = cascaded_cstr_ss(**model_params_sim)

    reconstructed_model_info = {
        "system_model": system_model_obj,
        "cov_input": loaded_sim_data["cov_input"],
        "cov_measurement": loaded_sim_data["cov_measurement"],
    }

    print(f"  Successfully reconstructed CSTR model with {num_reactors} reactor(s).")

    return (
        loaded_sim_data,
        best_tuned_filter_params,
        reconstructed_model_info,
        filter_type_name,
        num_reactors,
    )


def plot_cstr_comparison_data(
    T_out: np.ndarray,
    X_states_abs_true: np.ndarray,
    Y_measurements_abs: np.ndarray,
    X_hat_skf_abs: np.ndarray,
    X_hat_tuned_abs: np.ndarray,
    n_reactors: int,
    output_path: str,
    skf_rmse: float,
    tuned_filter_rmse: float,
    tuned_filter_name_pretty: str,
):
    """
    Plots CSTR state trajectories (Concentrations and Temperatures) comparing true values,
    SKF estimates, and tuned filter estimates.

    Args:
        T_out: Time vector.
        X_states_abs_true: Absolute true state trajectories [C_A1, T1, C_A2, T2, ...].
        Y_measurements_abs: Absolute noisy measurements [T1_m, C_A1_m, T2_m, C_A2_m, ...].
        X_hat_skf_abs: Absolute SKF estimated state trajectories.
        X_hat_tuned_abs: Absolute tuned filter estimated state trajectories.
        n_reactors: Number of reactors in the cascade.
        output_path: Path to save the plot.
        skf_rmse: RMSE for the Steady Kalman Filter.
        tuned_filter_rmse: RMSE for the tuned filter.
        tuned_filter_name_pretty: Display name for the tuned filter.
    """
    num_state_plots_rows = n_reactors
    num_state_plots_cols = 2  # C_A and T for each reactor

    fig_height = max(3 * num_state_plots_rows, 7)  # Ensure reasonable height
    fig, axs = plt.subplots(
        num_state_plots_rows,
        num_state_plots_cols,
        figsize=(14, fig_height),
    )
    # If n_reactors is 1, axs is 1D, make it 2D for consistent indexing
    if n_reactors == 1:
        axs = np.expand_dims(axs, axis=0)

    # Align time vectors if estimates are shorter
    T_skf = T_out
    T_tuned = T_out

    X_true_plot = X_states_abs_true
    Y_meas_plot = Y_measurements_abs  # Measurements align with T_true

    if X_hat_skf_abs.shape[1] == T_out.shape[0] - 1:
        T_skf = T_out[1:]
    elif X_hat_skf_abs.shape[1] != T_out.shape[0]:
        print(
            f"Warning: SKF estimates length ({X_hat_skf_abs.shape[1]}) mismatch with T_out ({T_out.shape[0]}). Skipping SKF plot for states."
        )
        X_hat_skf_abs = None  # Prevent plotting

    if X_hat_tuned_abs.shape[1] == T_out.shape[0] - 1:
        T_tuned = T_out[1:]
    elif X_hat_tuned_abs.shape[1] != T_out.shape[0]:
        print(
            f"Warning: Tuned estimates length ({X_hat_tuned_abs.shape[1]}) mismatch with T_out ({T_out.shape[0]}). Skipping tuned plot for states."
        )
        X_hat_tuned_abs = None  # Prevent plotting

    handles_list = []
    labels_list = []

    for i in range(n_reactors):
        ax_conc = axs[i, 0]
        ax_temp = axs[i, 1]

        # Plot Concentration C_A(i+1)
        (true_conc_line,) = ax_conc.plot(
            T_out,  # Plot against full T_out
            X_states_abs_true[2 * i, :],  # Use X_states_abs_true directly
            "k-",
            linewidth=1.5,
            label=f"True $c_{{{i+1}}}$",
        )

        skf_conc_line, tuned_conc_line = None, None
        if X_hat_skf_abs is not None:
            (skf_conc_line,) = ax_conc.plot(
                T_skf,
                X_hat_skf_abs[2 * i, :],
                "b--",
                linewidth=1.2,
                label=f"KF Est.",  # Generic label for legend
            )
        if X_hat_tuned_abs is not None:
            (tuned_conc_line,) = ax_conc.plot(
                T_tuned,
                X_hat_tuned_abs[2 * i, :],
                "r-.",
                linewidth=1.2,
                label=f"{tuned_filter_name_pretty} Est.",  # Generic label for legend
            )

        ax_conc.set_ylabel(f"$c_{{{i+1}}}$ (kmol/m$^3$)")  # Corrected unit
        if i == n_reactors - 1:  # X-axis label only for bottom plots
            ax_conc.set_xlabel(r"$t$")
        ax_conc.grid(True, linestyle=":", alpha=0.7)
        # ax_conc.legend(loc="upper right") # Removed individual legend

        # Plot Temperature T(i+1)
        (true_temp_line,) = ax_temp.plot(
            T_out,  # Plot against full T_out
            X_states_abs_true[2 * i + 1, :],  # Use X_states_abs_true directly
            "k-",
            linewidth=1.5,
            label=f"True $T_{i+1}$",
        )

        # Plot measurements (temperature)
        (meas_temp_line,) = ax_temp.plot(
            T_out,  # Plot against full T_out
            Y_meas_plot[
                i, :
            ],  # Corrected index for temperature measurement of reactor i+1
            "x",
            color="gray",
            alpha=0.5,
            markersize=3,
            label=f"Meas. $T_{i+1}$",
        )

        skf_temp_line, tuned_temp_line = None, None
        if X_hat_skf_abs is not None:
            (skf_temp_line,) = ax_temp.plot(
                T_skf,
                X_hat_skf_abs[2 * i + 1, :],
                "b--",
                linewidth=1.2,
                label=f"KF Est.",  # Using RMSE in label made legend too busy for shared
            )
        if X_hat_tuned_abs is not None:
            (tuned_temp_line,) = ax_temp.plot(
                T_tuned,
                X_hat_tuned_abs[2 * i + 1, :],
                "r-.",
                linewidth=1.2,
                label=f"{tuned_filter_name_pretty} Est.",  # Using RMSE in label made legend too busy for shared
            )

        ax_temp.set_ylabel(f"$T_{i+1}$ (K)")
        if i == n_reactors - 1:  # X-axis label only for bottom plots
            ax_temp.set_xlabel("Time (s)")
        ax_temp.grid(True, linestyle=":", alpha=0.7)
        # ax_temp.legend(loc="upper right") # Removed individual legend

        # Collect handles and labels for the main figure legend (only from the first reactor plot)
        if i == 0:
            # Order: True, Measurement (temp only), SKF, Tuned
            # For concentrations, we have True, SKF, Tuned
            handles_list.append(true_conc_line)
            labels_list.append(f"True State")

            handles_list.append(
                meas_temp_line
            )  # Measurement only exists for temperature plots
            labels_list.append(f"Measurement")

            if skf_conc_line:  # Check if skf lines were plotted
                handles_list.append(skf_conc_line)
                labels_list.append(f"KF")
            elif skf_temp_line:  # Fallback if only temp was plotted for SKF
                handles_list.append(skf_temp_line)
                labels_list.append(f"KF")

            if tuned_conc_line:  # Check if tuned lines were plotted
                handles_list.append(tuned_conc_line)
                labels_list.append(r"ISKF ($\tilde k=2$)")
            elif tuned_temp_line:  # Fallback if only temp was plotted for tuned
                handles_list.append(tuned_temp_line)
                labels_list.append(r"ISKF ($\tilde k=2$)")

    # Add a single legend to the figure
    # Adjust ncol and bbox_to_anchor for desired placement (e.g., above subplots)
    fig.legend(
        handles_list,
        labels_list,
        loc="upper center",
        ncol=len(handles_list),  # Adjust to fit all items, e.g., 4 or 2
        bbox_to_anchor=(
            0.5,
            1.00 + (0.01 * (4 / num_state_plots_rows)),
        ),  # Dynamic adjustment for legend position
    )

    # plt.suptitle(
    #     f"CSTR State Trajectories Comparison: SKF vs. Tuned {tuned_filter_name_pretty}",
    #     fontsize=16,
    # ) # Removed suptitle as per user request for consistency with error plot
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95 + (0.01 * (4 / num_state_plots_rows))]
    )  # Adjust rect to make space for fig.legend

    print(f"Saving CSTR comparison plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


def plot_cstr_state_errors(
    time_vector_est: np.ndarray,
    X_states_abs_true_aligned: np.ndarray,
    X_hat_skf_abs: np.ndarray,
    X_hat_tuned_abs: np.ndarray,
    n_reactors: int,
    output_path: str,
    tuned_filter_name_pretty: str,
):
    """
    Plots absolute state estimate errors for CSTR (Concentrations and Temperatures).

    Args:
        time_vector_est: Time vector corresponding to the estimates.
        X_states_abs_true_aligned: Absolute true state trajectories, aligned with estimates.
        X_hat_skf_abs: Absolute SKF estimated state trajectories.
        X_hat_tuned_abs: Absolute tuned filter estimated state trajectories.
        n_reactors: Number of reactors in the cascade.
        output_path: Path to save the plot.
        tuned_filter_name_pretty: Display name for the tuned filter.
    """
    fig, axs = plt.subplots(n_reactors, 2, figsize=(14, 3 * n_reactors))
    # If n_reactors is 1, axs is 1D, make it 2D for consistent indexing
    if n_reactors == 1:
        axs = np.expand_dims(axs, axis=0)

    handles_list = []
    labels_list = []

    for i in range(n_reactors):
        # Concentration Error C_A(i+1)
        ax_conc_err = axs[i, 0]
        err_skf_conc = np.abs(
            X_hat_skf_abs[2 * i, :] - X_states_abs_true_aligned[2 * i, :]
        )
        err_tuned_conc = np.abs(
            X_hat_tuned_abs[2 * i, :] - X_states_abs_true_aligned[2 * i, :]
        )

        (line_skf_c,) = ax_conc_err.plot(
            time_vector_est,
            err_skf_conc,
            "b:",
            linewidth=2,
            label="KF",
        )
        (line_tuned_c,) = ax_conc_err.plot(
            time_vector_est,
            err_tuned_conc,
            "g--",  # Green dashed line for S2SHF
            linewidth=2,
            label=r"ISKF ($\tilde k=2$)",
        )

        if i == 0:
            handles_list.extend([line_skf_c, line_tuned_c])
            labels_list.extend(["KF", r"ISKF ($\tilde k=2$)"])

        conc_label = f"$c_{{{i+1}}}$"
        ax_conc_err.set_ylabel(f"{conc_label} error")  #  (kmol/m$^3$)
        ax_conc_err.grid(True, linestyle=":", alpha=0.7)
        ax_conc_err.set_xlabel(r"$t$")

        # Temperature Error T(i+1)
        ax_temp_err = axs[i, 1]
        err_skf_temp = np.abs(
            X_hat_skf_abs[2 * i + 1, :] - X_states_abs_true_aligned[2 * i + 1, :],
        )
        err_tuned_temp = np.abs(
            X_hat_tuned_abs[2 * i + 1, :] - X_states_abs_true_aligned[2 * i + 1, :]
        )

        ax_temp_err.plot(
            time_vector_est,
            err_skf_temp,
            "b:",
            linewidth=2,
            label="KF",
        )
        ax_temp_err.plot(
            time_vector_est,
            err_tuned_temp,
            "g--",  # Green dashed line for S2SHF
            linewidth=2,
            label=r"ISKF ($\tilde k=2$)",  # Label from filter_display_names
        )

        temp_label = f"$T_{{{i+1}}}$"
        ax_temp_err.set_ylabel(f"{temp_label} error")  #  (K)
        ax_temp_err.grid(True, linestyle=":", alpha=0.7)
        ax_temp_err.set_xlabel(r"$t$")

    fig.legend(
        handles_list,
        labels_list,
        loc="upper center",
        ncol=len(labels_list),
        bbox_to_anchor=(0.5, 0.99),
    )
    plt.tight_layout(
        pad=2.0, rect=[0, 0, 1, 0.95]
    )  # Adjust rect to make space for fig.legend

    print(f"Saving CSTR state error plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


def run_cstr_filters_and_plot(
    sim_data_loaded: Dict[str, Any],
    best_tuned_filter_params: Dict[str, Any],
    model_info: Dict[str, Any],
    tuned_filter_type_name: str,
    num_reactors: int,
    output_plot_path: str,
):
    """
    Run the Steady-State Kalman Filter and the tuned CSTR filter, then plot results.
    Handles linearization and conversion to absolute values for plotting.
    """
    # Extract data from loaded simulation data (these are linearized)
    T_out = sim_data_loaded["T_out"]
    Y_measurements_lin = sim_data_loaded["Y_measurements"]
    X_true_states_lin = sim_data_loaded["X_true_states"]
    x0_estimate_lin = sim_data_loaded["x0_estimate_filter"]
    P0_initial = sim_data_loaded["P0_initial_val"]

    # Load offset vectors
    state_offset_vector = sim_data_loaded.get("state_offset_vector")
    measurement_offset_vector = sim_data_loaded.get("measurement_offset_vector")

    if state_offset_vector is None or measurement_offset_vector is None:
        print(
            "Warning: Offset vectors not found in simulation data. Assuming zero offsets."
        )
        nx = model_info["system_model"].nstates
        ny = model_info["system_model"].noutputs
        state_offset_vector = np.zeros(nx)
        measurement_offset_vector = np.zeros(ny)

    # Extract model components
    system_model = model_info["system_model"]
    cov_input = model_info["cov_input"]
    cov_measurement = model_info["cov_measurement"]

    common_filter_args = {
        "system_model": system_model,
        "cov_input": cov_input,
        "cov_measurement": cov_measurement,
    }

    # 1. Run Steady-State Kalman Filter (operates on linearized data)
    print("Running Steady-State Kalman Filter (SKF)...")
    skf = SteadyKalmanFilter(**common_filter_args)
    X_hat_skf_lin = skf.estimate(
        T_out,
        Y_measurements_lin,
        x_initial_estimate=x0_estimate_lin,
        P_initial=P0_initial,
    )

    # 2. Instantiate and run the tuned filter (operates on linearized data)
    filter_class_map = {
        "huber": HuberKalmanFilter,
        "iskf": IterSatKalmanFilter,
        "steady_huber": SteadyHuberKalmanFilter,
        "steady_iskf": SteadyIterSatKalmanFilter,
        "steady_regularized": SteadyRegularizedKalmanFilter,
        "wolf": WeightedLikelihoodFilter,
        "steady_one_step_huber": SteadyOneStepIterSatFilter,
        "steady_two_step_huber": SteadyTwoStepIterSatFilter,
        "steady_three_term_huber": SteadyThreeStepIterSatFilter,
    }
    TunedFilterClass = filter_class_map.get(tuned_filter_type_name)
    if TunedFilterClass is None:
        raise ValueError(f"Unknown tuned filter type: {tuned_filter_type_name}")

    print(
        f"Running tuned {tuned_filter_type_name} filter with params: {best_tuned_filter_params}..."
    )
    tuned_filter_instance = TunedFilterClass(
        **common_filter_args, **best_tuned_filter_params
    )
    X_hat_tuned_lin = tuned_filter_instance.estimate(
        T_out,
        Y_measurements_lin,
        x_initial_estimate=x0_estimate_lin,
        P_initial=P0_initial,
    )

    # 3. Convert all relevant data to ABSOLUTE values for plotting and metrics
    X_states_abs_true = X_true_states_lin + state_offset_vector.reshape(-1, 1)
    Y_measurements_abs = Y_measurements_lin + measurement_offset_vector.reshape(-1, 1)
    X_hat_skf_abs = X_hat_skf_lin + state_offset_vector.reshape(-1, 1)
    X_hat_tuned_abs = X_hat_tuned_lin + state_offset_vector.reshape(-1, 1)

    # 4. Calculate RMSE metrics (on absolute values, matching simulate_cstr.py if it were to run these filters)
    rmse_func = METRIC_REGISTRY["rmse"]

    # Align true states for metric calculation if estimates are shorter
    X_true_for_metric_skf = X_states_abs_true
    if X_hat_skf_abs.shape[1] < X_states_abs_true.shape[1]:
        X_true_for_metric_skf = X_states_abs_true[:, -X_hat_skf_abs.shape[1] :]

    X_true_for_metric_tuned = X_states_abs_true
    if X_hat_tuned_abs.shape[1] < X_states_abs_true.shape[1]:
        X_true_for_metric_tuned = X_states_abs_true[:, -X_hat_tuned_abs.shape[1] :]

    skf_rmse = rmse_func(X_hat_skf_abs, X_true_for_metric_skf)
    tuned_filter_rmse = rmse_func(X_hat_tuned_abs, X_true_for_metric_tuned)

    print(f"  SKF RMSE: {skf_rmse:.4f}")
    print(f"  Tuned {tuned_filter_type_name} RMSE: {tuned_filter_rmse:.4f}")

    filter_display_names = {
        "huber": "Huber KF",
        "iskf": "ISKF",
        "steady_huber": "Steady Huber KF",
        "steady_iskf": "Steady ISKF",
        "steady_regularized": "Steady Reg. KF",
        "wolf": "WoLF",
        "steady_one_step_huber": "Steady 1-Step Huber",
        "steady_two_step_huber": "Steady 2-Step Huber",
        "steady_three_term_huber": "Steady 3-Term Huber",
    }
    tuned_filter_name_pretty = filter_display_names.get(
        tuned_filter_type_name, tuned_filter_type_name.replace("_", " ").title()
    )

    # 5. Plot
    plot_cstr_comparison_data(
        T_out,
        X_states_abs_true,
        Y_measurements_abs,
        X_hat_skf_abs,
        X_hat_tuned_abs,
        num_reactors,
        output_plot_path,
        skf_rmse,
        tuned_filter_rmse,
        tuned_filter_name_pretty,
    )

    # 6. Plot absolute state errors
    # Determine common estimate length and corresponding time vector / true states
    # Assuming X_hat_skf_abs and X_hat_tuned_abs have the same length,
    # which should be true if both are steady or both are full KFs.
    # If one is steady and other is full, their lengths might differ.
    # For simplicity, we'll use SKF's length as reference.
    # A more robust solution might be needed if lengths frequently differ.

    if X_hat_skf_abs.shape[1] != X_hat_tuned_abs.shape[1]:
        print(
            "Warning: SKF and Tuned filter estimate lengths differ. "
            f"SKF: {X_hat_skf_abs.shape[1]}, Tuned: {X_hat_tuned_abs.shape[1]}. "
            "Error plots might be misleading or not generated."
        )
        # Decide on a strategy: skip, or use shortest, or plot separately.
        # For now, we proceed, which might lead to an error if np.abs tries to operate on different shapes
        # or plots look misaligned. Let's try to align to the SKF length for now.
        # This part needs careful consideration if lengths differ.
        # A simple fix is to use the already aligned true states for metrics if lengths are same
        if X_hat_skf_abs.shape[1] == X_hat_tuned_abs.shape[1]:
            print("Estimate lengths match. Proceeding with error plots.")
        else:
            print(
                "Estimate lengths differ. SKIPPING error plots as alignment is non-trivial for combined plot."
            )
            return  # Skip error plotting

    common_est_len = X_hat_skf_abs.shape[1]
    time_vector_for_errors = T_out[-common_est_len:]
    true_states_for_errors = X_states_abs_true[:, -common_est_len:]

    # Derive output path for error plot
    output_dir_errors = os.path.dirname(output_plot_path)
    output_basename_errors = os.path.basename(output_plot_path)
    output_name_errors, output_ext_errors = os.path.splitext(output_basename_errors)

    if output_name_errors.endswith("_comparison_plot"):
        base_name_for_errors = output_name_errors[: -len("_comparison_plot")]
    else:
        base_name_for_errors = output_name_errors
    output_filename_errors = f"{base_name_for_errors}_errors_plot{output_ext_errors}"
    final_output_path_errors = os.path.join(output_dir_errors, output_filename_errors)

    plot_cstr_state_errors(
        time_vector_for_errors,
        true_states_for_errors,
        X_hat_skf_abs,
        X_hat_tuned_abs,
        num_reactors,
        final_output_path_errors,
        tuned_filter_name_pretty,
    )


def plot_cstr_iteration_sweep_results(
    data_file_path: str,
    output_path: str = None,
):
    """
    Plot the results of an iteration sweep for CSTR models.

    Args:
        data_file_path: Path to the pickle file containing iteration sweep results.
        output_path: Path to save the output plot (if None, auto-generated).
    """
    print(f"Loading CSTR iteration sweep results from {data_file_path}")

    with open(data_file_path, "rb") as f:
        results_data = pickle.load(f)

    model_info = results_data.get("model_info", {})
    metric_name = model_info.get("metric", "rmse")
    iter_count_sweep_values = model_info.get("iter_count_sweep", [])

    kf_data = results_data.get("kf", {})
    kf_score = kf_data.get("score")

    # Assuming steady_kalman_filter data might be keyed as 'sskf' or 'steady_kalman_filter'
    sskf_data = results_data.get("sskf", results_data.get("steady_kalman_filter", {}))
    sskf_score = sskf_data.get("score")

    # Optional: A more optimal baseline if available in the data
    optimal_baseline_data = results_data.get("exact_hsskf", {})  # Example key
    optimal_baseline_score = optimal_baseline_data.get("score")

    all_best_scores_sweep = results_data.get("all_best_scores", {})
    # best_global_params_sweep = results_data.get("best_global_params", {})

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))

    # Determine the varied parameter (e.g., step_size or (coef_s, coef_o))
    # This part might need adjustment based on how tune_num_iters.py for CSTR stores keys
    varied_param_keys = sorted(all_best_scores_sweep.keys())

    colormap = plt.get_cmap("viridis")
    num_varied_params = len(varied_param_keys)

    for idx, param_key in enumerate(varied_param_keys):
        param_scores_data = all_best_scores_sweep[param_key]
        if not param_scores_data:
            continue

        # Extract iter_counts and scores, ensuring iter_counts are integers for sorting
        iter_scores_pairs = sorted(
            [(int(k), v) for k, v in param_scores_data.items()], key=lambda x: x[0]
        )

        if not iter_scores_pairs:
            continue

        iter_counts, scores = zip(*iter_scores_pairs)

        # param_label = f"Param: {param_key}"  # Default label
        # # Try to make a prettier label if param_key is a number (e.g. step_size)
        # try:
        #     param_float = float(param_key)
        #     param_label = f"$\gamma = {param_float:.3f}$"  # Assuming gamma is the step size parameter
        # except (ValueError, TypeError):
        #     if isinstance(param_key, tuple) and len(param_key) == 2:
        #         param_label = (
        #             f"($\lambda_s={param_key[0]:.2f}, \lambda_o={param_key[1]:.2f}$)"
        #         )

        plt.plot(
            iter_counts,
            scores,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=7,
            color=(
                colormap(idx / max(1, num_varied_params - 1))
                if num_varied_params > 1
                else "C0"
            ),
            # label=param_label,
        )

    # if kf_score is not None:
    #     plt.axhline(
    #         y=kf_score,
    #         color="blue",
    #         linestyle=":",
    #         linewidth=2,
    #         label=f"KF ({kf_score:.3f})",
    #     )
    # if sskf_score is not None:
    #     plt.axhline(
    #         y=sskf_score,
    #         color="green",
    #         linestyle="--",
    #         linewidth=2,
    #         label=f"Steady KF ({sskf_score:.3f})",
    #     )
    # if optimal_baseline_score is not None:
    #     plt.axhline(
    #         y=optimal_baseline_score,
    #         color="purple",
    #         linestyle="-.",
    #         linewidth=2,
    #         label=f"Optimal Baseline ({optimal_baseline_score:.3f})",  # Adjust label as needed
    #     )

    plt.xlabel(r"$\tilde k$")
    plt.ylabel(
        f"{metric_name.upper()} Ratio"
        if "ratio" in metric_name
        else metric_name.upper()
    )
    # plt.title("Filter Performance vs. Number of Iterations for CSTR Model")

    if iter_count_sweep_values:
        plt.xticks(sorted(list(set(int(x) for x in iter_count_sweep_values))))
        # Consider log scale if appropriate for iter_count_sweep_values
        # if all(i > 0 for i in iter_count_sweep_values) and len(iter_count_sweep_values) > 3:
        #     plt.xscale("log")
        #     plt.xticks(sorted(iter_count_sweep_values), [str(x) for x in sorted(iter_count_sweep_values)])

    if num_varied_params > 1 or kf_score is not None or sskf_score is not None:
        plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_path:
        final_output_path = output_path
    else:
        input_basename = os.path.basename(data_file_path)
        input_name_no_ext = os.path.splitext(input_basename)[0]
        # Try to clean up the name if it's from tune_num_iters.py output
        # e.g., hsskf_iter_count_cstr_n4_p10_sp10_sm10_steps1000_seed0_realistic_rmse_results
        name_parts = (
            input_name_no_ext.split("_optimistic_")[0]
            .split("_realistic_")[0]
            .split("_results")[0]
            .replace("hsskf_iter_count_", "")  # Generalize if other filters are swept
        )
        output_filename = f"{name_parts}_iter_sweep_plot.pdf"
        final_output_path = os.path.join("figures", output_filename)

    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
    print(f"CSTR iteration sweep plot saved to {final_output_path}")
    plt.show()


def main():
    """Main function to handle command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Plot CSTR trajectory with filter estimates or iteration sweep results."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tune_filter_results",
        type=str,
        help="Path to the pickle file containing tuning results from tune_filter.py for a CSTR model.",
    )
    input_group.add_argument(
        "--sweep_iters_results",
        type=str,
        help="Path to the pickle file containing iteration sweep results for a CSTR model.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output plot (default: auto-generated in figures/ folder based on input filename).",
    )
    parser.add_argument(
        "--force-simulation",  # Kept for structural similarity with plot_vehicle_results.py
        action="store_true",  # Not currently used by CSTR trajectory plotting path
        help="Force using simulation data for trajectory plots if applicable (not used by CSTR plots currently).",
    )
    args = parser.parse_args()

    # Determine plot mode
    plot_mode = None
    if args.tune_filter_results:
        data_file_path = args.tune_filter_results
        plot_mode = "cstr_trajectory"
    elif args.sweep_iters_results:
        data_file_path = args.sweep_iters_results
        plot_mode = "iteration_sweep"
    else:
        # Should not happen due to mutually_exclusive_group being required
        print(
            "Error: Either --tune_filter_results or --sweep_iters_results must be specified."
        )
        sys.exit(1)

    if not os.path.exists(data_file_path):
        print(f"Error: The specified file '{data_file_path}' does not exist.")
        sys.exit(1)

    setup_matplotlib_for_latex()

    if plot_mode == "cstr_trajectory":
        try:
            sim_data_loaded, best_params, model_info, filter_type, num_reactors = (
                reconstruct_cstr_model_and_filters(
                    args.tune_filter_results
                )  # Use data_file_path for consistency
            )
        except Exception as e:
            print(f"Error during data loading or model reconstruction: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Determine output path for trajectory plots
        if args.output:
            output_plot_path = args.output
        else:
            input_basename = os.path.basename(data_file_path)  # Use data_file_path
            name_parts = (
                input_basename.split("_optimistic_")[0]
                .split("_realistic_")[0]
                .split("_results.pkl")[0]
            )
            output_filename = f"{name_parts}_comparison_plot.pdf"
            # Save CSTR plots into a subdirectory for better organization
            output_plot_path = os.path.join("figures", output_filename)

        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        run_cstr_filters_and_plot(
            sim_data_loaded,
            best_params,
            model_info,
            filter_type,
            num_reactors,
            output_plot_path,
        )
    elif plot_mode == "iteration_sweep":
        plot_cstr_iteration_sweep_results(data_file_path, args.output)


if __name__ == "__main__":
    main()
