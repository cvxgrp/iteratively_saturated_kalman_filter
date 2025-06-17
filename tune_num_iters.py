"""
Grid search script for HuberizedSSKalmanFilter with focus on iteration count tuning.

This script performs a grid search over the parameters of the HuberizedSSKalmanFilter
for different values of the num_iters parameter. The script:
1. Sets up a simulation environment (using the vehicle_ss model)
2. Defines a grid of hyperparameters including coef_s, coef_o, and num_iters
3. Runs the grid search across all parameter combinations
4. For each iteration count, finds the best performing coef_s/coef_o combination
5. Creates a line plot showing the best metric value vs iteration count
"""

import os
import pickle
import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from iskf.models.vehicle import vehicle_ss
from iskf.models.cstr import cascaded_cstr_ss
from iskf.grid_search import grid_search_filter_hyperparams
from iskf.filters.util import chi_squared_quantile
from iskf.metrics import METRIC_REGISTRY

from iskf.filters.steady_iskf import (
    SteadyIterSatKalmanFilter,
)
from iskf.filters.kalman_filter import KalmanFilter
from iskf.filters.steady_kalman_filter import SteadyKalmanFilter

# --- Global Configuration ---
SWEEP_RESOLUTION = 20
DEFAULT_METRIC = "rmse"  # Default metric, can be overridden by command line
NUM_PARALLEL_JOBS_RUN = -1
PARAMETER_SEARCH_DIR = os.path.join("results", "parameter_search_data")
PARAMETER_SEARCH_PLOTS = "figures"

# Define step size sweep range
STEP_SIZE_MIN = 1.0
STEP_SIZE_MAX = 1.0
STEP_SIZE_COUNT = 1

ITER_COUNT_SWEEP = np.arange(1, 11)

# Setup the coef_s and coef_o sweep ranges (needed for exact HSSKF baseline too)
alpha_min_hsskf, alpha_max_hsskf = 1e-12, 1 - 1e-12
coef_s_min = 1e-1  # np.sqrt(chi_squared_quantile(alpha_min_hsskf, nx))
coef_s_max = 10  # np.sqrt(chi_squared_quantile(alpha_max_hsskf, nx))
coef_o_min = 1e-1  # np.sqrt(chi_squared_quantile(alpha_min_hsskf, ny))
coef_o_max = 10  # np.sqrt(chi_squared_quantile(alpha_max_hsskf, ny))

coef_s_sweep = np.geomspace(coef_s_min, coef_s_max, SWEEP_RESOLUTION)
coef_o_sweep = np.geomspace(coef_o_min, coef_o_max, SWEEP_RESOLUTION)

coef_s_sweep = np.append(coef_s_sweep, np.inf)
coef_o_sweep = np.append(coef_o_sweep, np.inf)


def compute_filter_score(
    estimates, true_states, y_measurements, system_model, metric_name, optimistic
):
    """
    Compute filter performance score either in optimistic or realistic mode.

    Args:
        estimates: State estimates from the filter
        true_states: True states for comparison (optimistic mode)
        y_measurements: Actual measurements (realistic mode)
        system_model: System model object with C matrix for measurement projection
        metric_name: Name of metric to use from METRIC_REGISTRY
        optimistic: Whether to use optimistic (true state) or realistic (measurement) comparison

    Returns:
        score: The computed metric score
    """
    # Get the metric function from the registry
    metric_func = METRIC_REGISTRY[metric_name]

    # Align shapes for consistent evaluation
    x_true_for_metric = None
    if true_states.shape[1] == estimates.shape[1]:
        x_true_for_metric = true_states
    elif true_states.shape[1] == estimates.shape[1] + 1:
        # Skip first state if dimensions don't match
        x_true_for_metric = true_states[:, 1:]
    else:
        print(
            f"Warning: Shape mismatch: true_states {true_states.shape}, estimates {estimates.shape}"
        )
        if true_states.shape[1] > estimates.shape[1]:
            # Try to align by truncating true_states
            x_true_for_metric = true_states[:, -estimates.shape[1] :]
        else:
            # Try to align by truncating estimates
            return metric_func(estimates[:, -true_states.shape[1] :], true_states)

    if optimistic:
        # Optimistic: Compare estimated states with true states
        return metric_func(estimates, x_true_for_metric)
    else:
        # Realistic: Compare predicted measurement with actual measurement
        if hasattr(system_model, "C"):
            C = system_model.C
        else:
            C = (
                system_model.C.A
            )  # For LTI systems where C is a control.StateSpace attribute

        predicted_measurements = np.dot(C, estimates)
        return metric_func(predicted_measurements, y_measurements)


# --- Function to tune HSSKF with iteration count sweep ---
def _tune_hsskf_iter_count(
    system_model_obj,
    process_noise_cov,
    measurement_noise_cov,
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
):
    """Tunes HuberizedSSKalmanFilter with a focus on iteration count and step size."""
    print(
        "\n  Step 2: Setting up HSSKF grid search parameters with iteration count and step size sweep..."
    )

    # Run SteadyIterSatKalmanFilter with use_exact_mean_solve=True
    # and grid search over coef_s, coef_o for baseline
    print(
        "\n  Running Exact HSSKF (use_exact_mean_solve=True) with coef_s/coef_o grid search for baseline comparison..."
    )

    exact_hsskf_param_grid = [
        {
            "coef_s": cs,
            "coef_o": co,
            "num_iters": 1,  # Not critical for exact solver
            "step_size": 1.0,  # Not critical for exact solver
            "use_exact_mean_solve": True,
        }
        for cs in coef_s_sweep
        for co in coef_o_sweep
    ]

    initial_filter_for_exact_run = SteadyIterSatKalmanFilter(
        system_model=system_model_obj,
        cov_input=process_noise_cov,
        cov_measurement=measurement_noise_cov,
        # Other params will be set by grid_search_filter_hyperparams from exact_hsskf_param_grid
    )

    best_exact_params, exact_hsskf_score, _ = grid_search_filter_hyperparams(
        initial_filter_for_exact_run,
        exact_hsskf_param_grid,  # Grid over coef_s, coef_o with use_exact_mean_solve=True
        T_out,
        Y_meas,
        X_states if optimistic else None,
        x0_est,
        P0_init,
        metric,
        n_jobs=n_jobs,  # Can use multiple jobs for this baseline search too
    )

    print(f"  Best parameters for Exact HSSKF (use_exact_mean_solve=True):")
    print(f"    coef_s: {best_exact_params['coef_s']:.4f}")
    print(f"    coef_o: {best_exact_params['coef_o']:.4f}")
    print(f"  Exact HSSKF {metric.upper()} score: {exact_hsskf_score:.4f}")

    # Define step sizes to sweep over
    step_size_sweep = np.geomspace(STEP_SIZE_MIN, STEP_SIZE_MAX, STEP_SIZE_COUNT)
    print(f"    Step size sweep values: {step_size_sweep}")

    # Dictionary to store all results across different step sizes
    all_step_size_results = {}
    all_best_scores = {}  # For tracking best score across all combinations

    # Base filename components
    base_filename = f"hsskf_iter_count_{sim_data_filename}"
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"

    # Create one dictionary to store all the results
    complete_results = {
        "model_info": {
            "sim_data_filename": sim_data_filename,
            "metric": metric,
            "optimistic": optimistic,
            "sweep_resolution": sweep_res,
            "step_size_sweep": step_size_sweep.tolist(),
            "iter_count_sweep": ITER_COUNT_SWEEP.tolist(),
            "coef_s_sweep": coef_s_sweep.tolist(),
            "coef_o_sweep": coef_o_sweep.tolist(),
        },
        "exact_hsskf": {
            "best_params": best_exact_params,
            "score": exact_hsskf_score,
        },
        "results_by_step_size": {},
    }

    best_global_score = float("inf")
    best_global_params = None

    # First loop through step sizes
    for step_size in step_size_sweep:
        print(f"\n====================================================")
        print(f"  Processing step size: {step_size:.4f}")
        print(f"====================================================")

        # Dictionary to store the best results for each iteration count at this step size
        iter_count_results = {}

        # Loop through each iteration count for this step size
        for iter_count in ITER_COUNT_SWEEP:
            print(
                f"\n  --- Running grid search for num_iters = {iter_count}, step_size = {step_size:.4f} ---"
            )

            # Create parameter grid for this combination
            param_grid = [
                {
                    "coef_s": cs,
                    "coef_o": co,
                    "num_iters": iter_count,
                    "step_size": step_size,
                }
                for cs in coef_s_sweep
                for co in coef_o_sweep
            ]

            # Create initial filter with current iteration count and step size
            initial_filter = SteadyIterSatKalmanFilter(
                system_model=system_model_obj,
                cov_input=process_noise_cov,
                cov_measurement=measurement_noise_cov,
                num_iters=iter_count,
                step_size=step_size,
            )

            # Run grid search for this combination
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

            # Store the results
            iter_count_results[iter_count] = {
                "best_params": best_params,
                "best_score": best_score,
                "all_results": all_results,
            }

            # Update global best score if this is better
            if best_score < best_global_score:
                best_global_score = best_score
                best_global_params = {
                    "step_size": step_size,
                    "iter_count": iter_count,
                    "coef_s": best_params["coef_s"],
                    "coef_o": best_params["coef_o"],
                }

            print(
                f"  Best parameters for num_iters = {iter_count}, step_size = {step_size:.4f}:"
            )
            print(f"    coef_s: {best_params['coef_s']:.4f}")
            print(f"    coef_o: {best_params['coef_o']:.4f}")
            print(f"    Best {metric.upper()} score: {best_score:.4f}")

        # Store all results for this step size
        all_step_size_results[step_size] = iter_count_results
        complete_results["results_by_step_size"][float(step_size)] = iter_count_results

        # Extract best scores for each iteration count at this step size
        all_best_scores[step_size] = {
            ic: results["best_score"] for ic, results in iter_count_results.items()
        }

    # Add global best parameters and best scores to the complete results
    complete_results["best_global_params"] = best_global_params
    complete_results["best_global_score"] = best_global_score
    complete_results["all_best_scores"] = {
        float(k): v for k, v in all_best_scores.items()
    }

    # Save the combined results in a single file
    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )
    with open(results_filename, "wb") as f:
        pickle.dump(complete_results, f)
    print(f"\nAll results saved to {results_filename}")

    # Create a combined plot showing best metric vs iteration count for different step sizes
    _plot_combined_step_size_results(
        all_best_scores,
        ITER_COUNT_SWEEP,
        metric,
        optimistic,
        os.path.join(
            PARAMETER_SEARCH_PLOTS,
            f"{base_filename}_combined_plot{opt_suffix}{metric_suffix}.pdf",
        ),
        exact_hsskf_score,
        best_global_params,
    )

    # Create a heatmap of step size vs iteration count
    _plot_heatmap_step_size_iter_count(
        all_best_scores,
        metric,
        optimistic,
        os.path.join(
            PARAMETER_SEARCH_PLOTS,
            f"{base_filename}_heatmap{opt_suffix}{metric_suffix}.pdf",
        ),
        exact_hsskf_score,
    )

    # Print the global best parameters
    print("\n================================================")
    print(f"BEST OVERALL PARAMETERS:")
    print(f"  Step size: {best_global_params['step_size']:.4f}")
    print(f"  Iteration count: {best_global_params['iter_count']}")
    print(f"  coef_s: {best_global_params['coef_s']:.4f}")
    print(f"  coef_o: {best_global_params['coef_o']:.4f}")
    print(f"  Best {metric.upper()} score: {best_global_score:.4f}")

    if exact_hsskf_score is not None:
        improvement = (exact_hsskf_score - best_global_score) / exact_hsskf_score * 100
        print(f"  Improvement over Exact HSSKF: {improvement:.2f}%")
    print("================================================\n")

    # Evaluate baseline filters for comparison
    # Create and evaluate the steady-state Kalman filter (SSKF)
    print("Evaluating steady-state Kalman filter...")
    sskf = SteadyKalmanFilter(
        system_model=system_model_obj,
        cov_input=process_noise_cov,
        cov_measurement=measurement_noise_cov,
    )
    sskf_estimates = sskf.estimate(
        T_out, Y_meas, x_initial_estimate=x0_est, P_initial=P0_init
    )

    # Compute SSKF score using the helper function
    sskf_score = compute_filter_score(
        sskf_estimates, X_states, Y_meas, system_model_obj, metric, optimistic
    )
    print(f"SSKF {metric} score: {sskf_score:.6f}")

    # Create and evaluate the regular Kalman filter (KF)
    print("Evaluating regular Kalman filter...")
    kf = KalmanFilter(
        system_model=system_model_obj,
        cov_input=process_noise_cov,
        cov_measurement=measurement_noise_cov,
    )
    kf_estimates = kf.estimate(
        T_out, Y_meas, x_initial_estimate=x0_est, P_initial=P0_init
    )

    # Compute KF score using the helper function
    kf_score = compute_filter_score(
        kf_estimates, X_states, Y_meas, system_model_obj, metric, optimistic
    )
    print(f"KF {metric} score: {kf_score:.6f}")

    # Create results dictionary AFTER computing all scores
    results = {
        "model_info": {
            "metric": metric,
            "optimistic": optimistic,
            "sim_data_path": sim_data_filename,
            "step_size_sweep": step_size_sweep.tolist(),
            "iter_count_sweep": ITER_COUNT_SWEEP.tolist(),
        },
        "exact_hsskf": {
            "best_params": best_exact_params,
            "score": exact_hsskf_score,
        },
        "sskf": {
            "score": sskf_score,
        },
        "kf": {
            "score": kf_score,
        },
        "all_best_scores": all_best_scores,
        "best_global_params": best_global_params,
    }

    # Save the results dictionary
    if sweep_res:
        result_filename = os.path.basename(sim_data_filename).split(".")[0]
        result_filename = f"hsskf_iter_count_{result_filename}_{metric}_results.pkl"
        result_filepath = os.path.join(
            "results", "parameter_search_data", result_filename
        )
        os.makedirs(os.path.dirname(result_filepath), exist_ok=True)

        with open(result_filepath, "wb") as f:
            # This is where we save the complete results including SSKF and KF scores
            pickle.dump(results, f)

        print(f"Results saved to {result_filepath}")

    # Print a message to confirm the SSKF and KF scores were calculated and saved
    print(f"SSKF and KF scores calculated and included in saved results.")
    print(f"  SSKF {metric} score: {sskf_score:.6f}")
    print(f"  KF {metric} score: {kf_score:.6f}")

    return all_step_size_results, exact_hsskf_score


def _plot_combined_step_size_results(
    all_best_scores: dict,
    ITER_COUNT_SWEEP: np.ndarray,
    metric: str,
    optimistic: bool,
    save_path: str = None,
    baseline_score: float = None,
    best_params: dict = None,
):
    """Creates a comprehensive plot showing the best metric for each iteration count across multiple step sizes."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Create a colormap for different step sizes
    colormap = plt.get_cmap("viridis")
    num_step_sizes = len(all_best_scores)

    # Sort step sizes for consistent plotting
    step_sizes = sorted(all_best_scores.keys())

    # Create a list to store lines for annotations
    best_points = []
    best_overall_point = None

    # Plot each step size as a separate line
    for i, step_size in enumerate(step_sizes):
        # Make sure we have all iteration counts for this step size
        valid_iter_counts = sorted(
            [ic for ic in ITER_COUNT_SWEEP if ic in all_best_scores[step_size]]
        )
        if not valid_iter_counts:
            continue

        scores = [all_best_scores[step_size][ic] for ic in valid_iter_counts]

        # Find best score for this step size
        best_idx = np.argmin(scores)
        best_iter = valid_iter_counts[best_idx]
        best_score = scores[best_idx]
        best_points.append((best_iter, best_score, step_size))

        # Check if this is the global best
        if (
            best_params
            and step_size == best_params["step_size"]
            and best_iter == best_params["iter_count"]
        ):
            best_overall_point = (best_iter, best_score)

        # Plot the line with color from our colormap
        (line,) = plt.plot(
            valid_iter_counts,
            scores,
            "o-",
            color=colormap(i / max(1, num_step_sizes - 1)),
            linewidth=2,
            markersize=6,
            label=f"Step size = {step_size:.4e}",
        )

    # Add horizontal line for the baseline score (Exact HSSKF)
    if baseline_score is not None:
        plt.axhline(
            y=baseline_score,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Exact HSSKF RMSE ({baseline_score:.4f})",
        )

    # Highlight best point for each step size
    for iter_count, score, step_size in best_points:
        idx = step_sizes.index(step_size)
        color = colormap(idx / max(1, num_step_sizes - 1))
        plt.plot(iter_count, score, "s", color=color, markersize=10, alpha=0.7)

    # Highlight overall best point
    if best_overall_point:
        plt.plot(
            best_overall_point[0],
            best_overall_point[1],
            "*",
            color="red",
            markersize=18,
            alpha=0.9,
            label=f"Best overall: iter={best_params['iter_count']}, step={best_params['step_size']:.4e}",
        )

    # Set labels and title
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel(f"{metric.upper()} Score", fontsize=12)
    title = f"HSSKF Performance Across Different Step Sizes\n(Optimistic: {optimistic}, Metric: {metric.upper()})"
    if best_overall_point:
        improvement = (
            (baseline_score - best_overall_point[1]) / baseline_score * 100
            if baseline_score
            else 0
        )
        title += f"\nBest: {best_overall_point[1]:.4f} @ iter={best_params['iter_count']}, step={best_params['step_size']:.4e}"
        if baseline_score:
            title += f" ({improvement:.1f}% improvement over Exact HSSKF)"
    plt.title(title, fontsize=14)

    # Add legend with smaller font for readability
    plt.legend(loc="best", fontsize=9)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Enhance x-axis with logarithmic scale and custom ticks
    plt.xscale("log")
    plt.xticks(sorted(ITER_COUNT_SWEEP), [str(x) for x in sorted(ITER_COUNT_SWEEP)])

    # Add annotations for important points
    if best_overall_point:
        plt.annotate(
            f"{best_overall_point[1]:.4f}",
            xy=(best_overall_point[0], best_overall_point[1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            fontsize=10,
        )

    # Tight layout
    plt.tight_layout()

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Combined step size performance plot saved to {save_path}")
        plt.close()


def _plot_heatmap_step_size_iter_count(
    all_best_scores: dict,
    metric: str,
    optimistic: bool,
    save_path: str = None,
    baseline_score: float = None,
):
    """Creates a heatmap of step size vs iteration count with metric values."""
    # Convert dictionaries to arrays for heatmap
    step_sizes = sorted(all_best_scores.keys())
    iter_counts = sorted(
        list(
            set(ic for step_dict in all_best_scores.values() for ic in step_dict.keys())
        )
    )

    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(step_sizes), len(iter_counts)))

    # Fill the array with metric values
    for i, step_size in enumerate(step_sizes):
        for j, iter_count in enumerate(iter_counts):
            if iter_count in all_best_scores[step_size]:
                heatmap_data[i, j] = all_best_scores[step_size][iter_count]
            else:
                heatmap_data[i, j] = np.nan

    # Set up the plot
    sns.set_style("white")
    plt.figure(figsize=(14, 8))

    # Create a diverging colormap centered around the KF score if provided
    if baseline_score is not None:
        # Create a custom colormap that's centered around the KF score
        # Values better than KF will be one color, worse will be another
        min_val = np.nanmin(heatmap_data)
        max_val = np.nanmax(heatmap_data)

        # Need to map these to 0-1 range for the colormap
        # center_point = (baseline_score - min_val) / (max_val - min_val) # Centering logic for diverging map

        # Default to "RdYlGn_r" (reversed RdYlGn) colormap if baseline_score is available
        cmap = "RdYlGn_r"
    else:
        # Use viridis colormap if no baseline_score
        cmap = "viridis"

    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,  # Show values in cells
        fmt=".4f",  # Format for annotations
        cmap=cmap,  # Colormap
        xticklabels=iter_counts,
        yticklabels=[f"{ss:.4e}" for ss in step_sizes],
    )

    # Set labels and title
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Step Size", fontsize=12)
    plt.title(
        f"HSSKF Performance Heatmap: Step Size vs Iteration Count\n(Optimistic: {optimistic}, Metric: {metric.upper()})",
        fontsize=14,
    )

    # Draw a line or marker to indicate the KF score if provided
    if baseline_score is not None:
        plt.figtext(
            0.5,
            0.01,
            f"Exact HSSKF {metric.upper()} score: {baseline_score:.4f}",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

    # Adjust layout to make room for annotations
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap plot saved to {save_path}")
        plt.close()


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Tune iteration count and other hyperparameters for HuberizedSSKalmanFilter."
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
        default=SWEEP_RESOLUTION,
        help=f"Resolution of the coef_s/coef_o sweep grid. Default: {SWEEP_RESOLUTION}",
    )
    parser.add_argument(
        "--sim_data_path",
        type=str,
        required=True,
        help="Path to the pickled simulation data file generated by vehicle_example.py.",
    )
    parser.add_argument(
        "--step_size_count",
        type=int,
        default=STEP_SIZE_COUNT,
        help=f"Number of step sizes to sweep over. Default: {STEP_SIZE_COUNT}",
    )
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    optimistic = args.optimistic
    metric = args.metric.lower()
    sweep_res = args.sweep_resolution

    # Extract the base filename without extension
    sim_data_filename = pathlib.Path(args.sim_data_path).stem

    print(f"Starting HSSKF iteration count and step size tuning...")
    print(
        f"Optimistic mode: {optimistic} (Using {'true states' if optimistic else 'predicted measurements'} for metrics)"
    )
    print(f"Metric: {metric.upper()}")
    print(f"Sweep resolution: {sweep_res}")
    print(
        f"Step size range: {STEP_SIZE_MIN:.4e} to {STEP_SIZE_MAX:.4e}, {args.step_size_count} values"
    )
    print(f"Simulation data: {sim_data_filename}")

    # Create directories for saving results
    os.makedirs(PARAMETER_SEARCH_DIR, exist_ok=True)
    os.makedirs(PARAMETER_SEARCH_PLOTS, exist_ok=True)

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

        # Get the noise covariance matrices from the loaded data
        if "cov_input" in loaded_data:
            process_noise_cov = loaded_data["cov_input"]
            print(
                "    Using 'cov_input' from loaded data for process noise covariance."
            )
        else:
            print(
                "Error: Process noise covariance ('cov_input') not found in simulation data."
            )
            return

        if "cov_measurement" in loaded_data:
            measurement_noise_cov = loaded_data["cov_measurement"]
            print(
                "    Using 'cov_measurement' from loaded data for measurement noise covariance."
            )
        elif "meas_noise_filter" in loaded_data:  # Fallback for older data
            measurement_noise_cov = loaded_data["meas_noise_filter"]
            print(
                "    Using 'meas_noise_filter' (older format) from loaded data for measurement noise covariance."
            )
        else:
            print(
                "Error: Measurement noise covariance ('cov_measurement' or 'meas_noise_filter') not found in simulation data."
            )
            return

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

    # Run the HSSKF tuning with iteration count sweep
    all_step_size_results, exact_hsskf_score_main = _tune_hsskf_iter_count(
        system_model_obj,
        process_noise_cov,
        measurement_noise_cov,
        T_out,
        Y_measurements,
        X_true_states,
        x0_estimate_filter,
        P0_initial_val,
        nx_sim_val,
        ny_sim_val,
        sweep_res,
        metric,
        NUM_PARALLEL_JOBS_RUN,
        optimistic,
        sim_data_filename,
    )

    print(f"\nHSSKF iteration count and step size tuning finished.")
    print(f"Mode: {'Optimistic' if optimistic else 'Realistic'}")
    print(f"Metric: {metric.upper()}")
    print(f"Exact HSSKF {metric.upper()} score: {exact_hsskf_score_main:.4f}")

    # Extract initial best parameters for Exact HSSKF (used as a baseline)
    best_exact_params = {
        "step_size": None,
        "iter_count": "exact",
        "coef_s": None,
        "coef_o": None,
    }

    # Create the step size sweep array
    step_size_sweep = np.geomspace(STEP_SIZE_MIN, STEP_SIZE_MAX, args.step_size_count)

    # Extract all_best_scores
    all_best_scores = {}
    for step_size, iter_results in all_step_size_results.items():
        all_best_scores[step_size] = {
            ic: results["best_score"] for ic, results in iter_results.items()
        }

    # Extract best global parameters - find the best parameters across all step sizes and iterations
    best_global_params = {}
    best_global_score = float("inf")

    for step_size, iter_results in all_step_size_results.items():
        for iter_count, result_data in iter_results.items():
            if result_data["best_score"] < best_global_score:
                best_global_score = result_data["best_score"]
                best_global_params = {
                    "step_size": step_size,
                    "iter_count": iter_count,
                    "coef_s": result_data["best_params"]["coef_s"],
                    "coef_o": result_data["best_params"]["coef_o"],
                }

    # Now create and save the final results file with all data including SSKF and KF results
    # We need to load the results we saved earlier that contain the SSKF and KF scores

    # First, construct the filepath for the results file that was created in _tune_hsskf_iter_count
    result_filename = os.path.basename(args.sim_data_path).split(".")[0]
    result_filename = f"hsskf_iter_count_{result_filename}_{metric}_results.pkl"
    result_filepath = os.path.join("results", "parameter_search_data", result_filename)

    # Load the complete results file that includes SSKF and KF scores
    complete_detailed_results = None
    try:
        with open(result_filepath, "rb") as f:
            complete_detailed_results = pickle.load(f)
        print(f"Loaded detailed results with SSKF and KF scores from {result_filepath}")
    except FileNotFoundError:
        print(f"Warning: Could not find detailed results file at {result_filepath}")

    # Create the results dictionary for the main output file
    # Include SSKF and KF results if available
    if complete_detailed_results:
        # Add SSKF and KF results to the output file
        complete_results = {
            "model_info": {
                "metric": metric,
                "optimistic": optimistic,
                "sim_data_path": args.sim_data_path,
                "step_size_sweep": step_size_sweep.tolist(),
                "iter_count_sweep": ITER_COUNT_SWEEP.tolist(),
            },
            "exact_hsskf": {
                "best_params": best_exact_params,
                "score": exact_hsskf_score_main,
            },
            "sskf": complete_detailed_results.get("sskf", {}),
            "kf": complete_detailed_results.get("kf", {}),
            "all_best_scores": all_best_scores,
            "best_global_params": best_global_params,
        }
    else:
        # Create standard results without SSKF and KF data
        complete_results = {
            "model_info": {
                "metric": metric,
                "optimistic": optimistic,
                "sim_data_path": args.sim_data_path,
                "step_size_sweep": step_size_sweep.tolist(),
                "iter_count_sweep": ITER_COUNT_SWEEP.tolist(),
            },
            "exact_hsskf": {
                "best_params": best_exact_params,
                "score": exact_hsskf_score_main,
            },
            "all_best_scores": all_best_scores,
            "best_global_params": best_global_params,
        }

    # Create the custom directory for saving results
    os.makedirs(PARAMETER_SEARCH_DIR, exist_ok=True)

    # Save the combined results
    opt_suffix = "_optimistic" if optimistic else "_realistic"
    metric_suffix = f"_{metric}"
    base_filename = f"hsskf_iter_count_{sim_data_filename}"

    results_filename = os.path.join(
        PARAMETER_SEARCH_DIR,
        f"{base_filename}{opt_suffix}{metric_suffix}_results.pkl",
    )

    with open(results_filename, "wb") as f:
        pickle.dump(complete_results, f)

    print(f"\nAll results saved to {results_filename}")
    if complete_detailed_results:
        print(f"Including SSKF and KF scores in the output file:")
        print(
            f"  SSKF {metric} score: {complete_detailed_results.get('sskf', {}).get('score')}"
        )
        print(
            f"  KF {metric} score: {complete_detailed_results.get('kf', {}).get('score')}"
        )
    else:
        print(f"Warning: SSKF and KF scores were not included in the output file.")

    # Print the path to the saved data file
    print(f"\nAll results saved to: {os.path.abspath(PARAMETER_SEARCH_DIR)}")
    print(f"Plots saved to: {os.path.abspath(PARAMETER_SEARCH_PLOTS)}")


if __name__ == "__main__":
    main()
