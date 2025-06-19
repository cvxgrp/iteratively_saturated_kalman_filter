"""
Script to simulate a vehicle model and save the simulation data for later use
by hyperparameter tuning scripts. It also plots the raw simulation results.

If a filter_type is specified, it will also run that filter on the simulation data
and plot the filtering results, similar to vehicle_example.py.

Output files are saved in 'results/simulation_data/' with a filename
pattern 'vehicle_p<OUTLIER_PERCENT>_seed<SEED>.pkl'.
"""

import argparse
import os
import pickle
from typing import Optional
import json
import numpy as np
import matplotlib.pyplot as plt

# Project-specific imports
from iskf.models.vehicle import vehicle_ss
from iskf.simulator import Simulator
from iskf.metrics import METRIC_REGISTRY

# Import experimental filter implementations
from iskf.filters.kalman_filter import KalmanFilter
from iskf.filters.huber_kalman_filter import HuberKalmanFilter
from iskf.filters.iskf import (
    IterSatKalmanFilter,
)
from iskf.filters.steady_kalman_filter import SteadyKalmanFilter
from iskf.filters.steady_huber_kalman_filter import SteadyHuberKalmanFilter
from iskf.filters.steady_iskf import (
    SteadyIterSatKalmanFilter,
)
from iskf.filters.steady_regularized_kalman_filter import (
    SteadyRegularizedKalmanFilter,
)
from iskf.filters.weighted_likelihood_filter import WeightedLikelihoodFilter
from iskf.filters.steady_one_step_iskf import SteadyOneStepIterSatFilter
from iskf.filters.steady_two_step_iskf import SteadyTwoStepIterSatFilter
from iskf.filters.steady_three_term_iskf import SteadyThreeStepIterSatFilter


def plot_vehicle_simulation_data(
    T_out: np.ndarray,
    X_states: np.ndarray,
    Y_measurements: np.ndarray,
    X_hat_sequence: np.ndarray,
    U_proc_noise: np.ndarray,
    U_meas_noise: np.ndarray,
    dim_w_dyn: int,
    dim_w_meas: int,
):
    """
    Plots the results of the vehicle simulation and Kalman filter estimation.

    Args:
        T_out: Time vector.
        X_states: True state trajectory.
        Y_measurements: Noisy measurements.
        X_hat_sequence: Estimated state trajectory from Kalman filter.
        U_proc_noise: Generated process noise sequence.
        U_meas_noise: Generated measurement noise sequence.
        dim_w_dyn: Dimension of the dynamic process noise.
        dim_w_meas: Dimension of the measurement noise.
    """
    # Plot 1: Process Noise Components
    if dim_w_dyn > 0 and U_proc_noise.shape[1] > 0:  # Check if there is data to plot
        plt.figure(figsize=(12, 3 * dim_w_dyn))
        for i in range(dim_w_dyn):
            plt.subplot(dim_w_dyn, 1, i + 1)
            plt.plot(T_out, U_proc_noise[i, :], label=f"Process Noise w_dyn[{i}]")
            plt.ylabel(f"w_dyn[{i}]")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Time (s)")
        plt.suptitle("Process Noise Components (inputs to dynamics)")

    # Plot 2: Measurement Noise Components
    if dim_w_meas > 0 and U_meas_noise.shape[1] > 0:  # Check if there is data to plot
        plt.figure(figsize=(12, 3 * dim_w_meas))
        for i in range(dim_w_meas):
            plt.subplot(dim_w_meas, 1, i + 1)
            plt.plot(T_out, U_meas_noise[i, :], label=f"Measurement Noise w_meas[{i}]")
            plt.ylabel(f"w_meas[{i}]")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Time (s)")
        plt.suptitle("Measurement Noise Components (added to true output)")

    # Plot 3: Vehicle Trajectory (2D plane)
    plt.figure(figsize=(8, 6))
    plt.plot(X_states[0, 0], X_states[1, 0], "go", markersize=10, label="Start")
    plt.plot(X_states[0, -1], X_states[1, -1], "ro", markersize=10, label="End")
    plt.plot(X_states[0, :], X_states[1, :], label="True Vehicle Path", linewidth=2)
    plt.scatter(
        Y_measurements[0, :],
        Y_measurements[1, :],
        color="orange",
        alpha=0.6,
        marker="x",
        s=60,
        label="Measurements",
    )
    # Plot Kalman filter estimates. T_out[1:] corresponds to times for X_hat_sequence
    if X_hat_sequence is not None and X_hat_sequence.shape[1] > 0:
        plt.plot(
            X_hat_sequence[0, :],
            X_hat_sequence[1, :],
            label="Kalman Filter Estimate",
            linestyle="--",
            color="purple",
            linewidth=2,
        )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)

    x_min = np.min(X_states[0, :])
    x_max = np.max(X_states[0, :])
    y_min = np.min(X_states[1, :])
    y_max = np.max(X_states[1, :])

    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)


# Add a new function to plot state trajectories
def plot_state_trajectories(
    T_out: np.ndarray,
    X_states: np.ndarray,
    X_hat_sequence: np.ndarray,
    state_names=None,
):
    """
    Plots individual state trajectories for true and estimated states.

    Args:
        T_out: Time vector.
        X_states: True state trajectory.
        X_hat_sequence: Estimated state trajectory from filter.
        state_names: List of names for each state dimension.
    """
    if state_names is None:
        state_names = ["X Position", "Y Position", "X Velocity", "Y Velocity"]

    n_states = X_states.shape[0]

    # Get time indices that align with X_hat_sequence
    if X_hat_sequence.shape[1] == X_states.shape[1]:
        # Same length - use all time points
        T_hat = T_out
        X_true_aligned = X_states
    elif X_hat_sequence.shape[1] == X_states.shape[1] - 1:
        # X_hat is one shorter - estimates start from second time point
        T_hat = T_out[1:]
        X_true_aligned = X_states[:, 1:]
    else:
        print(
            f"Warning: Cannot align trajectories. X_states shape: {X_states.shape}, X_hat shape: {X_hat_sequence.shape}"
        )
        return

    # Create figure with subplots for each state
    plt.figure(figsize=(12, 3 * n_states))

    for i in range(n_states):
        plt.subplot(n_states, 1, i + 1)

        # Plot true state
        plt.plot(T_hat, X_true_aligned[i, :], "b-", linewidth=2, label="True State")

        # Plot estimated state
        plt.plot(
            T_hat, X_hat_sequence[i, :], "r--", linewidth=1.5, label="Estimated State"
        )

        plt.xlabel("Time (s)")
        plt.ylabel(state_names[i])
        plt.grid(True)
        plt.legend()

        # Add error metrics for this state dimension
        mse = np.mean((X_true_aligned[i, :] - X_hat_sequence[i, :]) ** 2)
        rmse = np.sqrt(mse)
        plt.title(f"{state_names[i]} (RMSE: {rmse:.4f})")

    plt.tight_layout()


def run_vehicle_simulation_and_save(
    random_seed: int,
    outlier_percent: int,
    num_simulation_steps: int,
    scale_outlier_proc: float,
    scale_outlier_meas: float,
    filter_type: Optional[str] = None,
    filter_kwargs_str: Optional[str] = None,
):
    """
    Runs a simulation of the vehicle model, saves the simulation data to a .pkl file
    in 'results/simulation_data/', and plots the raw simulation trajectories,
    measurements, and noise signals.

    If filter_type is specified, it will also run that filter on the simulation data
    and plot the filtering results.

    The filename will be 'vehicle_p<outlier_percent>_sp<scale_outlier_proc>_sm<scale_outlier_meas>_steps<num_simulation_steps>_seed<random_seed_value>.pkl'.

    Args:
        random_seed: The random seed for NumPy to ensure reproducibility.
        outlier_percent: The percentage of outliers in process and measurement noise.
        num_simulation_steps: The number of simulation steps.
        scale_outlier_proc: Scale factor for process noise outliers.
        scale_outlier_meas: Scale factor for measurement noise outliers.
        filter_type: Optional. The type of filter to use. If None, only the raw
                    simulation is run and saved. If specified, a filter is also run.
        filter_kwargs_str: Optional. JSON string of keyword arguments for the filter constructor. E.g., "{"coef_s": 0.9, "coef_o": 0.05}"
    """
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
    print(f"Using outlier percentage: {outlier_percent}%")
    print(f"Number of simulation steps: {num_simulation_steps}")
    print(f"Scale factor for process noise outliers: {scale_outlier_proc}")
    print(f"Scale factor for measurement noise outliers: {scale_outlier_meas}")

    # 1. Create the vehicle system model
    gamma_friction = 0.05
    time_step = 0.05
    # Calculate simulation_time_final
    simulation_time_final = (num_simulation_steps - 1) * time_step
    print(
        f"Derived total simulation time: {simulation_time_final}s (using dt={time_step})"
    )

    # --- Output Path Configuration ---
    output_dir = os.path.join("results", "simulation_data")
    print(f"Ensuring output directory exists: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Updated filename to include num_simulation_steps and scale factors
    filename = f"vehicle_p{outlier_percent}_sp{int(scale_outlier_proc)}_sm{int(scale_outlier_meas)}_steps{num_simulation_steps}_seed{random_seed}.pkl"
    save_sim_data_path = os.path.join(output_dir, filename)
    print(f"Simulation data will be saved to: {save_sim_data_path}")

    # Storing params for reconstruction, actual model object is not pickled
    vehicle_model_params = {"gamma": gamma_friction, "dt": time_step}
    # We still need an instance for the simulation itself and to get nx, ny
    vehicle_model_instance = vehicle_ss(**vehicle_model_params)
    nx = vehicle_model_instance.nstates
    ny = vehicle_model_instance.noutputs
    dim_w_dyn = 2  # Dimensionality of the dynamic process noise w_dyn

    # 2. Define simulator parameters
    process_noise_cov_w_dyn = np.eye(dim_w_dyn) * 10
    measurement_noise_cov = np.eye(ny) * 5
    dim_w_meas = measurement_noise_cov.shape[0]

    # Convert percentage to probability (0-1 range)
    p_outlier_prob = outlier_percent / 100.0
    p_outlier_proc = p_outlier_prob
    p_outlier_meas = p_outlier_prob

    # 3. Instantiate the Simulator
    sim = Simulator(
        system_model=vehicle_model_instance,
        process_noise_cov=process_noise_cov_w_dyn,
        measurement_noise_cov=measurement_noise_cov,
        p_outlier_process=p_outlier_proc,
        outlier_scale_process=scale_outlier_proc,
        p_outlier_measurement=p_outlier_meas,
        outlier_scale_measurement=scale_outlier_meas,
    )

    # 4. Define simulation run parameters
    initial_state_x0 = np.hstack((np.zeros(2), 5 * np.ones(2)))

    # Initial error covariance (used for saving, as tuning scripts expect it)
    P0_initial = np.eye(nx)

    # 5. Run simulation
    print("Running simulation...")
    T_out, Y_measurements, X_states, W_noise_inputs = sim.simulate(
        x0=initial_state_x0,
        T_final=simulation_time_final,  # Use calculated T_final
        num_steps=num_simulation_steps,  # Use input num_steps
        return_noise_inputs=True,
    )
    U_proc_noise = W_noise_inputs[:dim_w_dyn, :]
    U_meas_noise = W_noise_inputs[dim_w_dyn : (dim_w_dyn + dim_w_meas), :]
    print(
        f"Simulation complete. Data: T={T_out.shape}, Y={Y_measurements.shape}, X={X_states.shape}"
    )

    # 6. Prepare data for saving
    sim_data_for_tuning = {
        "model_type": "vehicle_ss",  # Identifier for the model type
        "model_params": vehicle_model_params,  # Parameters to reconstruct the model
        "cov_input": process_noise_cov_w_dyn,
        "cov_measurement": measurement_noise_cov,
        "T_out": T_out,
        "Y_measurements": Y_measurements,
        "X_true_states": X_states,
        "x0_estimate_filter": initial_state_x0,
        "P0_initial_val": P0_initial,  # Using the P0 defined for this purpose
        "nx_sim_val": nx,
        "ny_sim_val": ny,
        "simulation_time_final": simulation_time_final,  # Save calculated simulation time
        "num_simulation_steps": num_simulation_steps,  # Save num_simulation_steps
    }

    print(f"Attempting to open {save_sim_data_path} for writing...")
    try:
        with open(save_sim_data_path, "wb") as f:
            print(f"File {save_sim_data_path} opened. Attempting to pickle data...")
            pickle.dump(sim_data_for_tuning, f)
            print("Pickling complete. Data should be written to disk.")

        # Verify file creation and size
        if os.path.exists(save_sim_data_path):
            file_size = os.path.getsize(save_sim_data_path)
            print(
                f"Successfully saved data to {save_sim_data_path}. File size: {file_size} bytes."
            )
            if file_size == 0:
                print("Warning: The saved file is empty (0 bytes).")
        else:
            print(f"Error: File {save_sim_data_path} was NOT created.")

    except Exception as e:
        print(f"An error occurred during file writing or pickling: {e}")
        import traceback

        traceback.print_exc()
        return  # Stop execution if saving failed

    # If no filter_type is specified, just plot the raw simulation results and return
    if filter_type is None:
        # 7. Plotting raw simulation results
        print("Plotting raw simulation results...")
        plot_vehicle_simulation_data(
            T_out,
            X_states,
            Y_measurements,
            X_hat_sequence=None,  # No filter estimates to plot
            U_proc_noise=U_proc_noise,
            U_meas_noise=U_meas_noise,
            dim_w_dyn=dim_w_dyn,
            dim_w_meas=dim_w_meas,
        )
        print("Raw simulation data plot displayed.")
        return

    # 7. Setup and Run Filter (only if filter_type is specified)
    print(f"\nSetting up and running {filter_type} filter...")

    filter_instance = None
    filter_display_name = ""

    parsed_filter_kwargs = {}
    if filter_kwargs_str:
        try:
            parsed_filter_kwargs = json.loads(filter_kwargs_str)
            print(f"  Attempting to use custom filter kwargs: {parsed_filter_kwargs}")
        except json.JSONDecodeError as e:
            print(
                f"  Error decoding filter_kwargs JSON: {e}. Using default filter parameters."
            )

    common_filter_args = {
        "system_model": vehicle_model_instance,
        "cov_input": process_noise_cov_w_dyn,
        "cov_measurement": measurement_noise_cov,
    }

    if filter_type == "kalman":
        filter_display_name = "Kalman Filter"
        filter_instance = KalmanFilter(**common_filter_args, **parsed_filter_kwargs)
    elif filter_type == "huber":
        filter_display_name = "Huber Kalman Filter"
        filter_instance = HuberKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "iskf":
        filter_display_name = "Iteratively Saturated Kalman Filter"
        filter_instance = IterSatKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_kalman":
        filter_display_name = "Steady-State Kalman Filter"
        filter_instance = SteadyKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_huber":
        filter_display_name = "Steady-State Huber Kalman Filter"
        filter_instance = SteadyHuberKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_iskf":
        filter_display_name = "Steady-State Iteratively Saturated Kalman Filter"
        filter_instance = SteadyIterSatKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_regularized":
        filter_display_name = "Steady-State Regularized Kalman Filter"
        filter_instance = SteadyRegularizedKalmanFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "wolf":
        filter_display_name = "Weighted Likelihood Filter (WoLF)"
        filter_instance = WeightedLikelihoodFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_one_step_iskf":
        filter_display_name = "Steady One-Step Huber Filter"
        filter_instance = SteadyOneStepIterSatFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_two_step_iskf":
        filter_display_name = "Steady Two-Step Huber Filter"
        filter_instance = SteadyTwoStepIterSatFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    elif filter_type == "steady_three_term_iskf":
        filter_display_name = "Steady Three-Term Huber Filter"
        filter_instance = SteadyThreeStepIterSatFilter(
            **common_filter_args, **parsed_filter_kwargs
        )
    else:
        # This case should ideally be caught by argparse choices
        raise ValueError(
            f"Unknown filter_type: {filter_type}. "
            "Choose from 'kalman', 'huber', 'iskf', "
            "'steady_kalman', 'steady_huber', 'steady_iskf', "
            "'steady_regularized', 'wolf', 'steady_simple_huber', 'steady_two_step_iskf', 'steady_three_term_iskf'."
        )

    print(f"Initialized {filter_display_name}.")
    # Print actual parameters of the filter instance
    if hasattr(filter_instance, "coef_s") and hasattr(filter_instance, "coef_o"):
        print(
            f"  coef_s={getattr(filter_instance, 'coef_s', 'N/A')}, "
            f"coef_o={getattr(filter_instance, 'coef_o', 'N/A')}"
        )
    if hasattr(filter_instance, "scale_covariance_update"):
        print(
            f"  scale_covariance_update={getattr(filter_instance, 'scale_covariance_update', 'N/A')}"
        )

    if hasattr(filter_instance, "num_iters"):
        print(
            f"  num_iters={getattr(filter_instance, 'num_iters', 'N/A')}, "
            f"step_size={getattr(filter_instance, 'step_size', 'N/A')}"
        )
    if hasattr(filter_instance, "mean_update_initialization"):
        print(
            f"  mean_update_initialization='{getattr(filter_instance, 'mean_update_initialization', 'N/A')}'"
        )
    if hasattr(filter_instance, "use_exact_mean_solve"):
        print(
            f"  use_exact_mean_solve={getattr(filter_instance, 'use_exact_mean_solve', 'N/A')}"
        )

    if hasattr(filter_instance, "norm_type"):
        print(f"  norm_type={getattr(filter_instance, 'norm_type', 'N/A')}")

    if (
        hasattr(filter_instance, "coef")
        and hasattr(filter_instance, "weighting")
        and filter_type == "wolf"
    ):
        print(
            f"  WoLF coef={getattr(filter_instance, 'coef', 'N/A')}, "
            f"weighting='{getattr(filter_instance, 'weighting', 'N/A')}'"
        )

    if hasattr(filter_instance, "step_size") and filter_type in [
        "steady_one_step_iskf",
        "steady_two_step_iskf",
    ]:
        print(f"  step_size={getattr(filter_instance, 'step_size', 'N/A')}")
    if (
        hasattr(filter_instance, "coef")
        and not hasattr(filter_instance, "coef_s")
        and filter_type != "wolf"
    ):
        print(f"  coef={getattr(filter_instance, 'coef', 'N/A')}")

    # Process the simulation outputs through the selected filter
    X_hat_sequence = filter_instance.estimate(
        T_out,
        Y_measurements,
        x_initial_estimate=initial_state_x0,
        P_initial=P0_initial,
    )

    # 8. Calculate and display performance metrics
    print("\nCalculating performance metrics...")

    # Get the number of time steps in estimated and true states.
    n_timesteps_hat = X_hat_sequence.shape[1]
    n_timesteps_true = X_states.shape[1]

    x_true_for_metric = None
    x_pred_for_metric = X_hat_sequence

    # Align true states with predicted states for comparison.
    if n_timesteps_hat == n_timesteps_true:
        x_true_for_metric = X_states
        print(
            "  Info: Comparing estimates against all true states (X_hat and X_true have same length)."
        )
    elif n_timesteps_hat == n_timesteps_true - 1:
        x_true_for_metric = X_states[:, 1:]
        print("  Info: Comparing estimates (from k=1) against true states (from k=1).")
    else:
        print(
            f"  Warning: Shape mismatch prevents metric calculation. "
            f"X_states shape: {X_states.shape}, X_hat_sequence shape: {X_hat_sequence.shape}. "
            "Skipping metrics."
        )

    # Proceed with metric calculation if alignment was successful.
    if x_true_for_metric is not None:
        if x_pred_for_metric.shape == x_true_for_metric.shape:
            print("  Performance Metrics:")
            for metric_name, metric_func in METRIC_REGISTRY.items():
                metric_value = metric_func(x_pred_for_metric, x_true_for_metric)
                print(f"    {metric_name.upper()}: {metric_value:.4f}")

            # Add calculation of position-only RMSE (first two dimensions)
            position_rmse = np.sqrt(
                np.mean(
                    np.sum(
                        (x_pred_for_metric[:2, :] - x_true_for_metric[:2, :]) ** 2,
                        axis=0,
                    )
                )
            )
            print(f"    POSITION_RMSE: {position_rmse:.4f}")

            velocity_rmse = np.sqrt(
                np.mean(
                    np.sum(
                        (x_pred_for_metric[2:, :] - x_true_for_metric[2:, :]) ** 2,
                        axis=0,
                    )
                )
            )
            print(f"    VELOCITY_RMSE: {velocity_rmse:.4f}")
        else:
            print(
                f"  Warning: Post-alignment shape mismatch. "
                f"Pred shape: {x_pred_for_metric.shape}, True shape: {x_true_for_metric.shape}. "
                "Skipping metrics."
            )

    # 9. Plotting simulation results with filter estimates
    print(f"\nPlotting simulation results with {filter_display_name} estimates...")

    # Add the new plot for individual state trajectories
    if X_hat_sequence is not None and X_hat_sequence.shape[1] > 0:
        print("Plotting individual state trajectories...")
        plot_state_trajectories(
            T_out,
            X_states,
            X_hat_sequence,
            state_names=["X Position", "Y Position", "X Velocity", "Y Velocity"],
        )

    plot_vehicle_simulation_data(
        T_out,
        X_states,
        Y_measurements,
        X_hat_sequence,
        U_proc_noise,
        U_meas_noise,
        dim_w_dyn,
        dim_w_meas,
    )

    plt.show()

    print(
        f"Simulation and estimation with {filter_display_name} complete. Plots displayed."
    )

    # Print the full path of the generated file
    print(f"\nSimulation data saved to: {os.path.abspath(save_sim_data_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate vehicle model and save data for tuning."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for NumPy to ensure reproducibility of the simulation.",
    )
    parser.add_argument(
        "--outlier_percent",
        type=int,
        default=10,
        help="Percentage of outliers in process and measurement noise.",
    )
    parser.add_argument(
        "--num_simulation_steps",
        type=int,
        default=1000,
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--scale_outlier_proc",
        type=float,
        default=10,
        help="Scale factor for process noise outliers.",
    )
    parser.add_argument(
        "--scale_outlier_meas",
        type=float,
        default=10.0,
        help="Scale factor for measurement noise outliers.",
    )
    parser.add_argument(
        "--filter_kwargs",
        type=str,
        default=None,
        help="JSON string of keyword arguments for the filter constructor. E.g., \"{'coef_s': 0.9, 'coef_o': 0.05}\"",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        choices=[
            "kalman",
            "huber",
            "iskf",
            "steady_kalman",
            "steady_huber",
            "steady_iskf",
            "steady_regularized",
            "wolf",
            "steady_one_step_iskf",
            "steady_two_step_iskf",
            "steady_three_term_iskf",
        ],
        default=None,  # Default to None (no filter)
        help=(
            "Type of filter to run (optional):\n"
            "If not specified, only the raw simulation is run and saved.\n"
            "If specified, the filter is also run and its performance is evaluated."
            " Choices: 'kalman', 'huber', 'iskf', 'steady_kalman', "
            "'steady_huber', 'steady_iskf', 'steady_regularized', 'wolf', "
            "'steady_simple_huber', 'steady_two_step_iskf', 'steady_three_term_iskf'."
        ),
    )
    args = parser.parse_args()

    try:
        run_vehicle_simulation_and_save(
            random_seed=args.random_seed,
            outlier_percent=args.outlier_percent,
            num_simulation_steps=args.num_simulation_steps,
            scale_outlier_proc=args.scale_outlier_proc,
            scale_outlier_meas=args.scale_outlier_meas,
            filter_type=args.filter_type,
            filter_kwargs_str=args.filter_kwargs,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
