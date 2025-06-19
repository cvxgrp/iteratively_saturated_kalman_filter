"""
Script to simulate a cascaded CSTR model and save the simulation data for later use
by hyperparameter tuning scripts. It also plots the raw simulation results.

If a filter_type is specified, it will also run that filter on the simulation data
and plot the filtering results.

Output files are saved in 'results/simulation_data/' with a filename
pattern 'cstr_n<N_REACTORS>_p<OUTLIER_PERCENT>_sp<SCALE_OUTLIER_PROC>_sm<SCALE_OUTLIER_MEAS>_steps<SIMULATION_STEPS>_seed<SEED>.pkl'.
"""

import argparse
import os
import pickle
from typing import Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Project-specific imports
from iskf.models.cstr import cascaded_cstr_ss  # Changed from vehicle
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


def plot_cstr_simulation_data(
    T_out: np.ndarray,
    X_states: np.ndarray,
    Y_measurements: np.ndarray,
    X_hat_sequence: Optional[np.ndarray],
    U_proc_noise: np.ndarray,
    U_meas_noise: np.ndarray,
    dim_w_dyn: int,
    dim_w_meas: int,
    n_reactors: int,
):
    """
    Plots the results of the CSTR simulation and filter estimation.

    Args:
        T_out: Time vector.
        X_states: True state trajectory [C_A1, T1, C_A2, T2, ...].
        Y_measurements: Noisy measurements [T1_m, C_A1_m, T2_m, C_A2_m, ...].
        X_hat_sequence: Estimated state trajectory from the filter.
        U_proc_noise: Generated process noise sequence.
        U_meas_noise: Generated measurement noise sequence.
        dim_w_dyn: Dimension of the dynamic process noise.
        dim_w_meas: Dimension of the measurement noise.
        n_reactors: Number of reactors in the cascade.
    """
    # Plot 1: Process Noise Components
    if dim_w_dyn > 0 and U_proc_noise.shape[1] > 0:
        plt.figure(figsize=(12, max(1.2 * dim_w_dyn, 3)))  # Further reduced height
        for i in range(dim_w_dyn):
            plt.subplot(dim_w_dyn, 1, i + 1)
            plt.plot(
                T_out,
                U_proc_noise[i, :],
                label=f"Process Noise w_dyn[{i}] (e.g., T_ci, C_Af disturbances)",
            )
            plt.ylabel(f"w_dyn[{i}]")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Time (s)")
        plt.suptitle("Process Noise Components")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot 2: Measurement Noise Components
    if dim_w_meas > 0 and U_meas_noise.shape[1] > 0:
        plt.figure(figsize=(12, max(1.2 * dim_w_meas, 3)))  # Further reduced height
        for i in range(dim_w_meas):
            plt.subplot(dim_w_meas, 1, i + 1)
            # Output y is [T1, T2, ... Tn]
            output_label = f"T{i+1}"
            plt.plot(
                T_out,
                U_meas_noise[i, :],
                label=f"Measurement Noise v_meas[{i}] on {output_label}",
            )
            plt.ylabel(f"v_meas[{i}]")
            plt.grid(True)
            plt.legend()
        plt.xlabel("Time (s)")
        plt.suptitle("Measurement Noise Components")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot 3: CSTR States (Concentrations and Temperatures)
    # num_states_to_plot = 2 * n_reactors
    # Adjusted figsize for potentially many subplots
    # Each reactor has 2 states (C_A, T)
    # We create one figure and put 2*n_reactors subplots on it.
    # If n_reactors = 1, 2 plots. If n_reactors = 2, 4 plots. If n_reactors = 3, 6 plots.
    # Let's arrange them in N rows, 2 cols if N > 1, else 2 rows, 1 col.

    num_state_plots_rows = n_reactors
    num_state_plots_cols = 2  # C_A and T for each reactor

    plt.figure(figsize=(12, 1.5 * num_state_plots_rows))  # Further reduced height

    for i in range(n_reactors):
        # Plot Concentration C_A(i+1)
        # State index for C_A(i+1) is 2*i
        plt.subplot(num_state_plots_rows, num_state_plots_cols, 2 * i + 1)
        plt.plot(T_out, X_states[2 * i, :], label=f"True C_A{i+1}")
        if X_hat_sequence is not None and X_hat_sequence.shape[1] > 0:
            T_hat = T_out[1:] if X_hat_sequence.shape[1] == len(T_out) - 1 else T_out
            plt.plot(
                T_hat,
                X_hat_sequence[2 * i, :],
                "--",
                label=f"Estimated C_A{i+1}",
                alpha=0.8,
            )
        plt.ylabel(f"C_A{i+1} (mol/L)")
        plt.xlabel("Time (s)")
        plt.title(f"Concentration in Reactor {i+1}")
        plt.grid(True)
        plt.legend()

        # Plot Temperature T(i+1)
        # State index for T(i+1) is 2*i + 1
        plt.subplot(num_state_plots_rows, num_state_plots_cols, 2 * i + 2)
        plt.plot(T_out, X_states[2 * i + 1, :], label=f"True T{i+1}")
        if X_hat_sequence is not None and X_hat_sequence.shape[1] > 0:
            T_hat = T_out[1:] if X_hat_sequence.shape[1] == len(T_out) - 1 else T_out
            plt.plot(
                T_hat,
                X_hat_sequence[2 * i + 1, :],
                "--",
                label=f"Estimated T{i+1}",
                alpha=0.8,
            )
        plt.ylabel(f"T{i+1} (K)")  # Assuming Kelvin, adjust if different
        plt.xlabel("Time (s)")
        plt.title(f"Temperature in Reactor {i+1}")
        plt.grid(True)
        plt.legend()

    plt.suptitle("CSTR State Trajectories (True vs. Estimated)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.show()


def run_cstr_simulation_and_save(
    random_seed: int,
    outlier_percent: int,
    num_simulation_steps: int,  # Changed from simulation_time to num_simulation_steps
    scale_outlier_proc: float,
    scale_outlier_meas: float,
    filter_type: Optional[str] = None,
    filter_kwargs_str: Optional[str] = None,
):
    """
    Runs a simulation of the CSTR model, saves the data, and plots results.
    Uses a fixed number of reactors (N_REACTORS_SIM = 4).
    Incorporates offsets for states (Concentration, Temperature) and measurements.
    """
    np.random.seed(random_seed)
    N_REACTORS_SIM = 3

    # Define offsets
    TEMP_OFFSET = 373  # Kelvin
    CONC_OFFSET = 2  # kmol/m^3 # Concentration offset still used for states

    print(f"Using random seed: {random_seed}")
    print(f"Using outlier percentage: {outlier_percent}%")
    print(f"Number of reactors (fixed): {N_REACTORS_SIM}")
    print(f"Number of simulation steps: {num_simulation_steps}")  # Updated print
    print(f"Scale factor for process noise outliers: {scale_outlier_proc}")
    print(f"Scale factor for measurement noise outliers: {scale_outlier_meas}")
    print(
        f"Temperature offset: {TEMP_OFFSET} K, Concentration offset: {CONC_OFFSET} kmol/m^3"
    )

    # 1. Create the CSTR system model
    time_step = 0.05  # Default dt for cascaded_cstr_ss
    # Explicitly pass n to cascaded_cstr_ss to ensure consistency
    cstr_model_params = {"n": N_REACTORS_SIM, "dt": time_step}
    cstr_model_instance = cascaded_cstr_ss(**cstr_model_params)

    # Calculate simulation_time_final from num_simulation_steps and time_step
    simulation_time_final = (
        num_simulation_steps - 1
    ) * time_step  # T_final = (N_steps-1)*dt
    print(f"Derived total simulation time: {simulation_time_final}s")

    # --- Output Path Configuration ---
    output_dir = os.path.join("results", "simulation_data")
    print(f"Ensuring output directory exists: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)
    # Updated filename to include num_simulation_steps and scale factors
    filename = f"cstr_n{N_REACTORS_SIM}_p{outlier_percent}_sp{int(scale_outlier_proc)}_sm{int(scale_outlier_meas)}_steps{num_simulation_steps}_seed{random_seed}.pkl"
    save_sim_data_path = os.path.join(output_dir, filename)
    print(f"Simulation data will be saved to: {save_sim_data_path}")

    nx = cstr_model_instance.nstates  # Should be 2 * N_REACTORS_SIM
    ny = cstr_model_instance.noutputs  # Should be N_REACTORS_SIM (only temperatures)
    print(f"System dimensions: nx={nx}, ny={ny}")

    # Create offset vectors
    # States are [C_A1, T1, C_A2, T2, ...]
    state_offset_vector = np.zeros(nx)
    for i in range(N_REACTORS_SIM):
        state_offset_vector[2 * i] = CONC_OFFSET  # C_Ai
        state_offset_vector[2 * i + 1] = TEMP_OFFSET  # Ti

    # Outputs are [T1_m, T2_m, ..., Tn_m]
    measurement_offset_vector = np.full(ny, TEMP_OFFSET)  # All temperatures

    # Physical process noise dimension (w_dyn_physical affecting [Tci, CAfi] for each reactor)
    # As per cstr.py, B_process_noise_big has 2*n columns.
    dim_w_dyn_physical = 2 * N_REACTORS_SIM
    # Physical measurement noise dimension (v_physical affecting [Tmi])
    # As per cstr.py, D_meas_noise has n columns.
    dim_w_meas_physical = N_REACTORS_SIM  # ny

    print(
        f"Physical process noise dimension (dim_w_dyn_physical): {dim_w_dyn_physical}"
    )
    print(
        f"Physical measurement noise dimension (dim_w_meas_physical): {dim_w_meas_physical}"
    )

    # 2. Define simulator parameters (using physical noise dimensions)
    # Covariance for the physical process noise (size: dim_w_dyn_physical x dim_w_dyn_physical)
    process_noise_cov_physical = np.eye(dim_w_dyn_physical) * 1e-1
    # Covariance for the physical measurement noise (size: dim_w_meas_physical x dim_w_meas_physical)
    measurement_noise_cov_physical = np.eye(dim_w_meas_physical) * 1e0

    p_outlier_prob = outlier_percent / 100.0
    p_outlier_proc = p_outlier_prob
    p_outlier_meas = p_outlier_prob

    # 3. Instantiate the Simulator
    sim = Simulator(
        system_model=cstr_model_instance,
        process_noise_cov=process_noise_cov_physical,  # Use physical process noise cov
        measurement_noise_cov=measurement_noise_cov_physical,  # Use physical measurement noise cov
        p_outlier_process=p_outlier_proc,
        outlier_scale_process=scale_outlier_proc,
        p_outlier_measurement=p_outlier_meas,
        outlier_scale_measurement=scale_outlier_meas,
    )

    # 4. Define simulation run parameters
    # Initial state for the linearized model: small random perturbations around zero.
    initial_state_x0 = np.zeros(nx)
    P0_initial = np.eye(nx) * 1.0

    # 5. Run simulation (produces linearized states and measurements)
    print("Running CSTR simulation (linearized model)...")
    T_out, Y_measurements_lin, X_states_lin, W_noise_inputs = sim.simulate(
        x0=initial_state_x0,  # Use linearized initial state
        T_final=simulation_time_final,  # Use calculated T_final
        num_steps=num_simulation_steps,  # Use input num_steps
        return_noise_inputs=True,
    )
    # W_noise_inputs contains [physical_process_noise_sequence, physical_measurement_noise_sequence]
    # Extract physical process noise (e.g., disturbances on Tc, CAf for each reactor)
    U_proc_noise = W_noise_inputs[:dim_w_dyn_physical, :]
    # Extract physical measurement noise (on measured temperatures)
    U_meas_noise = W_noise_inputs[
        dim_w_dyn_physical : (dim_w_dyn_physical + dim_w_meas_physical), :
    ]
    print(
        f"Linear simulation complete. Data: T={T_out.shape}, Y_lin={Y_measurements_lin.shape}, X_lin={X_states_lin.shape}"
    )
    print(
        f"  U_proc_noise shape: {U_proc_noise.shape}, U_meas_noise shape: {U_meas_noise.shape}"
    )

    # Convert to absolute states and measurements by adding offsets
    X_states_abs = X_states_lin + state_offset_vector.reshape(-1, 1)
    Y_measurements_abs = Y_measurements_lin + measurement_offset_vector.reshape(-1, 1)
    print(
        f"Absolute states and measurements generated: X_abs={X_states_abs.shape}, Y_abs={Y_measurements_abs.shape}"
    )

    # 6. Prepare data for saving
    # For tune_filter.py to work without modification, we save linearized data
    # under the keys it expects.
    sim_data_for_tuning = {
        "model_type": "cascaded_cstr_ss",
        "model_params": cstr_model_params,
        "cov_input": process_noise_cov_physical,  # Covariance of physical w_dyn
        "cov_measurement": measurement_noise_cov_physical,  # Covariance of physical w_meas
        "T_out": T_out,
        "Y_measurements": Y_measurements_lin,  # Save LINEARIZED measurements for tuning
        "X_true_states": X_states_lin,  # Save LINEARIZED true states for tuning
        "x0_estimate_filter": initial_state_x0,  # Key expected by tune_filter.py, using the correct linear value
        "P0_initial_val": P0_initial,
        "state_offset_vector": state_offset_vector,  # Save for completeness
        "measurement_offset_vector": measurement_offset_vector,  # Save for completeness
        "nx_sim_val": nx,
        "ny_sim_val": ny,
        "n_reactors_sim": N_REACTORS_SIM,  # Store n_reactors used
        "dim_w_dyn_physical": dim_w_dyn_physical,  # Store physical process noise dim
        "dim_w_meas_physical": dim_w_meas_physical,  # Store physical measurement noise dim
        "simulation_time_final": simulation_time_final,  # Save calculated simulation time
        "num_simulation_steps": num_simulation_steps,  # Save num_simulation_steps
    }

    print(f"Attempting to open {save_sim_data_path} for writing...")
    try:
        with open(save_sim_data_path, "wb") as f:
            pickle.dump(sim_data_for_tuning, f)
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
        return

    X_hat_sequence_filter_abs = None  # Initialize for absolute estimates
    if filter_type is None:
        print("Plotting raw CSTR simulation results (absolute values)...")
        plot_cstr_simulation_data(
            T_out,
            X_states_abs,  # Plot absolute states
            Y_measurements_abs,  # Plot absolute measurements
            X_hat_sequence=None,  # No filter estimate yet
            U_proc_noise=U_proc_noise,
            U_meas_noise=U_meas_noise,
            dim_w_dyn=dim_w_dyn_physical,  # Use physical dim for plotting
            dim_w_meas=dim_w_meas_physical,  # Use physical dim for plotting
            n_reactors=N_REACTORS_SIM,  # Pass the fixed N_REACTORS_SIM
        )
        print("Raw CSTR simulation data plot displayed.")
        return

    # 7. Setup and Run Filter
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
            # Optionally, you might want to raise an error or exit
            # For now, it will proceed with filter's internal defaults or no kwargs

    # Common arguments for all filter constructors
    common_filter_args = {
        "system_model": cstr_model_instance,  # Linearized model (control.StateSpace)
        "cov_input": process_noise_cov_physical,  # Covariance for noise vector sys.B acts upon
        "cov_measurement": measurement_noise_cov_physical,  # Covariance for noise vector sys.D acts upon
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
        raise ValueError(f"Unknown filter_type: {filter_type}")

    print(f"Initialized {filter_display_name}.")
    # These print statements are still useful to show the *actual* parameters of the filter instance
    if hasattr(filter_instance, "coef_s") and hasattr(filter_instance, "coef_o"):
        print(
            f"  coef_s={getattr(filter_instance, 'coef_s', 'N/A')}, coef_o={getattr(filter_instance, 'coef_o', 'N/A')}"
        )
    if hasattr(filter_instance, "num_iters"):  # For ISKF
        print(
            f"  num_iters={getattr(filter_instance, 'num_iters', 'N/A')}, "
            f"step_size={getattr(filter_instance, 'step_size', 'N/A')}"
        )
    if hasattr(filter_instance, "coef") and not hasattr(filter_instance, "coef_s"):
        print(f"  coef={getattr(filter_instance, 'coef', 'N/A')}")
    if hasattr(filter_instance, "step_size"):
        print(f"  step_size={getattr(filter_instance, 'step_size', 'N/A')}")
    # Add more getattr checks for other relevant parameters if needed for other filters

    # Prepare inputs for the filter (linearized domain)
    Y_measurements_for_filter = Y_measurements_abs - measurement_offset_vector.reshape(
        -1, 1
    )
    x_initial_estimate_for_filter = initial_state_x0  # Use linear initial estimate

    print(f"Running {filter_display_name} on linearized measurements...")
    X_hat_sequence_lin_filter = filter_instance.estimate(
        T_out,
        Y_measurements_for_filter,  # Provide linearized measurements
        x_initial_estimate=x_initial_estimate_for_filter,  # Provide linear initial estimate
        P_initial=P0_initial,
    )
    print(
        f"Filter estimation (linearized) complete. X_hat_lin shape: {X_hat_sequence_lin_filter.shape}"
    )

    # Convert filter estimates to absolute values
    X_hat_sequence_filter_abs = X_hat_sequence_lin_filter + state_offset_vector.reshape(
        -1, 1
    )
    print(
        f"Filter estimates converted to absolute. X_hat_abs shape: {X_hat_sequence_filter_abs.shape}"
    )

    # 8. Calculate and display performance metrics (using absolute values)
    print("\nCalculating performance metrics (on absolute values)...")
    n_timesteps_hat = X_hat_sequence_filter_abs.shape[1]
    n_timesteps_true = X_states_abs.shape[1]  # Use absolute true states
    x_true_for_metric, x_pred_for_metric = (
        None,
        X_hat_sequence_filter_abs,
    )  # Use absolute predictions

    if n_timesteps_hat == n_timesteps_true:
        x_true_for_metric = X_states_abs  # Use absolute true states
    elif n_timesteps_hat == n_timesteps_true - 1:
        x_true_for_metric = X_states_abs[:, 1:]  # Use absolute true states
    else:
        print(
            f"  Warning: Shape mismatch for metrics. X_true_abs: {X_states_abs.shape}, X_hat_abs: {X_hat_sequence_filter_abs.shape}"
        )

    if (
        x_true_for_metric is not None
        and x_pred_for_metric.shape == x_true_for_metric.shape
    ):
        print("  Performance Metrics:")
        for metric_name, metric_func in METRIC_REGISTRY.items():
            metric_value = metric_func(x_pred_for_metric, x_true_for_metric)
            print(f"    {metric_name.upper()}: {metric_value:.4f}")
    else:
        if x_true_for_metric is not None:  # check if it was set
            print(
                f"  Warning: Post-alignment shape mismatch. Pred: {x_pred_for_metric.shape}, True: {x_true_for_metric.shape}"
            )
        print("  Skipping metrics calculation.")

    # 9. Plotting simulation results with filter estimates
    print(
        f"\nPlotting CSTR simulation results with {filter_display_name} estimates (absolute values)..."
    )
    plot_cstr_simulation_data(
        T_out,
        X_states_abs,  # Plot absolute true states
        Y_measurements_abs,  # Plot absolute measurements
        X_hat_sequence_filter_abs,  # Plot absolute estimates
        U_proc_noise,
        U_meas_noise,
        dim_w_dyn_physical,  # Use physical dim for plotting
        dim_w_meas_physical,  # Use physical dim for plotting
        N_REACTORS_SIM,  # Pass the fixed N_REACTORS_SIM
    )
    print(
        f"Simulation and estimation with {filter_display_name} complete. Plot displayed."
    )

    # Print the full path of the generated file
    print(f"\nSimulation data saved to: {os.path.abspath(save_sim_data_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate cascaded CSTR model and save data for tuning."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for NumPy. Default: 0",
    )
    parser.add_argument(
        "--outlier_percent",
        type=int,
        default=10,
        help="Percentage of outliers in noise. Default: 10",
    )
    parser.add_argument(
        "--num_simulation_steps",  # Changed from --simulation_time
        type=int,  # Changed from float to int
        default=1000,  # Changed default to 1000 steps
        help="Number of simulation steps. Default: 1000",  # Updated help
    )
    parser.add_argument(
        "--scale_outlier_proc",
        type=float,
        default=10,
        help="Scale factor for process noise outliers. Default: 100.0",
    )
    parser.add_argument(
        "--scale_outlier_meas",
        type=float,
        default=10,
        help="Scale factor for measurement noise outliers. Default: 10000.0",
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
            None,
        ],  # Allow None
        default=None,
        help="Type of filter to run (optional).",
    )
    args = parser.parse_args()

    # Convert "None" string from choices to Python None
    selected_filter_type = None if args.filter_type == "None" else args.filter_type

    try:
        run_cstr_simulation_and_save(
            random_seed=args.random_seed,
            outlier_percent=args.outlier_percent,
            num_simulation_steps=args.num_simulation_steps,  # Pass num_simulation_steps
            scale_outlier_proc=args.scale_outlier_proc,
            scale_outlier_meas=args.scale_outlier_meas,
            filter_type=selected_filter_type,
            filter_kwargs_str=args.filter_kwargs,
        )
    except Exception as e:
        print(f"An error occurred in main execution: {e}")
        import traceback

        traceback.print_exc()
