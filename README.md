# Iteratively Saturated Kalman Filter

A Python implementation of the iteratively saturated Kalman filter (ISKF), for state
estimation in the presence of outliers in both the measurements and dynamics.
This code accompanies the paper: https://stanford.edu/~boyd/papers/iskf.html

The ISKF is about as efficient as the standard Kalman filter (KF), and the
steady-state ISKF is about as efficient as the steady-state KF (just
matrix-vector multiplies).

To install dependencies, run:

```bash
pip install -r requirements.txt
```

Optionally, create a virtual environment first.

## Filter Implementations

The `iskf.filters` module provides various robust Kalman filter implementations:

### Basic Filters

- **`KalmanFilter`**: Standard Kalman filter
- **`SteadyKalmanFilter`**: Steady-state Kalman filter

### Huber-based Robust Filters

- **`HuberKalmanFilter`**: Huber-based robust Kalman filter
- **`SteadyHuberKalmanFilter`**: Steady-state Huber Kalman filter
- **`IterSatKalmanFilter`**: Iteratively saturated Kalman filter
- **`SteadyIterSatKalmanFilter`**: Steady-state version of ISKF

### Iteratively Saturated Kalman Filters (ISKF)

- **`SteadyOneStepIterSatFilter`**: Single-step iterative filter
- **`SteadyTwoStepIterSatFilter`**: Two-step iterative filter (ISKF with $\tilde k=2$)
- **`SteadyThreeStepIterSatFilter`**: Three-term Huber filter

### Other Robust Filters

- **`SteadyRegularizedKalmanFilter`**: Regularized Kalman filter
- **`WeightedLikelihoodFilter`**: Weighted likelihood filter (WoLF)

### Filter Usage

All filters inherit from `BaseFilter` and share a common interface:

```python
from iskf.filters import SteadyTwoStepIterSatFilter
from iskf.models.vehicle import vehicle_ss

# Create system model
system_model = vehicle_ss(gamma=0.05, dt=0.05)

# Initialize filter
filter_instance = SteadyTwoStepIterSatFilter(
    system_model=system_model,
    cov_input=process_noise_cov,
    cov_measurement=measurement_noise_cov,
    coef_s=1.345,  # Huber parameter for state
    coef_o=1.345,  # Huber parameter for observation
    step_size=1.0  # Step size for iterations
)

# Run filter
estimates = filter_instance.estimate(
    T_out=time_vector,
    Y_out=measurements,
    x_initial_estimate=initial_state,
    P_initial=initial_covariance
)
```

## System Models

The `iskf.models` module provides system models for simulation and filtering:

### Vehicle Model (`vehicle_ss`)

A 2D vehicle model with position and velocity states.

```python
from iskf.models.vehicle import vehicle_ss

# Create vehicle model
model = vehicle_ss(gamma=0.05, dt=0.05)
# States: [x_pos, y_pos, x_vel, y_vel]
# Outputs: [x_pos, y_pos] (position measurements)
```

### CSTR Model (`cascaded_cstr_ss`)

A cascaded Continuously Stirred Tank Reactor model.

```python
from iskf.models.cstr import cascaded_cstr_ss

# Create CSTR model with n reactors
model = cascaded_cstr_ss(n=3, dt=0.05)
# States: [C_A1, T1, C_A2, T2, C_A3, T3] (concentrations and temperatures)
# Outputs: [T1, T2, T3] (temperature measurements)
```

## Simulation Scripts

### Vehicle Simulation (`simulate_vehicle.py`)

Simulates a 2D vehicle model and saves simulation data for hyperparameter tuning.

```bash
# Generate simulation data only
python simulate_vehicle.py --random_seed 42 --outlier_percent 10 --num_simulation_steps 1000

# Generate data and run a filter
python simulate_vehicle.py --random_seed 42 --outlier_percent 10 --filter_type steady_two_step_iskf
```

**Key Arguments:**

- `--random_seed`: Random seed for reproducibility (default: 42)
- `--outlier_percent`: Percentage of outliers in noise (default: 10)
- `--num_simulation_steps`: Number of simulation steps (default: 1000)
- `--filter_type`: Optional filter to run immediately
- `--filter_kwargs`: JSON string of filter parameters

**Output:** Saves data to `results/simulation_data/vehicle_p{outlier_percent}_sp{scale_proc}_sm{scale_meas}_steps{steps}_seed{seed}.pkl`

### CSTR Simulation (`simulate_cstr.py`)

Simulates a cascaded CSTR model with 3 reactors.

```bash
# Generate CSTR simulation data
python simulate_cstr.py --random_seed 0 --outlier_percent 15 --num_simulation_steps 1000

# With filter
python simulate_cstr.py --outlier_percent 10 --filter_type steady_iskf
```

**Output:** Saves data to `results/simulation_data/cstr_n{n_reactors}_p{outlier_percent}_sp{scale_proc}_sm{scale_meas}_steps{steps}_seed{seed}.pkl`

## Hyperparameter Tuning

### Filter Parameter Tuning (`tune_filter.py`)

Performs grid search over filter hyperparameters using simulation data. **It is recommended to use separate simulation files for fitting and testing to avoid overfitting.**

```bash
# Tune ISKF (k̃=2) parameters (with separate fit and test data)
python tune_filter.py --filter_type steady_two_step_iskf \
    --sim_data_path results/simulation_data/vehicle_p10_sp10_sm10_steps1000_seed42.pkl \
    --test_data_path results/simulation_data/vehicle_p10_sp10_sm10_steps1000_seed0.pkl \
    --metric rmse --optimistic

# Tune Huber filter (with separate fit and test data)
python tune_filter.py --filter_type steady_huber \
    --sim_data_path results/simulation_data/cstr_n3_p10_sp10_sm10_steps1000_seed0.pkl \
    --test_data_path results/simulation_data/cstr_n3_p10_sp10_sm10_steps1000_seed1.pkl
```

**Key Arguments:**

- `--filter_type`: Filter to tune (`steady_two_step_iskf`, `steady_huber`, `wolf`, etc.)
- `--sim_data_path`: Path to simulation data for fitting (required)
- `--test_data_path`: Path to separate test data for evaluation (recommended)
- `--metric`: Evaluation metric (`rmse`, `mne`, `rmedse`, etc.)
- `--optimistic`: Use true states for evaluation (vs. predicted measurements)
- `--sweep_resolution`: Grid search resolution (default: 15)

**Output:** Saves results to `results/parameter_search_data/{filter_type}_{data_info}_{metric}_results.pkl`

> **Note:**
> For robust evaluation, always use different random seeds for `--sim_data_path` and `--test_data_path` to ensure the filter is not tuned and tested on the same data.

### Iteration Count Tuning (`tune_num_iters.py`)

Specifically tunes the number of iterations for iterative filters.

```bash
# Tune iteration count for ISKF
python tune_num_iters.py \
    --sim_data_path results/simulation_data/vehicle_p10_sp10_sm10_steps1000_seed42.pkl \
    --metric rmse --optimistic
```

**Output:** Saves results to `results/parameter_search_data/steady_iskf_iter_count_{data_info}_{metric}_results.pkl`

## Visualization Tools

### Vehicle Results Plotting (`plot_vehicle_results.py`)

Visualizes filter performance on vehicle trajectory data.

```bash
# Plot trajectory with tuned filter
python plot_vehicle_results.py --tune_filter_results \
    results/parameter_search_data/steady_two_step_iskf_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl

# Plot iteration sweep results
python plot_vehicle_results.py --sweep_iters_results \
    results/parameter_search_data/steady_iskf_iter_count_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl
```

**Features:**

- Vehicle trajectory plots showing true path, measurements, and filter estimates
- State trajectory plots with error analysis
- Iteration performance plots

### CSTR Results Plotting (`plot_cstr_results.py`)

Visualizes filter performance on CSTR data.

```bash
# Plot CSTR results
python plot_cstr_results.py --tune_filter_results \
    results/parameter_search_data/steady_iskf_cstr_n3_p10_sp10_sm10_steps1000_seed0_optimistic_rmse_results.pkl
```

**Features:**

- CSTR state trajectories (concentrations and temperatures)
- State error plots
- Comparative performance visualization

## Usage Examples

### Basic Filter Usage

```python
import numpy as np
from iskf.models.vehicle import vehicle_ss
from iskf.filters import SteadyTwoStepIterSatFilter
from iskf.simulator import Simulator

# 1. Create system model
model = vehicle_ss(gamma=0.05, dt=0.05)

# 2. Setup simulation with outlier parameters
sim = Simulator(
    system_model=model,
    process_noise_cov=np.eye(2) * 10,
    measurement_noise_cov=np.eye(2) * 5,
    p_outlier_process=0.1,
    outlier_scale_process=10.0,
    p_outlier_measurement=0.1,
    outlier_scale_measurement=10.0
)

# 3. Generate data
T_out, Y_meas, X_true, _ = sim.simulate(
    x0=np.array([0, 0, 5, 5]),
    T_final=(1000 - 1) * 0.05,  # Ensure T_final is a multiple of dt
    num_steps=1000,
    return_noise_inputs=True
)

# 4. Create and run filter
filter_instance = SteadyTwoStepIterSatFilter(
    system_model=model,
    cov_input=np.eye(2) * 10,
    cov_measurement=np.eye(2) * 5,
    coef_s=1.345,
    coef_o=1.345,
    step_size=1.0
)

X_est = filter_instance.estimate(
    T_out, Y_meas,
    x_initial_estimate=np.array([0, 0, 5, 5]),
    P_initial=np.eye(4)
)

# 5. Evaluate performance
from iskf.metrics import rmse
error = rmse(X_est, X_true[:, 1:])  # Align dimensions
print(f"RMSE: {error:.4f}")
```

### Available Metrics

The `iskf.metrics` module provides several evaluation metrics:

- **`rmse`**: Root Mean Squared Error
- **`rmedse`**: Root Median Squared Error
- **`mne`**: Mean Norm Error
- **`median_ne`**: Median Norm Error
- **`max_ne`**: Maximum Norm Error

## Workflow Overview

The typical experimental workflow consists of three main steps:

### 1. Generate Simulation Data

```bash
# Vehicle model
python simulate_vehicle.py --outlier_percent 10 --random_seed 42

# CSTR model
python simulate_cstr.py --outlier_percent 10 --random_seed 0
```

### 2. Tune Filter Parameters

```bash
# Tune ISKF parameters
python tune_filter.py --filter_type steady_two_step_iskf \
    --sim_data_path results/simulation_data/vehicle_p10_sp10_sm10_steps1000_seed42.pkl

# Tune iteration count
python tune_num_iters.py \
    --sim_data_path results/simulation_data/vehicle_p10_sp10_sm10_steps1000_seed42.pkl
```

### 3. Visualize Results

```bash
# Plot tuned filter performance
python plot_vehicle_results.py --tune_filter_results \
    results/parameter_search_data/steady_two_step_iskf_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl

# Plot iteration analysis
python plot_vehicle_results.py --sweep_iters_results \
    results/parameter_search_data/steady_iskf_iter_count_vehicle_p10_sp10_sm10_steps1000_seed42_realistic_rmse_results.pkl
```
