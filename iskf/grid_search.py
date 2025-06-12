"""
Framework for performing grid search over filter hyperparameters.

This module provides utilities to systematically evaluate different sets of
hyperparameters for various Kalman filter implementations (subclasses of BaseFilter)
by simulating a system, running the filter, and calculating a performance metric.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from frozendict import frozendict

from iskf.metrics import METRIC_REGISTRY
from iskf.filters.base_filter import BaseFilter

# Setup logging
logger = logging.getLogger(__name__)


def _run_single_filter_evaluation(
    base_filter_for_cloning: BaseFilter,  # The initial filter instance for cloning
    filter_hyperparams: Dict[str, Any],
    T_out: np.ndarray,
    Y_measurements: np.ndarray,
    X_states: Optional[np.ndarray],  # True states for metric calculation, now optional
    initial_state_x0_for_filter: np.ndarray,  # For filter's initial estimate
    filter_init_P0: np.ndarray,
    metric_name: str,
) -> float:
    """
    Runs a single filter instance (cloned and configured) with one set of hyperparameters
    on pre-simulated data.

    Args:
        base_filter_for_cloning: The initial BaseFilter instance to be cloned and configured.
        filter_hyperparams: A dictionary of hyperparameters for the new filter instance.
        T_out: Time vector from simulation.
        Y_measurements: Measurement sequence from simulation.
        X_states: True state trajectories from simulation. If None, metrics will be calculated
                 using Y_measurements instead of X_states.
        initial_state_x0_for_filter: Initial state estimate for the filter.
        filter_init_P0: Initial error covariance matrix P0 for the filter.
        metric_name: Name of the metric (from METRIC_REGISTRY) to calculate.

    Returns:
        The calculated performance metric score (float).
    """
    # 1. Create a new filter instance with the specified hyperparameters
    # The new_with_kwargs_like method is expected to copy system_model, Q_covariance,
    # R_covariance from the base_filter_for_cloning and apply new kwargs.
    filter_instance = base_filter_for_cloning.new_with_kwargs_like(**filter_hyperparams)

    # 2. Run Filter based on whether X_states is provided or not
    try:
        if X_states is not None:
            X_pred = filter_instance.estimate(
                T_out,
                Y_measurements,
                x_initial_estimate=initial_state_x0_for_filter,
                P_initial=filter_init_P0,
            )
            metric_func = METRIC_REGISTRY[metric_name]
            score = metric_func(X_pred, X_states[:, 1:])
        else:
            Y_pred = filter_instance.predict_measurements(
                T_out,
                Y_measurements,
                x_initial_estimate=initial_state_x0_for_filter,
                P_initial=filter_init_P0,
            )
            metric_func = METRIC_REGISTRY[metric_name]
            score = metric_func(Y_pred, Y_measurements[:, 1:])
    except Exception as e:
        logger.error(f"Error running filter: {e}")
        score = np.inf

    return score


def grid_search_filter_hyperparams(
    initial_filter_instance: BaseFilter,
    param_grid: List[Dict[str, Any]],
    T_out: np.ndarray,  # Time vector from pre-run simulation
    Y_measurements: np.ndarray,  # Measurement sequence from pre-run simulation
    X_states: Optional[
        np.ndarray
    ] = None,  # True states from pre-run simulation (optional)
    initial_state_x0_for_filter: np.ndarray = None,  # Initial state estimate for the filter
    filter_init_P0: np.ndarray = None,  # Initial error covariance P0 for the filter
    metric_name: str = "rmse",
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], float, Dict[frozendict, float]]:
    """
    Performs a grid search over specified hyperparameters for a given initial filter instance.
    The initial filter provides the base system model and noise characteristics.
    Simulation data (measurements, and optionally true states) must be provided as input.

    Args:
        initial_filter_instance: An instantiated filter (e.g., HuberKalmanFilter)
                                 which will be used as a template. Its `new_with_kwargs_like`
                                 method will be called with each parameter set from param_grid.
        param_grid: A list of dictionaries, where each dict contains one
                    combination of hyperparameters to test.
        T_out: Time vector from a simulation.
        Y_measurements: Measurement sequence from a simulation.
        X_states: True state trajectories from a simulation, for metric calculation.
                 If None, metrics will be computed comparing predicted measurements
                 with actual measurements.
        initial_state_x0_for_filter: Initial state estimate for each filter run.
        filter_init_P0: Initial error covariance P0 for each filter run.
        metric_name: The metric to optimize (e.g., "rmse", "rmedse").
        n_jobs: Number of parallel jobs for joblib.Parallel (-1 uses all CPUs).

    Returns:
        A tuple containing:
        - best_params (Dict[str, Any]): The hyperparameter set that yielded the best score.
        - best_score (float): The best score achieved.
        - all_results (Dict[frozendict, float]): A dictionary mapping each hyperparameter
          set (as a frozendict) to its score.
    """
    filter_name = initial_filter_instance.__class__.__name__
    print(
        f"Starting grid search for {filter_name} with {len(param_grid)} combinations..."
    )
    print(
        f"  Using pre-simulated data: T_out: {T_out.shape}, Y_measurements: {Y_measurements.shape}"
    )
    if X_states is not None:
        print(f"  X_states: {X_states.shape}")
    else:
        print(
            "  X_states not provided, metrics will be calculated using predicted vs actual measurements"
        )

    # 1. Create a list of delayed calls for parallel filter evaluation
    delayed_calls = [
        delayed(_run_single_filter_evaluation)(
            base_filter_for_cloning=initial_filter_instance,
            filter_hyperparams=params,
            T_out=T_out,
            Y_measurements=Y_measurements,
            X_states=X_states,
            initial_state_x0_for_filter=initial_state_x0_for_filter,
            filter_init_P0=filter_init_P0,
            metric_name=metric_name,
        )
        for params in param_grid
    ]

    # 2. Execute evaluations in parallel
    print(f"  Evaluating {len(param_grid)} hyperparameter combinations...")
    results_list = Parallel(n_jobs=n_jobs)(
        tqdm(delayed_calls, desc=f"Grid searching {filter_name}", leave=False)
    )

    # 3. Store all results in a dictionary with hyperparameter sets as keys.
    all_results: Dict[frozendict, float] = {
        frozendict(param_set): score
        for param_set, score in zip(param_grid, results_list)
    }

    # 4. Find the best parameter set (lowest score is best).
    if not all_results:
        raise ValueError("Grid search yielded no results.")

    best_param_set_frozen = min(all_results, key=all_results.get)
    best_params = dict(best_param_set_frozen)
    best_score = all_results[best_param_set_frozen]

    # 5. Check if the best parameters are at the boundary of the search space
    _check_boundary_parameters(best_params, param_grid)

    print(f"Grid search complete for {filter_name}.")
    print(f"Best parameters found: {best_params}")
    print(f"Best {metric_name} score: {best_score:.4f}")

    return best_params, best_score, all_results


def _check_boundary_parameters(
    best_params: Dict[str, Any], param_grid: List[Dict[str, Any]]
) -> None:
    """
    Checks if any of the best parameters are at the boundary of the search space.
    If so, prints a warning message with specific direction to expand.

    Args:
        best_params: Dictionary of best parameters found by grid search
        param_grid: List of dictionaries containing all parameter combinations tested
    """
    # Extract all unique values for each parameter across the grid
    param_values = {}
    for params in param_grid:
        for param_name, param_value in params.items():
            if param_name not in param_values:
                param_values[param_name] = set()
            param_values[param_name].add(param_value)

    # Check if any best parameter is at the boundary
    boundary_params = []
    for param_name, best_value in best_params.items():
        if param_name in param_values:
            values = sorted(param_values[param_name])
            if best_value == values[0]:
                boundary_params.append((param_name, best_value, "lower"))
            elif best_value == values[-1]:
                boundary_params.append((param_name, best_value, "upper"))

    # Print warning if any parameters are at the boundary
    if boundary_params:
        param_warnings = []
        expand_suggestions = []

        for param_name, value, boundary in boundary_params:
            param_warnings.append(f"{param_name}={value}")
            if boundary == "lower":
                expand_suggestions.append(
                    f"decrease the minimum value for '{param_name}'"
                )
            else:
                expand_suggestions.append(
                    f"increase the maximum value for '{param_name}'"
                )

        warning_msg = (
            f"WARNING: The following optimal parameters are at the boundary of the search space: "
            f"{', '.join(param_warnings)}.\n"
            f"Consider expanding your grid search range: {', '.join(expand_suggestions)}."
        )
        print(warning_msg)
        logger.warning(warning_msg)
