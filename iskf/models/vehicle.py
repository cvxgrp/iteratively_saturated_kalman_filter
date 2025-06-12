"""Vehicle state-space model for simulation with process and measurement noise.

This module provides a function to create a discrete-time state-space model for a vehicle
with friction, where the system is driven by both process noise (unmodelled forces) and
measurement noise.
"""

import numpy as np
import control


def vehicle_ss(gamma: float = 0.05, dt: float = 0.05) -> control.StateSpace:
    """Creates a state-space model for a vehicle with friction.

    The model combines process noise (unmodelled forces) and measurement noise into a single
    input vector w = [w_dyn, w_meas]^T. The system matrices are constructed so that:
        x[k+1] = A * x[k] + B_system * w[k]
        y[k]   = C * x[k] + D_system * w[k]

    States: [pos_x, pos_y, vel_x, vel_y]
    Outputs: [pos_x, pos_y]

    Args:
        gamma: Friction coefficient
        dt: Time step

    Returns:
        A control.StateSpace object representing the system
    """
    # State transition matrix
    A = np.array(
        [
            [1, 0, dt * (1 - 0.5 * gamma * dt), 0],
            [0, 1, 0, dt * (1 - 0.5 * gamma * dt)],
            [0, 0, 1 - gamma * dt, 0],
            [0, 0, 0, 1 - gamma * dt],
        ],
        dtype=float,
    )

    # Process noise effect on states (like acceleration inputs)
    B_dynamic_noise = np.array(
        [[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]], dtype=float
    )
    dim_w_dyn = B_dynamic_noise.shape[1]

    # Output matrix (maps states to measurements)
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    num_outputs = C.shape[0]

    # Measurement noise effect (direct feedthrough)
    dim_w_meas = num_outputs
    D_measurement_noise = np.eye(dim_w_meas, dtype=float)

    # Combine noise effects into system matrices
    num_states = A.shape[0]
    B_system = np.hstack(
        (B_dynamic_noise, np.zeros((num_states, dim_w_meas), dtype=float))
    )
    D_system = np.hstack(
        (np.zeros((num_outputs, dim_w_dyn), dtype=float), D_measurement_noise)
    )

    return control.ss(A, B_system, C, D_system, dt=dt)
