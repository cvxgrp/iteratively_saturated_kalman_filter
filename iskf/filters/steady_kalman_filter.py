"""
Steady-state Kalman filter implementation.

This module provides a SteadyKalmanFilter class that inherits from BaseFilter.
A steady-state Kalman filter uses a constant gain matrix rather than updating
the error covariance and gain at each time step, providing computational efficiency
for stationary systems.
"""

import numpy as np
import control
import scipy.linalg as la
from .base_filter import BaseFilter


class SteadyKalmanFilter(BaseFilter):
    """
    Steady-state Kalman Filter for linear discrete-time systems.

    This filter estimates the state of a system modeled as:
        x[k+1] = A x[k] + B w[k]
        y[k]   = C x[k] + v[k]

    where w[k] is process noise and v[k] is measurement noise.

    Unlike the standard Kalman filter, the steady-state filter uses a pre-computed
    constant Kalman gain matrix, which is optimal for stationary systems after
    the filter has converged.

    Attributes:
        K (np.ndarray): The steady-state Kalman gain matrix.
        P_steady (np.ndarray): The steady-state posterior error covariance matrix.
        P_predicted_steady (np.ndarray): The steady-state predicted error covariance matrix.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
    ):
        """
        Initializes the SteadyKalmanFilter structure.

        The filter computes the steady-state Kalman gain during initialization
        by solving the discrete-time algebraic Riccati equation.

        Args:
            system_model: The system model in state-space form.
            cov_input: Covariance matrix of the process noise.
            cov_measurement: Covariance matrix of the measurement noise.
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        self.cov = self.cov_steady
        self.cov_pred = self.cov_pred_steady
        self.cov_pred_inv = self.cov_pred_inv_steady
        self.cov_pred_inv_sqrt = self.cov_pred_inv_sqrt_steady
        self.gain_matrix = self.gain_steady

    def reset(self, x_initial_estimate: np.ndarray, P_initial: np.ndarray = None):
        """
        Resets the filter state to a new initial estimate.

        For a steady-state filter, we initialize x_hat but ignore P_initial
        since we use the pre-computed steady-state covariance.

        Args:
            x_initial_estimate: Initial state estimate
            P_initial: Ignored in the steady-state filter
        """
        # Use the parent class reset but with our steady-state P (posterior)
        super().reset(x_initial_estimate, self.cov)

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)

    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """Update step."""
        innovation = y_k - self.C.dot(self.x_pred)
        self.x_hat = self.x_pred + self.gain_matrix.dot(innovation)
