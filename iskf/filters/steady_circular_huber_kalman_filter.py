"""
Steady-State Huberized State-Space Kalman Filter.

This module implements a steady-state version of the Huberized Kalman filter for
linear discrete-time systems. In this steady-state version, the error covariance
matrix P is pre-computed and does not change between update steps.
"""

from typing import Optional, Union

import numpy as np
import control
import scipy.linalg as la

from .circular_huber_kalman_filter import CircularHuberKalmanFilter


class SteadyCircularHuberKalmanFilter(CircularHuberKalmanFilter):
    """
    Steady-State Circular Huber Kalman Filter.

    This filter extends the regular Huber Kalman filter by pre-computing matrices
    during initialization and using a fixed error covariance that doesn't change
    between updates, making it more computationally efficient.

    Attributes:
        P_steady (np.ndarray): Steady-state error covariance matrix.
        P_predicted (np.ndarray): Pre-computed predicted covariance.
        P_predicted_inv (np.ndarray): Pre-computed inverse of predicted covariance.
        P_predicted_inv_sqrt (np.ndarray): Pre-computed Cholesky factor of inverse.
        R_inv_sqrt (np.ndarray): Pre-computed Cholesky factor of R inverse.
        M_scale (np.ndarray): Pre-computed scaling matrix for gradient updates.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Optional[Union[float, str]] = None,
        coef_o: Optional[Union[float, str]] = None,
        num_iters: Union[int, float, str] = np.inf,
        step_size: float = 1.0,
        mean_update_initialization: str = "predict",
        use_exact_mean_solve: bool = False,
    ):
        """
        Initialize the Steady-State Huberized Kalman Filter.

        Inherits parameters from CircularHuberKalmanFilter but pre-computes
        matrices during initialization.
        """
        # Initialize the parent class
        super().__init__(
            system_model,
            cov_input,
            cov_measurement,
            coef_s=coef_s,
            coef_o=coef_o,
            num_iters=num_iters,
            step_size=step_size,
            mean_update_initialization=mean_update_initialization,
            use_exact_mean_solve=use_exact_mean_solve,
        )

        self.cov = self.cov_steady
        self.cov_pred = self.cov_pred_steady
        self.cov_pred_inv = self.cov_pred_inv_steady
        self.cov_pred_inv_sqrt = self.cov_pred_inv_sqrt_steady
        self.gain_matrix = self.gain_steady

    def reset(self, x_initial_estimate, P_initial=None):
        """
        Resets the filter's state estimate but keeps the steady-state covariance.

        Args:
            x_initial_estimate: The new initial estimate for the state vector.
        """
        super().reset(x_initial_estimate, self.cov)

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)

    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """
        Performs the update/correction step.

        Args:
            y_k: The measurement vector received at the current time step.
            dt: The time step duration (not used in steady-state filter).

        Returns:
            The updated state estimate x_hat.
        """
        if self.use_exact_mean_solve or np.isinf(self.num_iters):
            self.x_hat = self._solve_circular_huber_exact(y_k, self.cov_pred_inv_sqrt)
        else:
            self.x_hat = self._solve_circular_huber_scaled_gradient(
                y_k, self.cov_pred_inv_sqrt, self.gain_matrix
            )
