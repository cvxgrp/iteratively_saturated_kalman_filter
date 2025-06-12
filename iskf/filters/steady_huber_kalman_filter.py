"""
Steady-State Huber Kalman Filter.

This module implements a steady-state version of the Huber Regression Filter.
It's a robust Kalman filter variant that uses Huber loss in its update step
but assumes a fixed covariance matrix, which can lead to computational savings.
"""

from typing import Optional, Union

import numpy as np
import control
import scipy.linalg as la

from .huber_kalman_filter import HuberKalmanFilter


class SteadyHuberKalmanFilter(HuberKalmanFilter):
    """
    Steady-State Huber Regression Filter.

    This filter is identical to the HuberKalmanFilter except that it does not
    update the covariance matrix after initialization. It assumes that the
    filter will reach a steady state, and thus the covariance matrix can be
    fixed. This leads to computational savings as the matrix inversions and
    Cholesky decompositions only need to be performed once.

    Attributes:
        P_pred_fixed (np.ndarray | None): Fixed predicted error covariance matrix.
        P_pred_inv_fixed (np.ndarray | None): Fixed inverse of predicted error covariance.
        P_pred_inv_sqrt_fixed (np.ndarray | None): Fixed Cholesky factor of inverse predicted covariance.
        R_inv_fixed (np.ndarray | None): Fixed inverse of measurement noise covariance.
        R_inv_sqrt_fixed (np.ndarray | None): Fixed Cholesky factor of inverse measurement covariance.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Optional[Union[float, str]] = None,
        coef_o: Optional[Union[float, str]] = None,
    ):
        """
        Initializes the SteadyHuberKalmanFilter.

        Args:
            system_model: The state-space model of the system.
            cov_input: Process noise covariance matrix for the dynamic noise component.
            cov_measurement: Measurement noise covariance matrix (ny, ny).
            coef_s (optional): Huber threshold for state prediction errors.
                              Defaults to sqrt(chi2.ppf(0.7, 1)).
            coef_o (optional): Huber threshold for measurement residuals.
                              Defaults to sqrt(chi2.ppf(0.7, 1)).
        """
        # Call parent constructor with scale_covariance_update=False as we won't update covariance
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
            coef_s=coef_s,
            coef_o=coef_o,
            scale_covariance_update=False,  # No need for scaling in steady-state
        )
        self.cov = self.cov_steady
        self.cov_pred = self.cov_pred_steady
        self.cov_pred_inv_sqrt = self.cov_pred_inv_sqrt_steady

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
        """Update step."""
        self.x_hat = self._solve_huber_exact(y_k, self.cov_pred_inv_sqrt)
