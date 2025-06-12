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

from .base_filter import BaseFilter
from .util import chi_squared_quantile


class SteadyOneStepHuberFilter(BaseFilter):
    """
    Steady-State Simple Huber Filter.

    This filter implements a steady-state version of the Huber Kalman filter that uses
    pre-computed matrices and a fixed error covariance matrix. The steady-state approach
    makes the filter more computationally efficient by avoiding covariance updates.

    Attributes:
        P (np.ndarray): Steady-state error covariance matrix.
        K (np.ndarray): Steady-state Kalman gain matrix.
        V_inv_sqrt (np.ndarray): Pre-computed Cholesky factor of measurement noise inverse.
        coef (float): Huber threshold for measurement residuals.
        step_size (float): Step size for the iterative mean update.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef: Optional[Union[float, str]] = None,
        step_size: float = 1.0,
    ):
        """
        Initialize the Steady-State Simple Huber Filter.

        Args:
            system_model: The state-space model of the system.
            cov_input: Process noise covariance matrix. Shape (nx, nx).
            cov_measurement: Measurement noise covariance matrix. Shape (ny, ny).
            coef: Huber threshold for measurement residuals. If None, computed from chi-squared quantile.
            step_size: Step size for the iterative mean update. Default is 2.0.
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        if coef is None:
            self.coef = np.sqrt(chi_squared_quantile(0.7, self.ny))
        elif coef == "inf":
            self.coef = np.inf
        else:
            self.coef = coef

        self.step_size = step_size

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
            P_initial: Ignored in the steady-state filter, which always uses P_steady.
        """
        super().reset(x_initial_estimate, self.cov)

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)

    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """Update step."""
        denom = np.linalg.norm(
            self.cov_meas_inv_sqrt.dot(y_k - self.C.dot(self.x_pred))
        )
        beta = 1.0 if denom < self.coef else self.coef / denom
        self.x_hat = self.x_pred + self.step_size * beta * self.gain_matrix.dot(
            y_k - self.C.dot(self.x_pred)
        )
