"""
Regularized Kalman Filter (RKF).

This module implements a Regularized Kalman Filter based on the principles
described in robust control and estimation literature, where regularization terms
are added to the standard Kalman filter cost function to handle uncertainties
or to enforce certain properties on the state estimate.

This implementation adapts the CVXPY-based update step from
`code/huberkf/filters/regularized.py` to fit the `BaseFilter` framework.
The covariance update remains standard.
"""

from typing import Optional, Union

import numpy as np
import control
import cvxpy as cp
import scipy.linalg as la  # Import scipy.linalg

from .base_filter import BaseFilter


class SteadyRegularizedKalmanFilter(BaseFilter):
    """
    Steady-State Regularized Kalman Filter.

    The mean update step solves an optimization problem:
    min_{x,s,o} 0.5 * ||P_steady_inv_sqrt (x - x_pred - s)||^2 + coef_s * ||s||_norm
              + 0.5 * ||R_inv_sqrt_fixed (y - Cx - o)||^2 + coef_o * ||o||_norm

    If coef_s or coef_o are np.inf, the respective regularization terms (and slack variables)
    are omitted from the optimization problem.

    The error covariance matrix P used for state penalty is fixed at the steady-state DARE solution P_steady.

    Attributes:
        P_steady (np.ndarray): Steady-state posterior error covariance matrix (nx, nx).
        P_steady_inv_sqrt (np.ndarray): Pre-computed Cholesky factor of P_steady inverse.
        R_inv_sqrt_fixed (np.ndarray): Pre-computed Cholesky factor of R inverse.
        coef_s (float): Regularization coefficient for the state deviation term.
                        Use np.inf to disable state regularization.
        coef_o (float): Regularization coefficient for the measurement residual term.
                        Use np.inf to disable measurement regularization.
        norm_type (int): The norm to use for regularization (e.g., 1 for L1, 2 for L2).
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Optional[Union[float, str]] = None,
        coef_o: Optional[Union[float, str]] = None,
        norm_type: int = 1,
    ):
        """
        Initializes the SteadyRegularizedKalmanFilter.

        Args:
            system_model: The state-space model.
            cov_input: Process noise covariance matrix for the dynamic noise component.
            cov_measurement: Measurement noise covariance matrix (ny, ny).
            coef_s (optional): Regularization coefficient for state deviations.
                               Defaults to 1.0. Use np.inf to disable.
            coef_o (optional): Regularization coefficient for measurement residuals.
                               Defaults to 1.0. Use np.inf to disable.
            norm_type (optional): Type of norm for regularization (1 or 2). Defaults to 1 (L1 norm).
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        if coef_s is None:
            self.coef_s = 1.0
        elif coef_s == "inf":
            self.coef_s = np.inf
        else:
            self.coef_s = coef_s

        if coef_o is None:
            self.coef_o = 1.0
        elif coef_o == "inf":
            self.coef_o = np.inf
        else:
            self.coef_o = coef_o

        self.norm_type = norm_type

        if not isinstance(self.coef_s, (float, int)):
            raise TypeError("coef_s must be a float or int (np.inf is a float).")
        if not isinstance(self.coef_o, (float, int)):
            raise TypeError("coef_o must be a float or int (np.inf is a float).")
        if norm_type not in [1, 2]:
            raise ValueError("norm_type must be 1 (L1 norm) or 2 (L2 norm).")

        self.cov = self.cov_steady
        self.cov_pred = self.cov_pred_steady
        self.cov_pred_inv = self.cov_pred_inv_steady
        self.cov_pred_inv_sqrt = self.cov_pred_inv_sqrt_steady
        self.gain_matrix = self.gain_steady

    def reset(
        self, x_initial_estimate: np.ndarray, P_initial: Optional[np.ndarray] = None
    ):
        """
        Resets the filter's state estimate. The error covariance remains fixed at P_steady.

        Args:
            x_initial_estimate: New initial state estimate (nx,).
            P_initial (optional): Ignored. The steady-state covariance P_steady is always used.
        """
        super().reset(x_initial_estimate, self.cov)

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)

    def update_step(self, y_k: np.ndarray, dt: float) -> np.ndarray:
        """
        Performs one step of Steady-State Regularized Kalman filtering.
        Covariance matrix P_current remains fixed at P_steady.

        Args:
            y_k: Measurement vector (ny,).
            dt: Time step duration (not explicitly used if model is discrete).

        Returns:
            Updated state estimate x_hat (nx,).
        """
        # --- 3. Mean Update (CVXPY) ---
        x_var = cp.Variable(self.nx)
        objective = 0.0

        # State regularization term
        if np.isfinite(self.coef_s):
            s_var = cp.Variable(self.nx)
            objective += 0.5 * cp.sum_squares(
                self.cov_pred_inv_sqrt @ (x_var - self.x_pred - s_var)
            )
            objective += self.coef_s * cp.norm(s_var, self.norm_type)
        else:  # No state regularization
            objective += 0.5 * cp.sum_squares(
                self.cov_pred_inv_sqrt @ (x_var - self.x_pred)
            )

        # Measurement regularization term
        if np.isfinite(self.coef_o):
            o_var = cp.Variable(self.ny)
            objective += 0.5 * cp.sum_squares(
                self.cov_meas_inv_sqrt @ (y_k - self.C @ x_var - o_var)
            )
            objective += self.coef_o * cp.norm(o_var, self.norm_type)
        else:  # No measurement regularization
            objective += 0.5 * cp.sum_squares(
                self.cov_meas_inv_sqrt @ (y_k - self.C @ x_var)
            )

        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver=cp.CLARABEL)
        self.x_hat = x_var.value
