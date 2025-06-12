"""
Huber Regression Filter.

This module implements a Huber Regression Filter, which is a robust Kalman filter
variant that uses Huber loss in its update step to handle outliers in both
state predictions and measurements. It adapts the logic from
`code/huberkf/filters/huber_regression.py` to fit the `BaseFilter` framework.
"""

from typing import Optional, Union

import numpy as np
import control

from .base_filter import BaseFilter
from .util import chi_squared_quantile


class HuberKalmanFilter(BaseFilter):
    """
    Huber Regression Filter.

    This filter applies Huber loss to both the state prediction error term and
    the measurement residual term in the mean update step, making it robust
    to outliers. The covariance update can also be scaled based on residuals.

    Attributes:
        P_current (np.ndarray | None): Current posterior error covariance matrix (nx, nx).
        x_pred (np.ndarray | None): Predicted (prior) state estimate (nx,).
        coef_s (float): Huber threshold for the state prediction error term.
        coef_o (float): Huber threshold for the measurement residual term.
        scale_covariance_update (bool): Whether to scale the covariance update using residuals.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Optional[Union[float, str]] = None,
        coef_o: Optional[Union[float, str]] = None,
        scale_covariance_update: bool = True,
    ):
        """
        Initializes the HuberRegressionFilter.

        Args:
            system_model: The state-space model of the system.
            cov_input: Process noise covariance matrix for the dynamic noise component.
                      This is the preferred way to specify process noise.
            cov_measurement: Measurement noise covariance matrix (ny, ny).
            coef_s (optional): Huber threshold for state prediction errors.
                               Defaults to sqrt(chi2.ppf(0.7, 1)).
            coef_o (optional): Huber threshold for measurement residuals.
                               Defaults to sqrt(chi2.ppf(0.7, 1)).
            scale_covariance_update (optional): If True, scales the covariance update
                                          based on state and measurement residuals.
                                          Defaults to True.
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        # Default Huber thresholds are based on chi-squared distribution with 1
        # DoF (for scalar residual components)
        if coef_s is None:
            self.coef_s = np.sqrt(chi_squared_quantile(0.7, 1))
        elif coef_s == "inf":
            self.coef_s = np.inf
        else:
            self.coef_s = coef_s

        if coef_o is None:
            self.coef_o = np.sqrt(chi_squared_quantile(0.7, 1))
        elif coef_o == "inf":
            self.coef_o = np.inf
        else:
            self.coef_o = coef_o
        self.coef_o = (
            coef_o if coef_o is not None else np.sqrt(chi_squared_quantile(0.7, 1))
        )

        self.scale_covariance_update = scale_covariance_update

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)
        self.cov_pred = self.A.dot(self.cov).dot(self.A.T) + self.cov_process_noise

    def _solve_huber_exact(
        self, y_k: np.ndarray, cov_pred_inv_sqrt: np.ndarray
    ) -> np.ndarray:
        """
        Update the state estimate using Huber regression.

        Args:
            y_k: Current measurement vector
            cov_pred_inv_sqrt: Cholesky factor of inverse predicted covariance

        Returns:
            np.ndarray: Updated state estimate
        """
        import cvxpy as cp

        # Create a CVXPY problem to solve for the optimal state estimate
        x_var = cp.Variable(self.nx, name="x_state_huber")

        # # Cost term related to state prediction error
        cost_state = cp.sum(
            cp.huber(cov_pred_inv_sqrt @ (x_var - self.x_pred), self.coef_s)
        )

        # # Cost term related to measurement residual
        cost_measurement = cp.sum(
            cp.huber(self.cov_meas_inv_sqrt @ (y_k - self.C @ x_var), self.coef_o)
        )

        objective = cp.Minimize(cost_state + cost_measurement)
        problem = cp.Problem(objective)
        problem.solve(solver=cp.CLARABEL)
        return x_var.value

    def _huber_cov_update(
        self,
        y_k: np.ndarray,
        cov_pred_inv: np.ndarray,
        cov_pred_inv_sqrt: np.ndarray,
    ):
        """
        Update the error covariance matrix based on the state estimate.

        Args:
            x_hat_updated: Updated state estimate
            y_k: Current measurement vector
            P_pred_inv: Inverse of predicted error covariance
            P_pred_inv_sqrt_term: Cholesky factor of inverse predicted covariance
            R_k_inv: Inverse of measurement noise covariance
            R_inv_sqrt_term: Cholesky factor of inverse measurement noise covariance

        Returns:
            np.ndarray: Updated error covariance matrix
        """
        # Calculate scaling factors for robust covariance update if enabled
        if self.scale_covariance_update:
            rs = np.abs(cov_pred_inv_sqrt.dot(self.x_hat - self.x_pred))
            ro = np.abs(self.cov_meas_inv_sqrt.dot(y_k - self.C.dot(self.x_hat)))

            # Element-wise scaling factor alpha for state term
            alpha = np.ones(rs.shape)
            alpha[rs > self.coef_s] = self.coef_s / rs[rs > self.coef_s]

            # Element-wise scaling factor beta for measurement term
            beta = np.ones(ro.shape)
            beta[ro > self.coef_o] = self.coef_o / ro[ro > self.coef_o]
        else:
            alpha, beta = 1.0, 1.0

        # Effective inverse of predicted state covariance, weighted by alpha
        weighted_P_pred_inv = (
            cov_pred_inv_sqrt.T.dot(np.diag(alpha)).dot(cov_pred_inv_sqrt)
            if isinstance(alpha, np.ndarray)
            else cov_pred_inv
        )

        # Effective inverse of measurement covariance, weighted by beta
        weighted_R_inv = (
            self.cov_meas_inv_sqrt.T.dot(np.diag(beta)).dot(self.cov_meas_inv_sqrt)
            if isinstance(beta, np.ndarray)
            else self.cov_meas_inv
        )

        cov_current_updated_inv = weighted_P_pred_inv + self.C.T.dot(
            weighted_R_inv
        ).dot(self.C)

        try:
            self.cov = np.linalg.inv(cov_current_updated_inv)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Updated covariance P_current_inv is singular. Error: {e}"
            ) from e

    def update_step(self, y_k: np.ndarray, dt: float):
        """
        Performs one step of the Huber Regression filtering (predict and correct).
        The `dt` argument is for compatibility with BaseFilter, not directly used
        if A and Q are already discretized.

        Args:
            y_k: The measurement vector received at the current time step (ny,).
            dt: The time step duration.

        Returns:
            The updated state estimate x_hat (nx,).
        """
        try:
            cov_pred_inv = np.linalg.inv(self.cov_pred)
            cov_pred_inv_sqrt = np.linalg.cholesky(cov_pred_inv).T
            # gain_matrix = self.cov_pred.dot(self.C.T).dot(
            #     np.linalg.inv(
            #         self.C.dot(cov_pred_inv).dot(self.C.T) + self.cov_measurement
            #     )
            # )
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Covariance matrix operations failed. Error: {e}"
            ) from e

        # --- 3. Mean Update (CVXPY) ---
        self.x_hat = self._solve_huber_exact(y_k, cov_pred_inv_sqrt)

        # --- 4. Covariance Update ---
        self._huber_cov_update(
            y_k,
            cov_pred_inv,
            cov_pred_inv_sqrt,
        )
