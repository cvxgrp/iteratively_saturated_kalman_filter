"""
Huberized State-Space Kalman Filter (H-SSKF).

This module implements a Huberized Kalman filter for linear discrete-time systems,
designed to be robust to outliers in both process and measurement noise by
applying Huber's influence function principles during the state estimation update.
"""

from typing import Optional, Union

import numpy as np
import control

from .base_filter import BaseFilter
from .util import chi_squared_quantile


class IterSatKalmanFilter(BaseFilter):
    """
    Iteratively Saturated Kalman Filter.

    This filter extends the standard Kalman filter by incorporating Huber's
    influence function to limit the impact of outliers during the state update step.
    It aims for robustness when noise distributions have heavy tails.

    Attributes:
        P_current (np.ndarray | None): Current posterior error covariance matrix (nx, nx).
        P_predicted (np.ndarray | None): Predicted (prior) error covariance matrix (nx, nx).
        coef_s (float): Huber threshold for the state prediction residual term.
        coef_o (float): Huber threshold for the measurement residual term.
        num_iters (int): Number of iterations for the iterative scaled gradient method.
        step_size (float): Step size for the iterative mean update.
        scale_covariance_update (bool): If True, the covariance update is scaled based on residuals.
        mean_update_initialization (str): Method to initialize the iterative mean update.
        use_exact_mean_solve (bool): If True, attempts to use CVXPY for an exact solution.
        R_inv (np.ndarray): Pre-computed inverse of the measurement noise covariance.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Optional[Union[float, str]] = None,
        coef_o: Optional[Union[float, str]] = None,
        num_iters: Union[int, float, str] = 3,
        step_size: float = 1.0,
        scale_covariance_update: bool = True,
        mean_update_initialization: str = "predict",
        use_exact_mean_solve: bool = False,
    ):
        """
        Initialize the Huberized State-Space Kalman Filter.

        Args:
            system_model: The state-space model of the system.
            cov_input: Process noise covariance matrix for the dynamic noise component.
                      Shape (nx, nx). This is the preferred way to specify process noise.
            cov_measurement: Measurement noise covariance matrix. Shape (ny, ny).
            coef_s: Huber threshold for state prediction residual. If None, computed from chi-squared quantile.
            coef_o: Huber threshold for measurement residual. If None, computed from chi-squared quantile.
            num_iters: Number of iterations for the iterative mean update. Default is 1.
            step_size: Step size for the iterative mean update. Default is 1.8.
            scale_covariance_update: Whether to scale the covariance update based on residuals. Default is True.
            mean_update_initialization: Method to initialize the iterative mean update.
                                      Must be one of: 'prior', 'predict', or 'kalman'. Default is 'predict'.
            use_exact_mean_solve: Whether to use CVXPY for an exact solution to the mean update. Default is False.
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )
        self.use_exact_mean_solve = use_exact_mean_solve

        if coef_s is None:
            self.coef_s = np.sqrt(chi_squared_quantile(0.7, self.nx))
        elif coef_s == "inf":
            self.coef_s = np.inf
        else:
            self.coef_s = coef_s

        if coef_o is None:
            self.coef_o = np.sqrt(chi_squared_quantile(0.7, self.ny))
        elif coef_o == "inf":
            self.coef_o = np.inf
        else:
            self.coef_o = coef_o

        if num_iters == "inf" or np.isinf(num_iters):
            self.num_iters = np.inf
            self.use_exact_mean_solve = True
        elif num_iters < 0:
            raise ValueError("num_iters must be a nonnegative integer.")
        else:
            self.num_iters = int(num_iters)

        if step_size <= 0:
            raise ValueError("step_size must be a positive number.")
        self.step_size = step_size

        self.scale_covariance_update = scale_covariance_update

        if mean_update_initialization not in ["prior", "predict", "kalman"]:
            raise ValueError(
                "mean_update_initialization must be 'prior', 'predict', or 'kalman'."
            )
        self.mean_update_initialization = mean_update_initialization

        self.C_stacked = np.vstack([self.C, np.eye(self.nx)])

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)
        self.cov_pred = self.A.dot(self.cov).dot(self.A.T) + self.cov_process_noise

    def _solve_iskf_scaled_gradient(
        self, y_k: np.ndarray, cov_pred_inv_sqrt: np.ndarray, gain_matrix: np.ndarray
    ) -> np.ndarray:
        """Solve using scaled gradient method."""
        x = self.x_pred.copy()
        for _ in range(self.num_iters):
            a_denom = np.linalg.norm(cov_pred_inv_sqrt.dot(x - self.x_pred))
            alpha = 1.0 if a_denom < self.coef_s else self.coef_s / a_denom

            b_denom = np.linalg.norm(self.cov_meas_inv_sqrt.dot(y_k - self.C.dot(x)))
            beta = 1.0 if b_denom < self.coef_o else self.coef_o / b_denom

            gain_stacked = np.hstack(
                [
                    beta * gain_matrix,
                    alpha * (np.eye(self.nx) - gain_matrix.dot(self.C)),
                ]
            )
            y_eff = np.hstack([y_k, self.x_pred])
            x += self.step_size * gain_stacked.dot(y_eff - self.C_stacked.dot(x))

        return x

    def _solve_iskf_exact(
        self, y_k: np.ndarray, cov_pred_inv_sqrt: np.ndarray
    ) -> np.ndarray:
        """Update step using cvxpy."""
        import cvxpy as cp

        x = cp.Variable(self.nx)
        obj = 0.0
        if np.isfinite(self.coef_s):
            obj += cp.huber(cp.norm(cov_pred_inv_sqrt @ (x - self.x_pred)), self.coef_s)
        else:
            obj += cp.sum_squares(cov_pred_inv_sqrt @ (x - self.x_pred))

        if np.isfinite(self.coef_o):
            obj += cp.huber(
                cp.norm(self.cov_meas_inv_sqrt @ (y_k - self.C @ x)), self.coef_o
            )
        else:
            obj += cp.sum_squares(self.cov_meas_inv_sqrt @ (y_k - self.C @ x))

        cp.Problem(cp.Minimize(obj)).solve()
        return x.value

    def _iskf_cov_update(
        self,
        y_k: np.ndarray,
        cov_pred_inv: np.ndarray,
        cov_pred_inv_sqrt: np.ndarray,
    ) -> None:
        """Update the covariance matrix."""
        state_residual_vec = self.x_hat - self.x_pred
        measurement_residual_vec = y_k - self.C.dot(self.x_hat)

        # scale covariance update
        alpha_s_cov = 1.0
        beta_o_cov = 1.0
        if self.scale_covariance_update:
            rs_norm = np.linalg.norm(cov_pred_inv_sqrt.dot(state_residual_vec))
            ro_norm = np.linalg.norm(
                self.cov_meas_inv_sqrt.dot(measurement_residual_vec)
            )

            alpha_s_cov = self.coef_s / rs_norm if rs_norm > self.coef_s else 1.0
            beta_o_cov = self.coef_o / ro_norm if ro_norm > self.coef_o else 1.0

            alpha_s_cov = max(alpha_s_cov, 1e-9)
            beta_o_cov = max(beta_o_cov, 1e-9)

        cov_inv = alpha_s_cov * cov_pred_inv + beta_o_cov * self.C.T.dot(
            self.cov_meas_inv
        ).dot(self.C)

        try:
            self.cov = np.linalg.inv(cov_inv)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Updated covariance P_current_inv is singular. Error: {e}"
            ) from e

    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """Update step."""
        try:
            cov_pred_inv = np.linalg.inv(self.cov_pred)
            cov_pred_inv_sqrt = np.linalg.cholesky(cov_pred_inv).T
            s_matrix = self.C.dot(self.cov_pred).dot(self.C.T) + self.cov_measurement
            gain_matrix = self.cov_pred.dot(self.C.T).dot(np.linalg.inv(s_matrix))
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Covariance matrix operations failed. Error: {e}"
            ) from e

        if self.use_exact_mean_solve or np.isinf(self.num_iters):
            self.x_hat = self._solve_iskf_exact(y_k, cov_pred_inv_sqrt)
        else:
            self.x_hat = self._solve_iskf_scaled_gradient(
                y_k, cov_pred_inv_sqrt, gain_matrix
            )

        # Update covariance
        self._iskf_cov_update(y_k, cov_pred_inv, cov_pred_inv_sqrt)
