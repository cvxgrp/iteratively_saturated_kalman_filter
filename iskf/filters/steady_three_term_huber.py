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

from .base_filter import BaseFilter


class SteadyThreeTermHuberFilter(BaseFilter):
    """
    A steady-state Kalman filter with Huber regularization for robust state estimation.

    This filter implements a regularized Kalman filter that uses Huber loss functions
    to handle outliers in both state predictions and measurements.

    Setting coef_s or coef_o to np.inf disables the respective regularization term.

    Attributes:
        coef_s (float): Regularization coefficient for state deviations. Use np.inf to disable.
        coef_o (float): Regularization coefficient for measurement residuals. Use np.inf to disable.
        step_size (float): Step size for the optimization solver.
        num_iters (int): Maximum number of iterations for the optimization solver.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef_s: Union[float, str] = 1.0,
        coef_o: Union[float, str] = 1.0,
        step_size: float = 1.8,
        num_iters: Union[int, float] = 2,
    ):
        """
        Initialize the SteadyThreeTermHuberFilter.

        This filter implements a steady-state Kalman filter with Huber regularization
        for robust state estimation. It uses Huber loss functions to handle outliers
        in both state predictions and measurements.

        Args:
            system_model: The state-space model.
            cov_input: Process noise covariance matrix for the dynamic noise component.
            cov_measurement: Measurement noise covariance matrix (ny, ny).
            coef_s (optional): Regularization coefficient for state deviations.
                             Defaults to 1.0. Use np.inf to disable.
            coef_o (optional): Regularization coefficient for measurement residuals.
                             Defaults to 1.0. Use np.inf to disable.
            step_size (optional): Step size for the optimization solver. Defaults to 1.8.
            num_iters (optional): Maximum number of iterations for the optimization solver.
                                Defaults to 2. Use "inf" for unlimited iterations.
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        if coef_s == "inf":
            self.coef_s = np.inf
        else:
            self.coef_s = coef_s

        if coef_o == "inf":
            self.coef_o = np.inf
        else:
            self.coef_o = coef_o

        self.step_size = step_size
        self.num_iters = num_iters
        if self.num_iters == "inf":
            self.num_iters = np.inf

        if not isinstance(self.coef_s, (float, int)):
            raise TypeError("coef_s must be a float or int (np.inf is a float).")
        if not isinstance(self.coef_o, (float, int)):
            raise TypeError("coef_o must be a float or int (np.inf is a float).")

        self.cov_input_inv = np.linalg.inv(self.cov_input)
        self.cov_input_inv_sqrt = np.linalg.cholesky(self.cov_input_inv).T
        self.cov = self.cov_steady
        self.cov_inv = self.cov_inv_steady
        self.cov_inv_sqrt = self.cov_inv_sqrt_steady
        self.cov_pred = self.cov_pred_steady
        self.cov_pred_inv = self.cov_pred_inv_steady
        self.cov_pred_inv_sqrt = self.cov_pred_inv_sqrt_steady
        self.gain_matrix = self.gain_steady

        self.s_matrix = self.C.dot(self.cov_pred).dot(self.C.T) + self.cov_measurement
        self.s_matrix_inv = np.linalg.inv(self.s_matrix)

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
        pass

    def _solve_grad(self, P, W_inv_sqrt, V_inv_sqrt, S_inv, y, xhat):
        W = self.cov_input
        B = self.B
        C = self.C
        A = self.A
        m = self.nu
        n = self.nx

        block1 = np.block(
            [
                [
                    -W.dot(B.T).dot(C.T).dot(S_inv),
                    np.eye(m) - W.dot(B.T).dot(C.T).dot(S_inv).dot(C).dot(B),
                ],
                [
                    -P.dot(A.T).dot(C.T).dot(S_inv),
                    -P.dot(A.T).dot(C.T).dot(S_inv).dot(C).dot(B),
                ],
            ]
        )

        block2 = np.vstack(
            [
                -W.dot(B.T).dot(C.T).dot(S_inv).dot(C).dot(A),
                np.eye(n) - P.dot(A.T).dot(C.T).dot(S_inv).dot(C).dot(A),
            ]
        )

        def scaled_grad(rz):
            r = rz[:m]
            z = rz[m:]
            a_denom = np.linalg.norm(V_inv_sqrt.dot(y - C @ A @ z - C @ B @ r))
            alpha = 1.0 if a_denom < self.coef_o else self.coef_o / a_denom
            alpha = alpha * (y - C @ A @ z - C @ B @ r)

            b_denom = np.linalg.norm(W_inv_sqrt.dot(r))
            beta = 1.0 if b_denom < self.coef_s else self.coef_s / b_denom
            beta = beta * r

            return block1.dot(np.hstack([alpha, beta])) + block2.dot(z - xhat)

        rz = np.hstack([np.zeros(m), xhat])
        for _ in range(self.num_iters):
            rz -= self.step_size * scaled_grad(rz)

        r_opt = rz[:m]
        z_opt = rz[m:]
        x_opt = A @ z_opt + B @ r_opt
        return x_opt

    def _solve_cvxpy(self, P_inv_sqrt, W_inv_sqrt, V_inv_sqrt, y, xhat):
        import cvxpy as cp

        n = self.A.shape[0]
        m = self.B.shape[1]
        z = cp.Variable(n)
        r = cp.Variable(m)

        objective = 0.5 * cp.sum_squares(P_inv_sqrt @ (z - xhat))

        if np.isfinite(self.coef_o):
            objective += 0.5 * cp.huber(
                cp.norm(V_inv_sqrt @ (y - self.C @ self.A @ z - self.C @ self.B @ r)),
                M=self.coef_o,
            )
        else:
            objective += 0.5 * cp.sum_squares(
                V_inv_sqrt @ (y - self.C @ self.A @ z - self.C @ self.B @ r)
            )

        if np.isfinite(self.coef_s):
            objective += 0.5 * cp.huber(cp.norm(W_inv_sqrt @ r), M=self.coef_s)
        else:
            objective += 0.5 * cp.sum_squares(W_inv_sqrt @ r)

        problem = cp.Problem(cp.Minimize(objective))
        problem.solve()
        x_opt = self.A @ z.value + self.B @ r.value

        return x_opt

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
        if np.isfinite(self.num_iters):
            self.x_hat = self._solve_grad(
                self.cov,
                self.cov_input_inv_sqrt,
                self.cov_meas_inv_sqrt,
                self.s_matrix_inv,
                y_k,
                self.x_hat,
            )
        else:
            self.x_hat = self._solve_cvxpy(
                self.cov_inv_sqrt,
                self.cov_input_inv_sqrt,
                self.cov_meas_inv_sqrt,
                y_k,
                self.x_hat,
            )
