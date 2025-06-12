"""
Kalman filter implementation.

This module provides a KalmanFilter class that inherits from BaseFilter.
It implements the standard Kalman filter equations directly using NumPy.
"""

import numpy as np
from .base_filter import BaseFilter


class KalmanFilter(BaseFilter):
    """
    Kalman Filter for linear discrete-time systems.

    This filter estimates the state of a system modeled as:
        x[k+1] = A x[k] + B_sys w_combined[k]
        y[k]   = C x[k] + D_sys w_combined[k]

    where `w_combined = [w_dyn, w_meas]^T` consists of dynamic noise `w_dyn`
    (affecting states) and measurement noise `w_meas` (affecting measurements).

    The underlying LQE is constructed for a plant:
        x[k+1] = A_lqe x[k] + B_lqe w_dyn[k]
        y[k]   = C_lqe x[k] (+ v[k] implicitly, where cov(v[k]) = R_measurement_noise_cov)

    Attributes:
        P_current (np.ndarray | None): Current error covariance matrix (nx, nx), initialized by `reset`.
        This class relies on `BaseFilter` to store and manage:
        - `A`: State transition matrix.
        - `C`: Observation matrix.
        - `cov_process_noise` (Q): Process noise covariance.
        - `cov_measurement` (R): Measurement noise covariance.
        - `x_hat`: Current state estimate, initialized by `reset`.
        - `P`: Current error covariance matrix, initialized by `reset`.
        - `nx`: Number of states.
        - `ny`: Number of outputs/measurements.
    """

    def __init__(
        self,
        system_model,  # Kept for BaseFilter compatibility, type hint removed
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
    ):
        """
        Initializes the KalmanFilter structure.
        The actual state estimate (x_hat) and error covariance (P_current)
        are initialized by calling the `reset` method, typically via `estimate`.

        Args:
            system_model: The full system model.
            Q_dynamic_noise_cov: Covariance of the dynamic noise component `w_dyn` that enters
                                 the system via the G matrix. This is the preferred way
                                 to specify process noise.
            R_measurement_noise_cov: Covariance of the measurement noise component `w_meas`.
                                     Shape (ny, ny). Alias for R_covariance for compatibility.
            Q_covariance: Process noise covariance matrix, shape (nx, nx). Legacy parameter.
            R_covariance: Measurement noise covariance matrix, shape (ny, ny). Legacy parameter.
            G_matrix: Optional matrix that maps dynamic noise to state dynamics.
                      If not provided but Q_dynamic_noise_cov is, it will be extracted
                      from the first columns of system_model.B.
        """
        # Call super().__init__ with all parameters for backward compatibility
        # BaseFilter will handle the logic for Q and R
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

    def predict_step(self, dt: float) -> None:
        """
        Performs the prediction step of the Kalman filter.
        Updates the internal state prediction (x_pred) and covariance prediction (cov_pred).

        Args:
            dt: The time step duration. Not used in the discrete Kalman filter
                as the dynamics are already time-discretized.
        """
        # State prediction
        self.x_pred = self.A.dot(self.x_hat)

        # Covariance prediction
        self.cov_pred = self.A.dot(self.cov).dot(self.A.T) + self.cov_process_noise

    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """
        Performs one step of the Kalman filtering process (predict and correct).

        Args:
            y_k: The measurement vector received at the current time step (ny,).
            dt: The time step duration. (Not explicitly used as the estimator is discrete).

        Returns:
            The updated state estimate x_hat (nx,).
        """
        s = self.C.dot(self.cov_pred).dot(self.C.T) + self.cov_measurement
        gain_matrix = self.cov_pred.dot(self.C.T).dot(np.linalg.inv(s))

        innovation = y_k - self.C.dot(self.x_pred)
        self.x_hat = self.x_pred + gain_matrix.dot(innovation)
        self.cov = (np.eye(self.nx) - gain_matrix.dot(self.C)).dot(self.cov_pred)
