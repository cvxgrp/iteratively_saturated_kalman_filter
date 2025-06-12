"""
Weighted Observation Likelihood Filter (WoLF) - Experiment Version.

This module implements the Weighted Observation Likelihood Filter (WoLF)
for state estimation, designed to be robust to outliers in measurements.
It inherits from the BaseFilter class and is adapted from the WoLF
implementation in `code/huberkf/filters/wolf.py`.
"""

import numpy as np
import control

from .base_filter import BaseFilter


class WeightedLikelihoodFilter(BaseFilter):
    """
    Weighted Observation Likelihood Filter (WoLF) - Experiment Version.

    This filter adapts the standard Kalman filter equations by weighting
    the measurement likelihood, making it more robust to non-Gaussian noise
    or outliers in the measurements.

    Attributes:
        P (np.ndarray | None): Current estimate of the error covariance matrix, shape (nx, nx).
                               Initialized by `reset`.
        # cov_process_noise and cov_measurement are inherited from BaseFilter.
        coef (float): Coefficient used by the weighting functions. Its interpretation
                      varies depending on the chosen weighting function.
        weighting (str): The type of weighting function to use. Supported values:
                         "IMQ", "MD", "TMD", "HUBER".
        weight_fn (callable): The selected weighting function.
        R_inv (np.ndarray): Pre-computed inverse of the measurement noise covariance matrix.
        cov_meas_inv_sqrt (np.ndarray): Pre-computed Cholesky factor of the inverse of cov_measurement.
                              Used for Mahalanobis distance calculations.
                              L_R_inv residual gives a vector whose squared L2 norm is the
                              Mahalanobis distance: residual.T R_inv residual.
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
        coef: float = 1.0,
        weighting: str = "IMQ",
    ):
        """
        Initializes the WeightedLikelihoodFilter filter structure.
        The actual state estimate (x_hat) and error covariance (P)
        are initialized by calling the `reset` method, typically via `estimate`.

        Args:
            system_model: A `control.StateSpace` object representing the discrete-time
                                          system dynamics.
            cov_input: Process noise covariance matrix for the dynamic noise component.
                       This is the preferred way to specify process noise.
            cov_measurement: Measurement noise covariance matrix (ny, ny).
            coef: Coefficient for the weighting function.
            weighting: Type of weighting function ("IMQ", "MD", "TMD", "HUBER").
                       Defaults to "IMQ".
        """
        super().__init__(
            system_model,
            cov_input=cov_input,
            cov_measurement=cov_measurement,
        )

        self.coef = coef
        self.weighting = weighting.upper()

        if self.weighting not in ["IMQ", "MD", "TMD", "HUBER"]:
            raise ValueError(f"Unsupported weighting function type: {self.weighting}")

        # Dynamically assign the weighting function
        self.weight_fn = getattr(self, f"_{self.weighting.lower()}_weight")

    # --- Weighting Functions ---
    def _imq_weight(self, y_k: np.ndarray, y_hat_prior: np.ndarray) -> float:
        """
        Inverse Multi-Quadratic (IMQ) weighting function.
        Calculates weight based on the Euclidean norm of the residual.

        Args:
            y_k: Current measurement.
            y_hat_prior: Predicted measurement.

        Returns:
            Calculated weight (scalar).
        """
        residual = y_k - y_hat_prior
        # The power is -1, as per the paper's w_k^2 formula (alpha = -1 for IMQ kernel)
        # w_sq = (1 + ||residual||^2 / coef^2)^-1
        return np.power(1 + (np.linalg.norm(residual) / self.coef) ** 2, -1.0)

    def _md_weight(self, y_k: np.ndarray, y_hat_prior: np.ndarray) -> float:
        """
        Mahalanobis Distance (MD) based weighting function.
        This is effectively an IMQ weight on the Mahalanobis-transformed residual.

        Args:
            y_k: Current measurement.
            y_hat_prior: Predicted measurement.

        Returns:
            Calculated weight (scalar).
        """
        residual = y_k - y_hat_prior
        # The squared norm of this transformed residual is the Mahalanobis distance.
        transformed_residual_norm = np.linalg.norm(self.cov_meas_inv_sqrt.dot(residual))
        return np.power(1 + (transformed_residual_norm / self.coef) ** 2, -1.0)

    def _tmd_weight(self, y_k: np.ndarray, y_hat_prior: np.ndarray) -> float:
        """
        Threshold Mahalanobis Distance (TMD) weighting function.
        Assigns weight 1 if Mahalanobis distance squared is below a threshold (self.coef),
        and 0 otherwise.

        Args:
            y_k: Current measurement.
            y_hat_prior: Predicted measurement.

        Returns:
            Calculated weight (1.0 or 0.0).
        """
        residual = y_k - y_hat_prior
        mahalanobis_dist_sq = np.linalg.norm(self.cov_meas_inv_sqrt.dot(residual)) ** 2
        if mahalanobis_dist_sq <= self.coef:
            return 1.0
        return 0.0  # Or a very small number to avoid issues if R/w_sq is used.
        # The reference code clips to 1e-12 later.

    def _huber_weight(self, y_k: np.ndarray, y_hat_prior: np.ndarray) -> float:
        """
        Huber-like weighting function.
        Approximates Huber's influence function using Mahalanobis distance.
        Weight is min(1, coef / Mahalanobis_distance_norm).

        Args:
            y_k: Current measurement.
            y_hat_prior: Predicted measurement.

        Returns:
            Calculated weight (scalar).
        """
        residual = y_k - y_hat_prior
        mahalanobis_dist_norm = np.linalg.norm(self.cov_meas_inv_sqrt.dot(residual))
        if (
            mahalanobis_dist_norm < 1e-9
        ):  # Avoid division by zero if residual is very small
            return 1.0
        return min(1.0, self.coef / mahalanobis_dist_norm)

    def predict_step(self, dt: float) -> None:
        """Predict step."""
        self.x_pred = self.A.dot(self.x_hat)
        self.cov_pred = self.A.dot(self.cov).dot(self.A.T) + self.cov_process_noise

    def update_step(self, y_k: np.ndarray, dt: float) -> np.ndarray:
        """
        Performs one step of the WoLF filtering process (predict and correct).

        The `dt` parameter is noted for compatibility with BaseFilter but is not
        explicitly used in these discrete update equations if A and cov_process_noise
        are already discretized for the interval.

        Args:
            y_k: The measurement vector received at the current time step k.
                 Shape (ny,).
            dt: The time duration since the last update. (Not directly used here
                if system model and Q are already discrete for the step).

        Returns:
            The updated state estimate (x_hat_k_posterior) after processing y_k.
        """
        y_hat_prior = self.C.dot(self.x_pred)

        # Calculate weight w_sq based on (y_k, y_hat_prior)
        # The function self.weight_fn already returns w_k^2 (or the weight for R/w_k^2)
        w_sq = np.clip(self.weight_fn(y_k, y_hat_prior), 1e-12, None)

        S_k_mod = self.C.dot(self.cov_pred).dot(self.C.T) + self.cov_measurement / w_sq
        try:
            S_k_mod_inv = np.linalg.inv(S_k_mod)
        except np.linalg.LinAlgError as exc:
            # If S_k_mod is singular, try to use pseudo-inverse or handle error.
            # For simplicity, we re-raise, but robust code might use pinv.
            raise np.linalg.LinAlgError(
                "Modified innovation covariance S_k_mod is singular."
            ) from exc

        # Posterior covariance (using the form from reference WoLF for P_update)
        cov_posterior = self.cov_pred - self.cov_pred.dot(self.C.T).dot(
            S_k_mod_inv
        ).dot(self.C).dot(self.cov_pred)

        # WoLF-specific Kalman Gain
        gain_wolf = w_sq * cov_posterior.dot(self.C.T).dot(self.cov_meas_inv)

        # Measurement residual: residual = y_k - y_hat_prior
        residual = y_k - y_hat_prior

        # Update state estimate
        x_hat_posterior = self.x_pred + gain_wolf.dot(residual)

        # Update filter state
        self.x_hat = x_hat_posterior
        self.cov = cov_posterior

        return self.x_hat
