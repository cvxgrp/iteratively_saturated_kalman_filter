"""
Base class for filtering algorithms.

This module defines an abstract base class `BaseFilter` that provides a common
interface for various filtering algorithms. Filters are designed to estimate the
internal state of a dynamic system based on a sequence of noisy measurements.
"""

from abc import ABC, abstractmethod
from typing import Optional
from typing_extensions import Self

import numpy as np
import control
import scipy.linalg as la


class BaseFilter(ABC):
    """
    Abstract base class for filters.

    This class provides a common interface for various filtering algorithms
    that estimate the state of a dynamic system given a series of measurements.

    This filter estimates the state of a system modeled as:
        x[k+1] = A x[k] + B u[k]
        y[k]   = C x[k] + v[k]

    where the process noise is w[k] = B u[k].

    Attributes:
        system_model (control.StateSpace): The state-space model of the system.
        A (np.ndarray): State transition matrix.
        B (np.ndarray): Input matrix (for noise inputs in this context).
        C (np.ndarray): Output matrix.
        nx (int): Number of states.
        nu (int): Number of inputs, and is the size of w_dyn
        ny (int): Number of outputs.
        x_hat (np.ndarray | None): The current state estimate, shape (nx,).
                                   Initialized to None and set by `reset`.
        Q_covariance (np.ndarray): Process noise covariance matrix, shape (nx, nx).
        R_covariance (np.ndarray): Measurement noise covariance matrix, shape
        (ny, ny).
    """

    def __init__(
        self,
        system_model: control.StateSpace,
        cov_input: np.ndarray,
        cov_measurement: np.ndarray,
    ):
        """
        Initialize the BaseFilter with system model and noise covariance matrices.

        The initial state estimate is set via the `reset` method, typically called by `estimate`.

        Args:
            system_model: A `control.StateSpace` object representing the discrete-time system dynamics:
                x[k+1] = A x[k] + B w[k]
                y[k]   = C x[k] + D w[k]
                where w[k] is the combined process and measurement noise vector.
            cov_input: Process noise covariance matrix for the dynamic noise component that enters
                the system via the B matrix. This is the preferred way to specify process noise.
            cov_measurement: Measurement noise covariance matrix, shape (ny, ny).
        """
        self.system_model = system_model

        self.nx = system_model.nstates  # Number of system states
        self.ny = system_model.noutputs  # Number of system outputs
        self.nu = cov_input.shape[0]  # Dimension of the noise input

        self.A = np.asarray(system_model.A, dtype=np.float64)
        self.B = np.asarray(system_model.B[:, : self.nu], dtype=np.float64)
        self.C = np.asarray(system_model.C, dtype=np.float64)

        self.cov_input = cov_input
        self.cov_process_noise = self.B.dot(cov_input).dot(self.B.T)
        self.cov_measurement = cov_measurement

        # pre-compute useful quantities
        try:
            self.cov_steady = la.solve_discrete_are(
                self.A.T, self.C.T, self.cov_process_noise, self.cov_measurement
            )
            self.cov_inv_steady = np.linalg.inv(self.cov_steady)
            self.cov_inv_sqrt_steady = np.linalg.cholesky(self.cov_inv_steady).T
            self.cov_pred_steady = (
                self.A.dot(self.cov_steady).dot(self.A.T) + self.cov_process_noise
            )
            self.cov_pred_inv_steady = np.linalg.inv(self.cov_pred_steady)
            self.cov_pred_inv_sqrt_steady = np.linalg.cholesky(
                self.cov_pred_inv_steady
            ).T

            s_matrix = (
                self.C.dot(self.cov_pred_steady).dot(self.C.T) + self.cov_measurement
            )
            self.gain_steady = self.cov_pred_steady.dot(self.C.T).dot(
                np.linalg.inv(s_matrix)
            )

            self.cov_meas_inv = np.linalg.inv(self.cov_measurement)
            self.cov_meas_inv_sqrt = np.linalg.cholesky(self.cov_meas_inv).T

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Non-invertible covariance matrix. Error: {e}") from e

        # Initialize the current state estimate to None. It will be set by reset().
        self.x_hat: np.ndarray | None = None
        # Private attribute to store the current error covariance matrix
        self.cov: np.ndarray | None = None
        # Stores the P_initial provided during the last definitive reset for future default use.
        self.cov_initial_on_reset: np.ndarray | None = None

        # Stores the output of the prediction step
        self.x_pred: np.ndarray | None = None
        self.cov_pred: np.ndarray | None = None

    def new_with_kwargs_like(self, **kwargs) -> Self:
        """Create a copy of the filter with the given parameters."""
        return self.__class__(
            system_model=self.system_model,
            cov_input=self.cov_input,
            cov_measurement=self.cov_measurement,
            **kwargs,
        )

    def reset(
        self, x_initial_estimate: np.ndarray, P_initial: Optional[np.ndarray] = None
    ):
        """
        Resets the filter's current state estimate and error covariance.

        Args:
            x_initial_estimate: The new initial estimate for the state vector, shape (nx,).
            P_initial (optional): The new initial estimate for the error covariance matrix.
                                  If not provided, uses the previously stored value.
                                  Shape (nx, nx).
        """
        if x_initial_estimate.shape != (self.nx,):
            raise ValueError(
                f"x_initial_estimate must have shape ({self.nx},) "
                f"but got {x_initial_estimate.shape}."
            )
        self.x_hat = np.copy(x_initial_estimate)

        # Use the provided P_initial or fall back to previously stored value
        if P_initial is not None:
            # Validate P_initial if provided
            if not isinstance(P_initial, np.ndarray) or P_initial.shape != (
                self.nx,
                self.nx,
            ):
                raise ValueError(
                    f"P_initial must be a NumPy array with shape ({self.nx}, {self.nx}), "
                    f"but got {P_initial.shape if isinstance(P_initial, np.ndarray) else type(P_initial)}."
                )
            # Store for future use and set the P property
            self.cov_initial_on_reset = np.copy(P_initial)
            self.cov = np.copy(P_initial)
        elif self.cov_initial_on_reset is not None:
            # Use the stored value
            self.cov = np.copy(self.cov_initial_on_reset)
        else:
            # No P_initial available
            raise ValueError(
                "P_initial must be provided to reset() at least once to initialize "
                "the error covariance, as no previously stored P_initial is available."
            )

    @abstractmethod
    def predict_step(self, dt: float) -> None:
        """
        Performs the prediction step of the filtering process.
        - Sets self.x_pred and self.cov_pred

        This method is called at each time step k. It uses the state estimate from the previous time step (k-1)
        and the system model to predict the state estimate for the current time
        step k.

        Args:
            dt: The time duration between the previous time step (k-1) and the
                current time step k. (T[k] - T[k-1]).

        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def update_step(self, y_k: np.ndarray, dt: float) -> None:
        """
        Performs the update/correction step of the filtering process.
        - Sets self.x_hat and self.cov

        This method is called at each time step k. It uses the measurement y_k
        taken at time k, and the time elapsed dt since the last update, to
        compute a new state estimate for time k. The filter's internal state
        (self.x_hat and any other relevant quantities like covariance) should be updated.

        Args:
            y_k: The measurement vector received at the current time step k.
                 Shape (ny,).
            dt: The time duration between the previous time step (k-1) and the
                current time step k. (T[k] - T[k-1]).

        Returns:
            The updated state estimate (x_hat_k) after processing y_k. Shape (nx,).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def estimate(
        self,
        T_out: np.ndarray,
        Y_out: np.ndarray,
        x_initial_estimate: np.ndarray,
        P_initial: np.ndarray,
    ) -> np.ndarray:
        """
        Processes a sequence of measurements from a simulation to generate state estimates.

        This method iterates through time, applying the filter's `update` method
        at each step. It initializes the filter state using `x_initial_estimate` and `P_initial`.

        The simulation outputs are typically obtained from `control.forced_response`:
        `T_out, Y_out, X_out = control.forced_response(sys, T, U, X0)`
        where `x_initial_estimate` would be `X_out[:, 0]` and `P_initial` would be
        the desired initial error covariance for the filter.

        Args:
            T_out: Time vector from simulation. Shape (N,). N is the number of time points.
            Y_out: Output (measurement) matrix from simulation. Shape (ny, N).
                   Y_out[:, k] is the measurement at time T_out[k].
            x_initial_estimate: The initial state estimate for the system at time T_out[0].
                                Shape (nx,). Used to initialize/reset the filter's state.
            P_initial: The initial error covariance matrix for the filter. Shape (nx, nx).
                       Used to initialize/reset the filter's covariance.

        Returns:
            X_hat_sequence: A matrix containing the sequence of state estimates.
                            Shape (nx, N-1).
                            X_hat_sequence[:, i] is the filter's estimate of the state
                            at time T_out[i+1], computed using measurement Y_out[:, i+1].
                            This corresponds to estimates for the true states X_true[:, 1:],
                            so it contains N-1 state vectors.
        """
        # Validate input dimensions and types
        N = T_out.shape[0]

        # If there are no time steps or only one, we cannot produce estimates for X_out[:, 1:]
        # (which would require at least two time points: T_out[0] and T_out[1]).
        if N <= 1:
            # Ensure self.x_hat is initialized so its dtype can be queried,
            # even if no estimates are generated.
            if self.x_hat is None:  # Should be set by reset if N > 1
                self.reset(
                    x_initial_estimate, P_initial
                )  # Initialize for dtype consistency
            # If self.x_hat is still None (e.g. if reset failed or wasn't called), default dtype
            dtype_to_use = self.x_hat.dtype if self.x_hat is not None else float
            return np.empty((self.nx, 0), dtype=dtype_to_use)

        # Reset the filter's state and covariance to the initial values.
        # self.x_hat is now x_initial_estimate, representing the estimate at T_out[0].
        # Subclass reset methods are expected to handle P_initial.
        self.reset(x_initial_estimate, P_initial)

        # Ensure x_hat is initialized by reset for dtype usage
        if self.x_hat is None:
            raise RuntimeError("Filter state x_hat was not initialized by reset().")

        # We will generate N-1 state estimates, corresponding to times T_out[1]...T_out[N-1]
        num_estimates_to_generate = N - 1

        # Pre-allocate array for estimates for efficiency and defined dtype.
        X_hat_sequence = np.empty(
            (self.nx, num_estimates_to_generate), dtype=self.x_hat.dtype
        )

        # Iterate from the first measurement *after* the initial state.
        # The first estimate will be for time T_out[1] using Y_out[:, 1].
        for i in range(num_estimates_to_generate):
            # k_sim_idx is the index in T_out, Y_out for the current measurement time point.
            # It goes from 1 to N-1.
            k_sim_idx = i + 1

            current_measurement = Y_out[:, k_sim_idx]
            # Calculate time step duration for the current interval.
            dt = T_out[k_sim_idx] - T_out[k_sim_idx - 1]

            # predict step
            self.predict_step(dt)

            # update / correction step
            self.update_step(current_measurement, dt)

            # store the estimate
            X_hat_sequence[:, i] = self.x_hat

        return X_hat_sequence

    def predict_measurements(
        self,
        T_out: np.ndarray,
        Y_out: np.ndarray,
        x_initial_estimate: np.ndarray,
        P_initial: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts the measurements for a given state trajectory. First calls the
        estimate method to get the state trajectory, then uses the system
        model's C matrix to project the state trajectory onto the measurement space.
        """
        X_hat = self.estimate(T_out, Y_out, x_initial_estimate, P_initial)
        Y_hat = self.C.dot(self.A).dot(
            np.hstack([x_initial_estimate.reshape(-1, 1), X_hat[:, :-1]])
        )
        return Y_hat
