"""
Simulator for a discrete-time LTI system with additive Gaussian noise and an outlier component.
"""

import numpy as np
import control as ct


class Simulator:
    """
    Simulates a discrete-time LTI system with additive Gaussian noise and an outlier component.

    The noise model is:
    noise = N(0, Q_nominal) + OutlierComponent
    where OutlierComponent = N(0, outlier_scale_factor * Q_nominal) with probability p_outlier,
    and OutlierComponent = 0 with probability (1 - p_outlier).

    This applies separately to process noise and measurement noise.
    The covariance matrices Q_nominal for process and measurement noise can be general
    positive semi-definite matrices.
    """

    def __init__(
        self,
        system_model: ct.StateSpace,
        process_noise_cov: np.ndarray,
        measurement_noise_cov: np.ndarray,
        p_outlier_process: float,
        outlier_scale_process: float,
        p_outlier_measurement: float,
        outlier_scale_measurement: float,
    ):
        """
        Initializes the Simulator.

        Args:
            system_model: The control.StateSpace object representing the discrete-time system.
                          It must have system_model.dt defined.
            process_noise_cov: Numpy array for the nominal covariance matrix (Q_proc_nominal)
                               of the process noise. Expected to be positive semi-definite.
                               If empty or indicates zero dimension (e.g. np.array([])), no process noise is assumed.
            measurement_noise_cov: Numpy array for the nominal covariance matrix (Q_meas_nominal)
                                   of the measurement noise. Expected to be positive semi-definite.
                                   If empty or indicates zero dimension (e.g. np.array([])), no measurement noise is assumed.
            p_outlier_process: Probability (0 to 1) of an outlier occurring in the process noise.
            outlier_scale_process: Factor by which the nominal process noise covariance is scaled
                                   for the outlier component. (e.g., 10 means Var(outlier) = 10 * Var(nominal)).
            p_outlier_measurement: Probability (0 to 1) of an outlier occurring in the measurement noise.
            outlier_scale_measurement: Factor by which the nominal measurement noise covariance is scaled
                                       for the outlier component.
        """
        if not isinstance(system_model, ct.StateSpace):
            raise TypeError("system_model must be a control.StateSpace object.")
        if system_model.dt is None:
            raise ValueError(
                "System model must be discrete-time with a defined system_model.dt."
            )

        self.sys = system_model
        self.dt = system_model.dt

        # Process Noise Covariance and Dimension
        self.Q_proc_nominal = np.asarray(process_noise_cov)
        if self.Q_proc_nominal.size == 0 or (
            len(self.Q_proc_nominal.shape) == 1 and self.Q_proc_nominal.shape[0] == 0
        ):  # Handles [], np.array([]), np.empty((0,))
            self.dim_w_proc = 0
            self.Q_proc_nominal = np.empty(
                (0, 0)
            )  # Standardize to 2D empty array for consistency
        else:
            if (
                len(self.Q_proc_nominal.shape) != 2
                or self.Q_proc_nominal.shape[0] != self.Q_proc_nominal.shape[1]
            ):
                raise ValueError(
                    f"Process noise covariance matrix must be a square 2D array. "
                    f"Got shape {self.Q_proc_nominal.shape}."
                )
            self.dim_w_proc = self.Q_proc_nominal.shape[0]

        # Measurement Noise Covariance and Dimension
        self.Q_meas_nominal = np.asarray(measurement_noise_cov)
        if self.Q_meas_nominal.size == 0 or (
            len(self.Q_meas_nominal.shape) == 1 and self.Q_meas_nominal.shape[0] == 0
        ):  # Handles [], np.array([]), np.empty((0,))
            self.dim_w_meas = 0
            self.Q_meas_nominal = np.empty(
                (0, 0)
            )  # Standardize to 2D empty array for consistency
        else:
            if (
                len(self.Q_meas_nominal.shape) != 2
                or self.Q_meas_nominal.shape[0] != self.Q_meas_nominal.shape[1]
            ):
                raise ValueError(
                    f"Measurement noise covariance matrix must be a square 2D array. "
                    f"Got shape {self.Q_meas_nominal.shape}."
                )
            self.dim_w_meas = self.Q_meas_nominal.shape[0]

        expected_total_input_dim = self.dim_w_proc + self.dim_w_meas
        if self.sys.B.shape[1] != expected_total_input_dim:
            raise ValueError(
                f"Total system input dimension sys.B.shape[1] ({self.sys.B.shape[1]}) "
                f"does not match sum of inferred process ({self.dim_w_proc}) and "
                f"measurement ({self.dim_w_meas}) noise dimensions ({expected_total_input_dim})."
            )

        # Store outlier parameters
        self.p_outlier_proc = p_outlier_process
        self.scale_outlier_proc = outlier_scale_process
        self.p_outlier_meas = p_outlier_measurement
        self.scale_outlier_meas = outlier_scale_measurement

        # Validate probabilities and scale factors
        for p_val, name in [
            (p_outlier_process, "p_outlier_process"),
            (p_outlier_measurement, "p_outlier_measurement"),
        ]:
            if not (0 <= p_val <= 1):
                raise ValueError(f"{name} ({p_val}) must be between 0 and 1.")
        for s_val, name in [
            (outlier_scale_process, "outlier_scale_process"),
            (outlier_scale_measurement, "outlier_scale_measurement"),
        ]:
            if s_val < 0:  # Typically s_val >= 1 for "larger variance"
                raise ValueError(f"{name} ({s_val}) must be non-negative.")

    def _generate_noise_sample(
        self,
        dim: int,
        Q_nominal: np.ndarray,
        p_outlier: float,
        outlier_scale_factor: float,
    ) -> np.ndarray:
        """
        Generates a single noise sample according to the mixture model.
        The model is: noise = N(0, Q_nominal) + OutlierComponent, where
        OutlierComponent = N(0, outlier_scale_factor * Q_nominal) with probability p_outlier,
        and OutlierComponent = 0 with probability (1 - p_outlier).

        Args:
            dim: Dimension of the noise vector.
            Q_nominal: Nominal covariance matrix (positive semi-definite) for the base Gaussian component.
            p_outlier: Probability of the outlier component occurring.
            outlier_scale_factor: Scaling factor for the covariance of the outlier component.

        Returns:
            A numpy array representing the generated noise sample.
        """
        mean = np.zeros(dim)

        # 1. Generate the base Gaussian noise
        base_noise = np.random.multivariate_normal(mean, Q_nominal)

        # 2. Generate the additive outlier component
        def multivariate_laplace(A_inv, lam, n_samples=1):
            """
            Draws x ~ p(x) ∝ exp(-lam * ||A x||_2) for invertible square A.
            Takes A_inv directly instead of computing it internally.

            Returns an array of shape (n_samples, n).
            """
            n = A_inv.shape[0]  # A_inv is n x n
            samples = np.zeros((n_samples, n))
            for i in range(n_samples):
                # 1) sample radius r ~ Gamma(shape=n, scale=1/lam)
                r = np.random.gamma(shape=n, scale=1 / lam)
                # 2) sample direction u ~ Uniform on S^{n-1}
                u = np.random.normal(size=n)
                u /= np.linalg.norm(u)
                # 3) form y = r * u, then x = A^{-1} y
                y = r * u
                x = A_inv.dot(y)
                samples[i] = x
            return samples

        outlier_additive_component = np.zeros(dim)
        if outlier_scale_factor > 1e-9 and np.random.rand() < p_outlier:
            outlier_additive_component = np.random.multivariate_normal(
                mean, (outlier_scale_factor**2) * Q_nominal
            )
            return outlier_additive_component

            # Q_inv = np.linalg.inv(Q_nominal)
            # Q_inv_sqrt = np.linalg.cholesky(Q_inv).T
            # A_inv = np.linalg.inv(Q_inv_sqrt)
            # outlier_additive_component = multivariate_laplace(
            #     A_inv, 1.0 / outlier_scale_factor
            # ).flatten()

        return base_noise + outlier_additive_component

    def _generate_process_noise_sample(self) -> np.ndarray:
        """Generates a single sample of process noise."""
        if self.dim_w_proc == 0:
            return np.array([])  # Return empty array if no process noise
        return self._generate_noise_sample(
            self.dim_w_proc,
            self.Q_proc_nominal,
            self.p_outlier_proc,
            self.scale_outlier_proc,
        )

    def _generate_measurement_noise_sample(self) -> np.ndarray:
        """Generates a single sample of measurement noise."""
        if self.dim_w_meas == 0:
            return np.array([])  # Return empty array if no measurement noise
        return self._generate_noise_sample(
            self.dim_w_meas,
            self.Q_meas_nominal,
            self.p_outlier_meas,
            self.scale_outlier_meas,
        )

    def simulate(
        self,
        x0: np.ndarray,
        T_final: float,
        num_steps: int,
        return_noise_inputs: bool = False,
    ):
        """
        Simulates the system with the generated noise over a specified time horizon.

        Args:
            x0: Initial state vector (numpy array).
            T_final: Final time for the simulation.
            num_steps: Total number of time points in the simulation (includes t=0).
                       The simulation runs for (num_steps - 1) * self.dt duration.
            return_noise_inputs: If True, also returns the matrix of generated noise inputs
                                 fed to the system.

        Returns:
            T_out: Numpy array of time points.
            Y_out: Numpy array representing the output trajectory (measurements).
                   Shape is (num_outputs, num_steps).
            X_out: Numpy array representing the state trajectory.
                   Shape is (num_states, num_steps).
            W_input_to_system (optional): Numpy array of combined noise inputs fed to the system.
                                     Shape is (dim_w_proc + dim_w_meas, num_steps).
                                     Returned if return_noise_inputs is True.
        """
        if not isinstance(x0, np.ndarray):
            x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != self.sys.A.shape[0]:
            raise ValueError(
                f"Initial state x0 dimension ({x0.shape[0]}) does not match "
                f"system state dimension ({self.sys.A.shape[0]})."
            )
        if num_steps <= 1:
            raise ValueError("num_steps must be greater than 1 for simulation.")

        # Generate the time vector for the simulation
        # control.forced_response for discrete systems uses T as time points
        time_vector = np.linspace(0, T_final, num_steps, endpoint=True)

        # Ensure that the generated time_vector is consistent with system's dt if num_steps and T_final are chosen
        # such that T_final / (num_steps - 1) is very close to self.dt.
        # For simplicity, we assume the user provides a T_final and num_steps that make sense
        # or understands that forced_response will use self.sys.dt for propagation.

        # Generate sequences of noise
        # Process noise:
        list_U_proc = []
        if self.dim_w_proc > 0:
            for _ in range(num_steps):
                list_U_proc.append(self._generate_process_noise_sample())
            # Transpose to get shape (dim_w_proc, num_steps)
            U_proc_noise = np.array(list_U_proc).T
        else:
            # No process noise, create an empty array with correct second dimension
            U_proc_noise = np.empty((0, num_steps))

        # Measurement noise:
        list_U_meas = []
        if self.dim_w_meas > 0:
            for _ in range(num_steps):
                list_U_meas.append(self._generate_measurement_noise_sample())
            # Transpose to get shape (dim_w_meas, num_steps)
            U_meas_noise = np.array(list_U_meas).T
        else:
            # No measurement noise, create an empty array with correct second dimension
            U_meas_noise = np.empty((0, num_steps))

        # Combine noise inputs as expected by the system's B and D matrices.
        # The system input w is [w_proc, w_meas]^T.
        W_input_to_system = np.vstack([U_proc_noise, U_meas_noise])

        # Validate shape of the combined noise input
        if W_input_to_system.shape[0] != self.sys.B.shape[1]:
            # This should not happen if __init__ checks passed and logic is correct
            raise RuntimeError(
                f"Internal error: Constructed noise input W dimension {W_input_to_system.shape[0]} "
                f"does not match system input dimension {self.sys.B.shape[1]}."
            )
        if W_input_to_system.shape[1] != num_steps:
            # This should not happen if loop range is correct
            raise RuntimeError(
                f"Internal error: Constructed noise input W time steps {W_input_to_system.shape[1]} "
                f"does not match requested num_steps {num_steps}."
            )

        # Simulate the system response using control.forced_response
        # For discrete-time systems, T is the list of time points at which the input is defined
        # and at which the output is desired.
        # U is the input array, with U[:, i] being the input at time T[i].
        T_out, Y_out, X_out = ct.forced_response(
            self.sys,
            T=time_vector,
            U=W_input_to_system,
            X0=x0,
            return_x=True,  # Ensures state trajectory X_out is returned
        )

        if return_noise_inputs:
            return T_out, Y_out, X_out, W_input_to_system
        else:
            return T_out, Y_out, X_out
