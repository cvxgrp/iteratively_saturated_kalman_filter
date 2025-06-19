#!/usr/bin/env python3
"""
Test script to verify the README sample code works correctly.
"""

import numpy as np
from iskf.models.vehicle import vehicle_ss
from iskf.filters import SteadyTwoStepIterSatFilter
from iskf.simulator import Simulator


def test_readme_example():
    """Test the fixed README sample code."""
    print("Testing README sample code...")

    # 1. Create system model
    model = vehicle_ss(gamma=0.05, dt=0.05)
    print("✓ Created vehicle model")

    # 2. Setup simulation with outlier parameters
    sim = Simulator(
        system_model=model,
        process_noise_cov=np.eye(2) * 10,
        measurement_noise_cov=np.eye(2) * 5,
        p_outlier_process=0.1,
        outlier_scale_process=10.0,
        p_outlier_measurement=0.1,
        outlier_scale_measurement=10.0,
    )
    print("✓ Created simulator with outlier parameters")

    # 3. Generate data
    time_step = 0.05
    num_steps = 1000
    T_final = (num_steps - 1) * time_step

    T_out, Y_meas, X_true, _ = sim.simulate(
        x0=np.array([0, 0, 5, 5]),
        T_final=T_final,
        num_steps=num_steps,
        return_noise_inputs=True,
    )
    print(
        f"✓ Generated simulation data: T={T_out.shape}, Y={Y_meas.shape}, X={X_true.shape}"
    )

    # 4. Create and run filter
    filter_instance = SteadyTwoStepIterSatFilter(
        system_model=model,
        cov_input=np.eye(2) * 10,
        cov_measurement=np.eye(2) * 5,
        coef_s=1.345,
        coef_o=1.345,
        step_size=1.0,
    )
    print("✓ Created filter instance")

    X_est = filter_instance.estimate(
        T_out, Y_meas, x_initial_estimate=np.array([0, 0, 5, 5]), P_initial=np.eye(4)
    )
    print(f"✓ Ran filter estimation: X_est={X_est.shape}")

    # 5. Evaluate performance
    from iskf.metrics import rmse

    error = rmse(X_est, X_true[:, 1:])  # Align dimensions
    print(f"✓ Calculated RMSE: {error:.4f}")

    print("\n🎉 README sample code runs successfully!")
    return True


if __name__ == "__main__":
    test_readme_example()
