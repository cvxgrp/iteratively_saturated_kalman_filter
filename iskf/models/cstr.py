# cstr_model.py

import numpy as np
import control


def cascaded_cstr_ss(n=3, dt=0.05):
    # --- Subsystem matrices (MathWorks linear CSTR example) ---
    A_sub = np.array([[-5.0, -0.3427], [47.68, 2.785]])
    B_sub = np.array([[1.0, 0.0], [0.0, 0.3]])
    C_sub = np.array([[0.0, 1.0]])
    # --- Build block‐diagonal A_big and control mapping B_ctrl, C_big ---
    A_big = np.zeros((2 * n, 2 * n))
    B_ctrl = np.zeros((2 * n, 2 * n))
    for i in range(n):
        idx = 2 * i
        A_big[idx : idx + 2, idx : idx + 2] = A_sub
        if i > 0:
            A_big[idx : idx + 2, idx - 2 : idx] = B_sub

        if i > 0:
            B_ctrl[idx : idx + 2, idx : idx + 2] = B_sub
        else:
            B_ctrl[idx : idx + 2, idx : idx + 2] = B_sub

    C_big = np.kron(np.eye(n), C_sub)

    # --- Process‐noise channels (same as control channels) ---
    B_process_noise_big = B_ctrl.copy()
    dim_w_dyn = B_process_noise_big.shape[1]

    # --- Measurement‐noise channels (direct feedthrough) ---
    num_outputs = C_big.shape[0]
    dim_w_meas = num_outputs
    D_meas_noise = np.eye(num_outputs)

    # --- Assemble full noise‐augmented system matrices ---
    B_system = np.hstack((B_process_noise_big, np.zeros((2 * n, dim_w_meas))))
    D_system = np.hstack((np.zeros((num_outputs, dim_w_dyn)), D_meas_noise))

    # --- Create and discretize ---
    sysc = control.ss(A_big, B_system, C_big, D_system)
    sysd = control.c2d(sysc, dt)

    return sysd
