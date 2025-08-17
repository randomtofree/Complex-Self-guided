# cspsa.py
# 包含 CSPSA 优化算法的实现。

import numpy as np
from simulation_utils2 import (
    calculate_instrumental_sga_violation,
    calculate_chsh_sga_violation,

)


# 定义增益序列

def run_instrumental_sga_simulation(config, run_id=0):
    if run_id > 0 and run_id % 10 == 0:
        print(f"  - [CSPSA] Starting Trial {run_id + 1}...")

    initial_state = config['state']
    iterations = config['iterations']
    hparams = config['hparams']['sga']
    a, s, b, r = hparams['a'], hparams['s'], hparams['b'], hparams['r']
    photon_num = config['photon_num']
    variation = config['state_variation']
    uncertainty = config['uncertainty']

    params = np.pi * np.random.uniform(-1 , 1, 10)
    history = []

    for k in range(1, iterations + 1):
        c_k = b / (k + 1.0) ** r
        a_k = a / (k + 1.0) ** s

        history.append(calculate_instrumental_sga_violation(initial_state, params, photon_num, variation, uncertainty))
        delta = np.random.choice([-1, 1], size=10)
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        val_plus = calculate_instrumental_sga_violation(initial_state, params_plus, photon_num, variation, uncertainty)
        val_minus = calculate_instrumental_sga_violation(initial_state, params_minus, photon_num, variation, uncertainty)

        gradient = (val_plus - val_minus) / (2 * c_k)
        params += a_k * gradient* delta

    history.append(calculate_instrumental_sga_violation(initial_state, params, photon_num, variation, uncertainty))
    return history


def run_chsh_sga_simulation(config, run_id=0):
    if run_id > 0 and run_id % 20 == 0:
        print(f"  - [CSPSA] Starting Trial {run_id + 1}...")

    initial_state = config['state']
    iterations = config['iterations']
    hparams = config['hparams']['sga']
    a, s, b, r = hparams['a'], hparams['s'], hparams['b'], hparams['r']
    photon_num = config['photon_num']
    variation = config['state_variation']
    uncertainty = config['uncertainty']

    params = np.pi * np.random.uniform(-1, 1, 8)
    #params = np.array([0, 0, np.pi/4, 0, np.pi/8,  0, np.pi/8, np.pi],dtype=np.float64)

    history = []

    for k in range(1, iterations + 1):
        c_k = b / (k + 1.0) ** r
        a_k = a / (k + 1.0) ** s

        history.append(calculate_chsh_sga_violation(initial_state, params, photon_num, variation, uncertainty))
        delta = np.random.choice([-1, 1], size=8)
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        val_plus = calculate_chsh_sga_violation(initial_state, params_plus, photon_num, variation, uncertainty)
        val_minus = calculate_chsh_sga_violation(initial_state, params_minus, photon_num, variation, uncertainty)

        gradient = (val_plus - val_minus) / (2 * c_k)
        params += a_k * gradient* delta

    history.append(calculate_chsh_sga_violation(initial_state, params, photon_num, variation, uncertainty))
    return history