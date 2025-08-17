# cspsa.py
# 包含 CSPSA 优化算法的实现。

import numpy as np
# 修正：导入正确的、专用于CSPSA的计算函数
from simulation_utils_poisson import calculate_cspsa_violation
from simulation_utils_poisson import calculate_cspsa_chsh_violation

def run_cspsa_simulation(config, run_id=0):
    """
    运行单次 CSPSA (复数) 模拟。
    """
    if run_id > 0 and run_id % 10 == 0:
        print(f"  - [CSPSA] Starting Trial {run_id + 1}...")

    # 解包配置参数
    initial_state = config['state']
    iterations = config['iterations']
    a, s, b, r = config['hparams']['cspsa']
    photon_num = config['photon_num']

    # --- 1. 参数初始化 (复数) ---
    params = np.random.uniform(-1, 1, 10) + 1j * np.random.uniform(-1, 1, 10)
    params /= np.linalg.norm(params)

    # 定义增益序列
    def alpha(k):
        return a / (k + 1.0) ** s

    def c(k):
        return b / (k + 1.0) ** r

    history = []
    # --- 2. 主优化循环 ---
    for k in range(1, iterations + 1):
        # 修正：调用正确的函数
        history.append(calculate_cspsa_violation(initial_state, params, photon_num))

        delta = np.random.choice([1, -1, 1j, -1j], size=10)
        ck = c(k)
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        # 修正：调用正确的函数
        val_plus = calculate_cspsa_violation(initial_state, params_plus, photon_num)
        val_minus = calculate_cspsa_violation(initial_state, params_minus, photon_num)

        gradient = (val_plus - val_minus) / (2 * ck * np.conj(delta))
        params += alpha(k) * gradient
        params /= np.linalg.norm(params)

    # 修正：调用正确的函数
    history.append(calculate_cspsa_violation(initial_state, params, photon_num))
    return history


def run_cspsa_chsh_simulation(config, run_id=0):
    """
    运行单次 CSPSA (复数) 模拟。
    """
    if run_id > 0 and run_id % 10 == 0:
        print(f"  - [CSPSA] Starting Trial {run_id + 1}...")

    # 解包配置参数
    initial_state = config['state']
    iterations = config['iterations']
    a, s, b, r = config['hparams']['cspsa']
    photon_num = config['photon_num']

    # --- 1. 参数初始化 (复数) ---
    params = np.random.uniform(-1, 1, 8) + 1j * np.random.uniform(-1, 1, 8)
    params /= np.linalg.norm(params)

    # 定义增益序列
    def alpha(k):
        return a / (k + 1.0) ** s

    def c(k):
        return b / (k + 1.0) ** r

    history = []
    # --- 2. 主优化循环 ---
    for k in range(1, iterations + 1):
        # 修正：调用正确的函数
        history.append(calculate_cspsa_chsh_violation(initial_state, params, photon_num))

        delta = np.random.choice([1, -1, 1j, -1j], size=8)
        ck = c(k)
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        # 修正：调用正确的函数
        val_plus = calculate_cspsa_chsh_violation(initial_state, params_plus, photon_num)
        val_minus = calculate_cspsa_chsh_violation(initial_state, params_minus, photon_num)

        gradient = (val_plus - val_minus) / (2 * ck * np.conj(delta))
        params += alpha(k) * gradient
        params /= np.linalg.norm(params)

    # 修正：调用正确的函数
    history.append(calculate_cspsa_chsh_violation(initial_state, params, photon_num))
    return history



