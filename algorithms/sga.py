# sga.py
# 包含了最终修正后的SGA优化算法实现。

import numpy as np
# 导入正确的、专用于SGA的计算函数
from simulation_utils import calculate_sga_violation
from simulation_utils import calculate_sga_chsh_violation

def run_sga_simulation(config, run_id=0):
    """
    运行单次SGA（实数）模拟。
    该实现现在精确匹配您的Mathematica代码逻辑。
    """
    if run_id > 0 and run_id % 10 == 0:
        print(f"  - [SGA] Starting Trial {run_id + 1}...")

    # 解包配置参数
    initial_state = config['state']
    iterations = config['iterations']
    a, s, b, r = config['hparams']['sga']

    # --- 1. 参数初始化 (10个实数) ---
    # 与您的Mathematica代码一致，从随机实数开始
    params = np.random.uniform(-1, 1, 10)

    # 定义增益序列
    def alpha(k):
        return a / (k + 1.0 + 0.0) ** s

    def c(k):
        return b / (k + 1.0) ** r

    history = []
    # --- 2. 主优化循环 ---
    for k in range(1, iterations + 1):
        # 使用为SGA设计的、正确的违背计算函数
        history.append(calculate_sga_violation(initial_state, params))

        delta = np.random.choice([-1, 1], size=10)
        ck = c(k)
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        val_plus = calculate_sga_violation(initial_state, params_plus)
        val_minus = calculate_sga_violation(initial_state, params_minus)

        # SPSA/SGA 的梯度更新方式
        gradient = (val_plus - val_minus) / (2 * ck)
        params += alpha(k) * gradient * delta

        # 移除对角度范围的错误限制，让优化器自由探索参数。
        # 这是之前版本中的一个错误。

    history.append(calculate_sga_violation(initial_state, params))
    return history


def run_sga_chsh_simulation(config, run_id=0):
    """
    运行单次SGA（实数）模拟。
    该实现现在精确匹配您的Mathematica代码逻辑。
    """
    if run_id > 0 and run_id % 10 == 0:
        print(f"  - [SGA] Starting Trial {run_id + 1}...")

    # 解包配置参数
    initial_state = config['state']
    iterations = config['iterations']
    a, s, b, r = config['hparams']['sga']

    # --- 1. 参数初始化 (8个实数) ---
    # 与您的Mathematica代码一致，从随机实数开始
    params = np.random.uniform(-1, 1, 8)

    # 定义增益序列
    def alpha(k):
        return a / (k + 1.0 + 0.0) ** s

    def c(k):
        return b / (k + 1.0) ** r

    history = []
    # --- 2. 主优化循环 ---
    for k in range(1, iterations + 1):
        # 使用为SGA设计的、正确的违背计算函数
        history.append(calculate_sga_chsh_violation(initial_state, params))

        delta = np.random.choice([-1, 1], size=8)
        ck = c(k)
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        val_plus = calculate_sga_chsh_violation(initial_state, params_plus)
        val_minus = calculate_sga_chsh_violation(initial_state, params_minus)

        # SPSA/SGA 的梯度更新方式
        gradient = (val_plus - val_minus) / (2 * ck)
        params += alpha(k) * gradient * delta

        # 移除对角度范围的错误限制，让优化器自由探索参数。
        # 这是之前版本中的一个错误。

    history.append(calculate_sga_chsh_violation(initial_state, params))
    return history