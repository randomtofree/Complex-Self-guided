# simulation_utils.py
# 包含了为SGA和CSPSA两种不同模型构建算符的核心函数。
# 修正版：SGA和CSPSA的实现现在完全独立且各自正确。

import numpy as np
import qutip as qt

#定义一些基层关键参数
#return 的数据考不考虑poisson噪声, 考不考虑shot noise， 考不考虑measurement misaligin， 考不考虑uncertainty。
# val = np.random.binomial(np.random.poisson(1000), np.real((I_op * rho).tr()))
# =============================================================================
# --- CSPSA 相关的函数 (Functions for CSPSA) ---
# 这部分是正确的，保持不变
# =============================================================================
def create_cspsa_projectors(c1, c2):
    """为CSPSA创建投影算符对 (基于复数参数)。"""
    psi = c1 * qt.basis(2, 0) + c2 * qt.basis(2, 1)
    if psi.norm() < 1e-9:
        return qt.qzero(2), qt.qeye(2)
    psi = psi.unit()
    P = qt.ket2dm(psi)
    P_perp = qt.qeye(2) - P
    return P, P_perp

def operator_to_probability(op, rho, photonn):
    """计算算符的期望值作为概率。"""
    photon_num = np.random.poisson(photonn)
    proba = np.random.binomial(photon_num, np.real((op * rho).tr()))/ photon_num
    return proba

def calculate_cspsa_violation(rho, params, photonn_set):
    """为CSPSA计算违背值 (基于'bell'算符)。"""
    param_pairs = np.reshape(params, (5, 2))
    P_A1, P_A1_perp = create_cspsa_projectors(param_pairs[0][0], param_pairs[0][1])
    P_A2, P_A2_perp = create_cspsa_projectors(param_pairs[1][0], param_pairs[1][1])
    P_A3, P_A3_perp = create_cspsa_projectors(param_pairs[2][0], param_pairs[2][1])
    P_B0, P_B0_perp = create_cspsa_projectors(param_pairs[3][0], param_pairs[3][1])
    P_B1, P_B1_perp = create_cspsa_projectors(param_pairs[4][0], param_pairs[4][1])

    term1_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B1_perp), rho, photonn_set)
    term2_op = operator_to_probability(qt.tensor(P_A2, P_B0), rho, photonn_set) + operator_to_probability(qt.tensor(P_A2_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A2, P_B0_perp), rho, photonn_set) - operator_to_probability(qt.tensor(P_A2_perp,
                                                                                                            P_B1_perp), rho, photonn_set)
    term3_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B1_perp), rho, photonn_set)
    term4_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B1_perp), rho, photonn_set)
    term5_op = operator_to_probability(qt.tensor(P_A3, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A3_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A3, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A3_perp,
                                                                                                            P_B1_perp), rho, photonn_set)


    I_op = -term1_op + 2 * term2_op + term3_op - term4_op + 2 * term5_op

    return I_op


def calculate_cspsa_chsh_violation(rho, params, photonn_set):
    """为CSPSA计算违背值 (基于'bell'算符)。"""
    param_pairs = np.reshape(params, (4, 2))
    P_A0, P_A0_perp = create_cspsa_projectors(param_pairs[0][0], param_pairs[0][1])
    P_A1, P_A1_perp = create_cspsa_projectors(param_pairs[1][0], param_pairs[1][1])
    P_B0, P_B0_perp = create_cspsa_projectors(param_pairs[2][0], param_pairs[2][1])
    P_B1, P_B1_perp = create_cspsa_projectors(param_pairs[3][0], param_pairs[3][1])

    term1_op = (operator_to_probability(qt.tensor(P_A0, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0_perp, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A0_perp,
                                                                                                            P_B0_perp), rho, photonn_set))
    term2_op = (operator_to_probability(qt.tensor(P_A0, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0, P_B1_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A0_perp,
                                                                                                            P_B1_perp), rho, photonn_set))
    term3_op = (operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B0_perp), rho, photonn_set))
    term4_op = (operator_to_probability(qt.tensor(P_A1, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B1_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B1_perp), rho, photonn_set))

    I_op = term1_op + term2_op + term3_op - term4_op
    return I_op

# =============================================================================
# --- SGA 相关的函数 (Functions for SGA) ---
# 这是全新的、与您的Mathematica代码完全等价的实现
# =============================================================================

def create_sga_projector_pair(theta, phi):
    """
    为SGA创建一对正交投影算符 (P, P_perp)。
    这完全等价于您Mathematica代码中的 `ab` 和 `cd` 函数的定义。
    """
    # 增加数值稳定性检查，防止 cos(theta) 接近于零
    if np.abs(np.cos(theta)) < 1e-9:
        # 在这种奇异点，返回一个默认的正交对，例如沿着Z轴
        P = qt.ket2dm(qt.basis(2, 0))
        P_perp = qt.ket2dm(qt.basis(2, 1))
        return P, P_perp

    # 翻译 'ab' 函数来构建第一个投影算符 P_ab
    psi_ab = np.cos(theta) * qt.basis(2, 0) + np.exp(1j * phi) * np.sin(theta) * qt.basis(2, 1)
    P_ab = qt.ket2dm(psi_ab)

    # 翻译 'cd' 函数来构建第二个投影算符 P_cd
    psi_cd = np.sin(theta) * qt.basis(2, 0) - np.exp(1j * phi) * np.cos(theta) * qt.basis(2, 1)
    P_cd = qt.ket2dm(psi_cd)

    # 返回由ab和cd函数定义的投影算符对
    return P_ab, P_cd


def calculate_sga_violation(rho, params, photonn_set):
    """
    为SGA计算违背值。
    参数是10个实数，每对(theta, phi)通过一个特殊的映射定义一个测量。
    """
    param_pairs = np.reshape(params, (5, 2))

    # --- 1. 使用SGA特定的函数创建投影算符 ---
    # 这里的 P, P_perp 是根据您Mathematica代码中的 ab 和 cd 函数生成的
    P_A1, P_A1_perp = create_sga_projector_pair(param_pairs[0][0], param_pairs[0][1])
    P_A2, P_A2_perp = create_sga_projector_pair(param_pairs[1][0], param_pairs[1][1])
    P_A3, P_A3_perp = create_sga_projector_pair(param_pairs[2][0], param_pairs[2][1])
    P_B0, P_B0_perp = create_sga_projector_pair(param_pairs[3][0], param_pairs[3][1])
    P_B1, P_B1_perp = create_sga_projector_pair(param_pairs[4][0], param_pairs[4][1])

    # --- 2. 构建与您SGA Mathematica代码中 'inst' 函数等价的算符 ---
    # 这个结构与CSPSA的'bell'算符是相同的
    term1_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) + operator_to_probability(
        qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho,
                                                                                photonn_set) - operator_to_probability(
        qt.tensor(P_A1_perp,
                  P_B1_perp), rho, photonn_set)
    term2_op = operator_to_probability(qt.tensor(P_A2, P_B0), rho, photonn_set) + operator_to_probability(
        qt.tensor(P_A2_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A2, P_B0_perp), rho,
                                                                                photonn_set) - operator_to_probability(
        qt.tensor(P_A2_perp,
                  P_B1_perp), rho, photonn_set)
    term3_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) + operator_to_probability(
        qt.tensor(P_A1, P_B0_perp), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho,
                                                                                photonn_set) - operator_to_probability(
        qt.tensor(P_A1_perp,
                  P_B1_perp), rho, photonn_set)
    term4_op = operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) - operator_to_probability(
        qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho,
                                                                                photonn_set) + operator_to_probability(
        qt.tensor(P_A1_perp,
                  P_B1_perp), rho, photonn_set)
    term5_op = operator_to_probability(qt.tensor(P_A3, P_B0), rho, photonn_set) - operator_to_probability(
        qt.tensor(P_A3_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A3, P_B0_perp), rho,
                                                                                photonn_set) + operator_to_probability(
        qt.tensor(P_A3_perp,
                  P_B1_perp), rho, photonn_set)

    I_op = -term1_op + 2 * term2_op + term3_op - term4_op + 2 * term5_op

    return I_op

def calculate_sga_chsh_violation(rho, params, photonn_set):
    """
    为SGA计算违背值。
    参数是10个实数，每对(theta, phi)通过一个特殊的映射定义一个测量。
    """
    param_pairs = np.reshape(params, (4, 2))

    # --- 1. 使用SGA特定的函数创建投影算符 ---
    # 这里的 P, P_perp 是根据您Mathematica代码中的 ab 和 cd 函数生成的
    P_A0, P_A0_perp = create_sga_projector_pair(param_pairs[0][0], param_pairs[0][1])
    P_A1, P_A1_perp = create_sga_projector_pair(param_pairs[1][0], param_pairs[1][1])
    P_B0, P_B0_perp = create_sga_projector_pair(param_pairs[2][0], param_pairs[2][1])
    P_B1, P_B1_perp = create_sga_projector_pair(param_pairs[3][0], param_pairs[3][1])

    # --- 2. 构建与您SGA Mathematica代码中 'inst' 函数等价的算符 ---
    # 这个结构与CSPSA的'bell'算符是相同的
    term1_op = (operator_to_probability(qt.tensor(P_A0, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0_perp, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A0_perp,
                                                                                                            P_B0_perp), rho, photonn_set))
    term2_op = (operator_to_probability(qt.tensor(P_A0, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A0, P_B1_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A0_perp,
                                                                                                            P_B1_perp), rho, photonn_set))
    term3_op = (operator_to_probability(qt.tensor(P_A1, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B0), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B0_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B0_perp), rho, photonn_set))
    term4_op = (operator_to_probability(qt.tensor(P_A1, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1_perp, P_B1), rho, photonn_set) - operator_to_probability(qt.tensor(P_A1, P_B1_perp), rho, photonn_set) + operator_to_probability(qt.tensor(P_A1_perp,
                                                                                                            P_B1_perp), rho, photonn_set))

    I_op = term1_op + term2_op + term3_op - term4_op

    return I_op