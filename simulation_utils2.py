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
    """创建CSPSA投影算符对"""
    psi = c1 * qt.basis(2, 0) + c2 * qt.basis(2, 1)
    if psi.norm() < 1e-9:
        return qt.qzero(2), qt.qeye(2)
    psi = psi.unit()
    return qt.ket2dm(psi), qt.qeye(2) - qt.ket2dm(psi)

def operator_to_probability(
    op,
    rho,
    photonn: int,
    variation: float,
    uncertainty: float
) -> np.ndarray:


    if variation == 0:
        rho_varied = rho
    else:
        theta = np.random.normal(loc=np.pi / 4, scale=variation)
        rho_varied = qt.ket2dm(np.cos(theta) * qt.tensor(qt.basis(2, 0), qt.basis(2, 0))+ np.sin(theta) * qt.tensor(qt.basis(2, 1), qt.basis(2, 1)))

    if uncertainty == 0:
        meas_varied = op
    else:
        h_coeffs = [np.random.normal(0, uncertainty, 3) for _ in range(2)]
        meas_unitary = qt.tensor(*[
            qt.Qobj((-1j * sum(h * qt.Qobj(p) for h, p in zip(hc, [qt.sigmax(), qt.sigmay(), qt.sigmaz()]))).expm())
            for hc in h_coeffs
        ])
        meas_varied = [meas_unitary * mea * meas_unitary.dag() for mea in op]


    photon_num = 0
    while photon_num == 0:
        photon_num = np.random.poisson(photonn)

    outcome_probabilities = [np.real((rho_varied * mea).tr()) for mea in meas_varied]
    prob_sum = np.sum(outcome_probabilities)
    normalized_pvals = np.array(outcome_probabilities) / prob_sum
    sample_counts = np.random.multinomial(n=photon_num, pvals=normalized_pvals)
    proba = sample_counts / photon_num
    return proba


def calculate_instrumental_cspsa_violation(rho, params, photon_num, variation, uncertainty):
    """为CSPSA计算违背值 (基于'bell'算符)。"""
    param_pairs = np.reshape(params, (5, 2))
    [(P_A1, P_A1_perp), (P_A2, P_A2_perp), (P_A3, P_A3_perp),
     (P_B0, P_B0_perp), (P_B1, P_B1_perp)] = [
        create_cspsa_projectors(c1, c2)
        for c1, c2 in param_pairs
    ]

    term1 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term2 = operator_to_probability([qt.tensor(P_A2, P_B0), qt.tensor(P_A2_perp, P_B1), qt.tensor(P_A2, P_B0_perp), qt.tensor(P_A2_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term3 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term4 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term5 = operator_to_probability([qt.tensor(P_A3, P_B0), qt.tensor(P_A3_perp, P_B1), qt.tensor(P_A3, P_B0_perp), qt.tensor(P_A3_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)

    term1_value = np.dot(term1,np.array([1, 1, -1, -1])) # multiplication of vectors
    term2_value = np.dot(term2,np.array([1, 1, -1, -1]))  # multiplication of vectors
    term3_value = np.dot(term3,np.array([1, 1, -1, -1]))  # multiplication of vectors
    term4_value = np.dot(term4,np.array([1, -1, -1, 1]))  # multiplication of vectors
    term5_value = np.dot(term5,np.array([1, -1, -1, 1]))  # multiplication of vectors

    inequ = -term1_value + 2 * term2_value + term3_value - term4_value + 2 * term5_value

    return inequ

def calculate_chsh_cspsa_violation(rho, params, photon_num, variation, uncertainty):
    """为CSPSA计算违背值 (基于'bell'算符)。"""
    param_pairs = np.reshape(params, (4, 2))
    [(P_A0, P_A0_perp), (P_A1, P_A1_perp), (P_B0, P_B0_perp),
      (P_B1, P_B1_perp)] = [
        create_cspsa_projectors(c1, c2)
        for c1, c2 in param_pairs
    ]

    term1 =  operator_to_probability([qt.tensor(P_A0, P_B0), qt.tensor(P_A0_perp, P_B0), qt.tensor(P_A0, P_B0_perp), qt.tensor(P_A0_perp, P_B0_perp)], rho, photon_num, variation, uncertainty)
    term2 =  operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B0), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B0_perp)], rho, photon_num, variation, uncertainty)
    term3 =  operator_to_probability([qt.tensor(P_A0, P_B1), qt.tensor(P_A0_perp, P_B1), qt.tensor(P_A0, P_B1_perp), qt.tensor(P_A0_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term4 =  operator_to_probability([qt.tensor(P_A1, P_B1), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B1_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)

    term1_value = np.dot(term1,np.array([1, -1, -1, 1])) # multiplication of vectors
    term2_value = np.dot(term2,np.array([1, -1, -1, 1])) # multiplication of vectors
    term3_value = np.dot(term3,np.array([1, -1, -1, 1])) # multiplication of vectors
    term4_value = np.dot(term4,np.array([1, -1, -1, 1])) # multiplication of vectors

    inequ = term1_value + term2_value + term3_value - term4_value

    return inequ

# =============================================================================
# --- SGA 相关的函数 (Functions for SGA) ---
# 这是全新的、与您的Mathematica代码完全等价的实现
# =============================================================================

def create_sga_projectors(theta, phi):

    psi_ab = np.cos(theta) * qt.basis(2, 0) + np.exp(-1j * phi) * np.sin(theta) * qt.basis(2, 1)
    P_ab = qt.ket2dm(psi_ab)
    return P_ab, qt.qeye(2) - P_ab

def calculate_instrumental_sga_violation(rho, params, photon_num, variation, uncertainty):
    """
    为SGA计算违背值。
    参数是10个实数，每对(theta, phi)通过一个特殊的映射定义一个测量。
    """
    param_pairs = np.reshape(params, (5, 2))
    [(P_A1, P_A1_perp), (P_A2, P_A2_perp), (P_A3, P_A3_perp),
     (P_B0, P_B0_perp), (P_B1, P_B1_perp)] = [
        create_sga_projectors(theta, phi)
        for theta, phi in param_pairs
    ]

    term1 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term2 = operator_to_probability([qt.tensor(P_A2, P_B0), qt.tensor(P_A2_perp, P_B1), qt.tensor(P_A2, P_B0_perp), qt.tensor(P_A2_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term3 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term4 = operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term5 = operator_to_probability([qt.tensor(P_A3, P_B0), qt.tensor(P_A3_perp, P_B1), qt.tensor(P_A3, P_B0_perp), qt.tensor(P_A3_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)

    term1_value = np.dot(term1,np.array([1, 1, -1, -1])) # multiplication of vectors
    term2_value = np.dot(term2,np.array([1, 1, -1, -1]))  # multiplication of vectors
    term3_value = np.dot(term3,np.array([1, 1, -1, -1]))  # multiplication of vectors
    term4_value = np.dot(term4,np.array([1, -1, -1, 1]))  # multiplication of vectors
    term5_value = np.dot(term5,np.array([1, -1, -1, 1]))  # multiplication of vectors

    inequ = -term1_value + 2 * term2_value + term3_value - term4_value + 2 * term5_value

    return inequ

def calculate_chsh_sga_violation(rho, params, photon_num, variation, uncertainty):
    """为CSPSA计算违背值 (基于'bell'算符)。"""
    param_pairs = np.reshape(params, (4, 2))
    [(P_A0, P_A0_perp), (P_A1, P_A1_perp), (P_B0, P_B0_perp),
      (P_B1, P_B1_perp)] = [
        create_sga_projectors(c1, c2)
        for c1, c2 in param_pairs
    ]

    term1 =  operator_to_probability([qt.tensor(P_A0, P_B0), qt.tensor(P_A0_perp, P_B0), qt.tensor(P_A0, P_B0_perp), qt.tensor(P_A0_perp, P_B0_perp)], rho, photon_num, variation, uncertainty)
    term2 =  operator_to_probability([qt.tensor(P_A1, P_B0), qt.tensor(P_A1_perp, P_B0), qt.tensor(P_A1, P_B0_perp), qt.tensor(P_A1_perp, P_B0_perp)], rho, photon_num, variation, uncertainty)
    term3 =  operator_to_probability([qt.tensor(P_A0, P_B1), qt.tensor(P_A0_perp, P_B1), qt.tensor(P_A0, P_B1_perp), qt.tensor(P_A0_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)
    term4 =  operator_to_probability([qt.tensor(P_A1, P_B1), qt.tensor(P_A1_perp, P_B1), qt.tensor(P_A1, P_B1_perp), qt.tensor(P_A1_perp, P_B1_perp)], rho, photon_num, variation, uncertainty)

    term1_value = np.dot(term1,np.array([1, -1, -1, 1])) # multiplication of vectors
    term2_value = np.dot(term2,np.array([1, -1, -1, 1])) # multiplication of vectors
    term3_value = np.dot(term3,np.array([1, -1, -1, 1])) # multiplication of vectors
    term4_value = np.dot(term4,np.array([1, -1, -1, 1])) # multiplication of vectors

    inequ = term1_value + term2_value + term3_value - term4_value

    return inequ