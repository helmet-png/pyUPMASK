"""GUMM：在天空上擬合「一個 2D 高斯（星團）＋一個均勻分布（場星）」。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
外圈每次內圈後都會用它清一次：通過 Ripley's K 檢定的星裡，仍可能混著
天空上離星團中心很遠的場星。把它們的天空位置擬合成
  密度 = π_g × 高斯(中心 μ, 共變異 Σ) + π_u × 均勻(在 [0,1]² 上，密度 1)
用 EM 演算法（期望值最大化）反覆更新參數；最後每顆星得到
「屬於高斯成分的機率」gumm_p。gumm_p 很低的星被當成場星丟掉。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
  numpy（np）   np.random.uniform（隨機初值）、np.eye、np.dot、np.log
  scipy.stats.multivariate_normal
                多維常態分布；.pdf(X) 算每個點的機率密度

======================================================================
【(b) 用到的參數與意義】
======================================================================
  clust_xy     要擬合的天空座標（[0,1]）
  n_epochs     EM 最多跑幾次（預設 1000）
  stable_per   收斂判定：概似「穩定」的次數達到 n_epochs × 0.1 = 100 次就停
  參數：π_g、π_u（兩個成分的權重）、μ（高斯中心）、cov（高斯共變異矩陣）

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 50–87 行｜GUMMProbs()：EM 主迴圈
  核心 2｜第 103–128 行｜expectation_step()：算每顆星屬於高斯／均勻的機率
  核心 3｜第 131–158 行｜maximization_step()：用這些機率更新 μ、cov、權重

======================================================================
【(d) 整體流程】
======================================================================
  隨機初值：μ 在 [0.1, 0.9]、cov 對角線在 [0.1, 0.9]、兩個權重各 0.5
    → 反覆：
        E 步：每顆星 γ_g = π_g·高斯密度 / (π_g·高斯密度 + π_u·1)
        M 步：μ = γ_g 加權平均位置；cov = γ_g 加權共變異；π_g = 平均 γ_g
        概似變化夠小的次數累積到 100 次 → 停
    → 回傳每顆星的 γ_g
⚠ 程式註解寫「Convergence check 1%」，實際判斷式用的是 0.1（10%）。
"""

import numpy as np
from scipy.stats import multivariate_normal


# ═══════════════ 核心 1：EM 主迴圈 ═══════════════
def GUMMProbs(clust_xy, n_epochs=1000, stable_per=.1):
    """
    Fit a model composed of a 2D Gaussian and a 2D uniform distribution in a
    square region with [0., 1.] range.

    Based on the GMM model implementation shown in:
    https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
    """
    cluster = initialize_cluster()

    lkl_old, nstable = -np.inf, 0
    for i in range(n_epochs):

        # ↓ E 步：用目前參數算每顆星屬於高斯的機率
        expectation_step(clust_xy, cluster)
        # ↓ M 步：用這些機率重新估參數
        maximization_step(clust_xy, cluster)

        likelihood = cluster['likelihood']

        # Convergence check 1%
        # ↓ 跟目前最佳概似的相對差小於 0.1 就算「穩定」一次
        if abs(likelihood - lkl_old) / likelihood < .1:
            nstable += 1
        if likelihood > lkl_old:
            lkl_old = likelihood
        if nstable == int(stable_per * n_epochs):
            # Converged. Breaking
            break

    # Extract probabilities associated to the 2D Gaussian
    # ↓ 每顆星屬於高斯（星團）成分的機率
    gumm_p = np.array(list(cluster['gamma_g'].flatten()))

    # cl_cent = cluster['mu']

    return gumm_p


def initialize_cluster():
    """
    Initialize the 2D Gaussian parameters, and the weights for both
    distributions.
    """
    mu = np.random.uniform(.1, .9, (2,))
    cov = np.eye(2) * np.random.uniform(.1, .9, (2, 2))

    cluster = {'pi_u': .5, 'pi_g': .5, 'mu': mu, 'cov': cov}

    return cluster


# ═══════════════ 核心 2：E 步 ═══════════════
def expectation_step(X, cluster):
    """
    """
    try:
        # Evaluate Gaussian distribution
        gamma_g = cluster['pi_g'] * multivariate_normal(
            mean=cluster['mu'], cov=cluster['cov']).pdf(X)

        # Evaluate uniform distribution (just a constant)
        # ↓ [0,1]² 上的均勻分布密度處處是 1，所以只剩權重 π_u
        gamma_u = cluster['pi_u'] * np.ones(X.shape[0])

        # Normalizing constant
        gammas_sum = gamma_g + gamma_u

        # Probabilities for each element
        cluster['gamma_g'] = gamma_g / gammas_sum
        cluster['gamma_u'] = gamma_u / gammas_sum

        # Save for breaking out
        # ↓ 對數概似 = Σ log(每顆星的總密度)，用來判斷收斂
        cluster['likelihood'] = np.sum(np.log(gammas_sum))

    except np.linalg.LinAlgError:
        pass


# ═══════════════ 核心 3：M 步 ═══════════════
def maximization_step(X, cluster):
    """
    """
    gamma_g = cluster['gamma_g']
    # ↓ N_k：「有效的」高斯成員數 = 所有星屬於高斯的機率總和
    N_k = gamma_g.sum(0)

    # Mean
    # ↓ 以 γ_g 為權重的平均位置
    mu = (gamma_g[:, np.newaxis] * X).sum(0) / N_k
    # Covariance
    cov = np.zeros((X.shape[1], X.shape[1]))
    # ↓ 以 γ_g 為權重的共變異：Σ γ_j (x_j − μ)(x_j − μ)ᵀ / N_k
    for j in range(X.shape[0]):
        diff = (X[j] - mu).reshape(-1, 1)
        cov += gamma_g[j] * np.dot(diff, diff.T)
    cov /= N_k

    # Weight for the Gaussian distribution
    N = float(X.shape[0])
    pi_g = N_k / N

    # Weight for the uniform distribution
    pi_u = np.sum(cluster['gamma_u'], axis=0) / N

    # Update parameters
    cluster.update({'pi_u': pi_u, 'pi_g': pi_g, 'mu': mu, 'cov': cov})
