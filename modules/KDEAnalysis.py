"""把一輪的 0/1 判定變成連續的成員機率（核密度估計）。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
一輪外圈結束時，每顆星只有「成員 1／場星 0」。這裡用兩群星各自建一個
核密度分布（KDE，把每顆星換成一個小高斯再疊起來，得到平滑的密度）：
  L_memb(x)  ＝ 成員星在 5 維空間（天空 x, y＋pmRA, pmDE, Plx）的密度
  L_field(x) ＝ 場星的密度
每顆星的成員機率 = L_memb / (L_memb + L_field) = 1 / (1 + L_field / L_memb)
（「成員／場星」兩個互斥且窮盡的假設、先驗相等時的貝氏公式。）
注意這裡用的是**原始**（未重抽、未降維）的 pmRA/pmDE/Plx。
已知限制：兩個密度沒有乘上各自的星數權重，等於假設先驗成員比例 50%。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
  warnings      關掉除以零的警告（L_memb = 0 時）
  numpy（np）   np.concatenate（把座標與資料接起來）、np.random.shuffle
  scipy.stats.gaussian_kde
                多維核密度估計；頻寬用 Scott 規則自動決定；
                .evaluate(點) 算這些點的密度

======================================================================
【(b) 用到的參數與意義】
======================================================================
  xy         全部星的天空座標（[0,1]）
  data       全部星的 pmRA、pmDE、Plx（原始值）
  cl_probs   這一輪的 0/1 判定
  Nst_max    場星最多取 5000 顆建 KDE（隨機挑，加速）

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 51–91 行｜probs()：建兩個 KDE 並算每顆星的機率

======================================================================
【(d) 整體流程】
======================================================================
  [x, y, pmRA, pmDE, Plx] 接成 5 維 → 依 0/1 分成場星與成員
    → 場星超過 5000 顆就隨機挑 5000 顆
    → 各建一個 KDE → 對**所有**星算兩個密度 → 機率 = 1/(1 + L_field/L_memb)
    → 失敗（例如成員太少、矩陣奇異）→ 印警告，沿用 0/1
"""

import warnings
import numpy as np
from scipy.stats import gaussian_kde


# ═══════════════ 核心 1：KDE 機率 ═══════════════
def probs(xy, data, cl_probs, Nst_max=5000):
    """
    Assign probabilities to all stars after generating the KDEs for field and
    member stars. The Cluster probability is obtained applying the formula for
    two mutually exclusive and exhaustive hypotheses.
    """

    # Combine coordinates with the rest of the features.
    # ↓ 形狀 (星數, 5)：x, y, pmRA, pmDE, Plx
    all_data = np.concatenate([xy.T, data.T]).T
    # Split into the two populations.
    field_stars = all_data[cl_probs == 0.]
    membs_stars = all_data[cl_probs == 1.]

    # To improve the performance, cap the number of stars using a random
    # selection of 'Nf_max' elements.
    if field_stars.shape[0] > Nst_max:
        idxs = np.arange(field_stars.shape[0])
        np.random.shuffle(idxs)
        field_stars = field_stars[idxs[:Nst_max]]

    # Evaluate all stars in both KDEs
    try:
        # ↓ gaussian_kde 要的形狀是 (維度, 點數)，所以轉置
        kd_field = gaussian_kde(field_stars.T)
        kd_memb = gaussian_kde(membs_stars.T)

        # ↓ 每顆星在「成員分布」與「場星分布」下的密度
        L_memb = kd_memb.evaluate(all_data.T)
        L_field = kd_field.evaluate(all_data.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Probabilities for mutually exclusive and exhaustive hypotheses
            cl_probs = 1. / (1. + (L_field / L_memb))

    except (np.linalg.LinAlgError, ValueError):
        print("WARNING: Could not perform KDE probabilities estimation")

    return cl_probs
