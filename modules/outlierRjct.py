"""分群前剔除極端值。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
自行或視差極端的星（例如很近的高自行星）會讓標準化與 PCA 被少數點主導。
M45 用 stdRegion()：每個維度各自算中位數與標準差，任一維度落在
「中位數 ± 5 個標準差」之外就剔除。被剔除的星最後機率記為 −1。
sklearnMethod()（'local' 或 'forest'）是另外兩種方法，M45 不用。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
  numpy（np）   np.nanmedian、np.nanstd（忽略 NaN 的中位數與標準差）、
                np.logical_and.reduce（多個條件全部成立）
  sklearn.neighbors.LocalOutlierFactor、sklearn.ensemble.IsolationForest
                只有 sklearnMethod() 用（M45 不用）

======================================================================
【(b) 用到的參數與意義】
======================================================================
  data   分群用資料（pmRA、pmDE、Plx）
  nstd   幾個標準差（M45：5）

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 42–59 行｜stdRegion()：逐維度 ±5σ 剔除

======================================================================
【(d) 整體流程】
======================================================================
  對 pmRA、pmDE、Plx 各自：算中位數、標準差 → 標出在範圍內的星
    → 三個維度都在範圍內才保留
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


# ═══════════════ 核心 1：逐維度 ±5σ 剔除 ═══════════════
def stdRegion(data, nstd):
    """
    """
    msk_all = []
    # Process each dimension separately
    for arr in data.T:
        # Mask outliers (np.nan).
        med, std = np.nanmedian(arr), np.nanstd(arr)
        dmin, dmax = med - nstd * std, med + nstd * std
        # ↓ 在範圍內 = True；NaN 的比較結果是 False，所以 NaN 也會被剔除。
        #   .data：如果是遮罩陣列，取底層的布林值
        msk = (arr > dmin) & (arr < dmax)
        msk_all.append(msk.data)
    # Combine into a single mask
    msk_data = np.logical_and.reduce(msk_all)

    return msk_data


def sklearnMethod(data, oultr_method, n_neighbors=50):
    """
    """
    # Predict outliers
    if oultr_method == 'local':
        y_pred = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(
            data)
    elif oultr_method == 'forest':
        y_pred = IsolationForest().fit_predict(data)

    # This elements are kept
    msk_data = y_pred > 0

    return msk_data
