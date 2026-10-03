"""GUMM 的機率切點與最後一次清理。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
GUMM.py 給每顆星一個「屬於星團高斯」的機率 gumm_p。這裡決定：
  - GUMMProbCut()：機率低於多少算場星。M45 用 'auto'：把 gumm_p 由小到大
    排好，畫成「百分位 vs 機率」的曲線，找曲線開始從接近 0 往上爬的轉折點
    （手肘），再加 0.05 讓切點稍微嚴一點。
  - lowCIGUMMClean()：外圈結束前，對被判為成員的星再做一次 GUMM，
    機率低於切點的改判為場星。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
  numpy（np）   np.percentile（百分位數）、np.arctan2／cos／sin（旋轉角度）、
                np.where
  GUMM.GUMMProbs   見 GUMM.py
  rotate() 改寫自 kneebow 套件（Georg Unterholzner）的找手肘方法

======================================================================
【(b) 用到的參數與意義】
======================================================================
  GUMM_perc   'auto'（M45）或 0–100 的數字（直接取該百分位當切點）
  gumm_p      每顆星屬於高斯的機率
  N_membs     清完後成員少於這個數就不採用這次清理（M45：25）
  ID, cl_probs     全部星的編號與這一輪的 0/1 判定
  clust_ID, clust_xy  內圈留到最後的星的編號與天空座標

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 52–109 行｜GUMMProbCut()：決定機率切點
  核心 2｜第 112–145 行｜rotate()：找手肘
  核心 3｜第 148–184 行｜lowCIGUMMClean()：最後一次 GUMM 清理

======================================================================
【(d) 整體流程】
======================================================================
  auto 切點：百分位 1%–98% 對應的 gumm_p 組成一條曲線
    → 把整條曲線旋轉到「頭尾連線變水平」→ 最低點就是手肘
    → 切點 = 手肘的機率 + 0.05
  最後清理：對留下的星重算 GUMM → 原本判 1、但 gumm_p ≤ 切點的改成 0
    → 改完後成員仍多於 25 顆才採用
⚠ lowCIGUMMClean() 印出「reject N stars」時的 N 其實是清完後**保留**的星數。
"""

import numpy as np
from .GUMM import GUMMProbs


# ═══════════════ 核心 1：決定機率切點 ═══════════════
def GUMMProbCut(GUMM_perc, gumm_p):
    """
    """
    # Select the probability cut.
    if GUMM_perc == 'auto':
        # Create the percentiles (/100.) vs probabilities array.
        # ↓ 百分位 0.01–0.98，及每個百分位對應的 gumm_p 值；
        #   兩者組成一條由左下往右上的曲線，形狀 = (98, 2)
        percentiles = np.arange(.01, .99, .01)
        perc_probs = np.array([
            percentiles, np.percentile(gumm_p, percentiles * 100.)]).T

        # Find 'elbow' where the probabilities start climbing from ~0.
        prob_cut = rotate(perc_probs)

        # import matplotlib.pyplot as plt
        # from scipy.spatial import distance
        # plt.subplot(221)
        # plt.title("P<{:.4f}".format(prob_cut))
        # plt.scatter(*perc_probs.T)
        # plt.subplot(222)
        # msk = gumm_p < prob_cut
        # plt.scatter(*clust_xy[~msk].T, c=gumm_p[~msk])

        # cl_index = []
        # probs_lst, probs_lst2 = np.arange(.01, .99, .01), []
        # for p in probs_lst:
        #     # Mask for stars that are *rejected*
        #     msk = gumm_p < p
        #     if (~msk).sum() == 0:
        #         break
        #     dmean = distance.cdist(np.array([cl_cent]), clust_xy[~msk]).mean()
        #     cl_index.append([msk.sum(), dmean])
        #     probs_lst2.append(p)

        # if (np.array(cl_index).T[0] == 0).all():
        #     prob_cut = 1.
        # else:
        #     cl_index /= np.array(cl_index).T.max(1)
        #     idx = np.argmin(abs(cl_index.T[0] - cl_index.T[1]))
        #     prob_cut = probs_lst2[idx]

        # plt.subplot(223)
        # plt.title("P<{:.4f}".format(probs_lst2[idx]))
        # plt.plot(probs_lst2, cl_index.T[0], label="N")
        # plt.plot(probs_lst2, cl_index.T[1], label="d")
        # plt.legend()

        # plt.subplot(224)
        # msk = gumm_p < probs_lst[idx]
        # plt.scatter(*clust_xy[~msk].T, c=gumm_p[~msk])

    # ↓ 給數字時直接取那個百分位當切點（M45 不走這裡）
    else:
        prob_cut = np.percentile(gumm_p, GUMM_perc)

    return prob_cut


# ═══════════════ 核心 2：找手肘 ═══════════════
def rotate(data):
    """
    Rotate a 2d vector.

    (Very) Stripped down version of the great 'kneebow' package, by Georg
    Unterholzner. Source:

    https://github.com/georg-un/kneebow

    data   : 2d numpy array. The data that should be rotated.
    return : probability corresponding to the elbow.
    """
    # The angle of rotation in radians.
    # ↓ 曲線頭尾兩點連線的角度
    theta = np.arctan2(
        data[-1, 1] - data[0, 1], data[-1, 0] - data[0, 0])

    # Rotation matrix
    co, si = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(((co, -si), (si, co)))

    # Rotate data vector
    # ↓ 把曲線旋轉 −θ，頭尾連線變成水平；這時離連線最遠（最低）的點就是手肘
    rot_data = data.dot(rot_matrix)

    # Find elbow index
    elbow_idx = np.where(rot_data == rot_data.min())[0][0]

    # Adding a small percentage to the selected probability improves the
    # results by making the cut slightly stricter.
    prob_cut = data[elbow_idx][1] + 0.05

    return prob_cut


# ═══════════════ 核心 3：最後一次 GUMM 清理 ═══════════════
def lowCIGUMMClean(N_membs, GUMM_perc, ID, cl_probs, clust_ID, clust_xy, prfl):
    """
    Remove stars marked as members if their GUMM probability is below
    the 'pob_cut' threshold.
    """
    gumm_p = GUMMProbs(clust_xy)
    prob_cut = GUMMProbCut(GUMM_perc, gumm_p)

    # Don't overwrite
    probs_cl = list(cl_probs)

    for i, st_id in enumerate(ID):
        p = probs_cl[i]
        # If this was marked as a cluster star
        if p == 1.:
            # Find its GUMM probability
            # ↓ 在留下來的星裡找到這顆星的位置，取它的 gumm_p
            j = np.where(clust_ID == st_id)[0][0]
            g_p = gumm_p[j]
            # If its GUMM probability is below the threshold
            if g_p <= prob_cut:
                # Mark star as non-member
                probs_cl[i] = 0.
            # else:
            #     # Replace the 1. with the GUMM probability
            #     probs_cl[i] = g_p

    # Replace the final probabilities list with the cleaner one.
    # ↓ 清完後成員多於 25 顆才採用，否則沿用清理前的結果
    msk = np.array(probs_cl) > 0.
    if msk.sum() > N_membs:
        cl_probs = probs_cl
        print(" \nGUMM analysis: reject {} stars as non-members".format(
            msk.sum()), file=prfl)

    return cl_probs
