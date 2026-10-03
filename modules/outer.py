"""外圈（outer loop）：pyUPMASK 每一輪做的事。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
pyUPMASK.py 會呼叫 loop() 共 OL_runs（M45：25）次，每次得到每顆星一個
成員機率，最後取平均。一輪的步驟：
  重抽資料 → 標準化 → PCA 降維 → 反覆「分群＋天空檢定」（內圈）直到收斂
  → GUMM 清理 → KDE 算連續機率。
因為每一輪重抽時加的隨機擾動不同，分群結果也會不同；平均 25 輪就把
「某顆星剛好被分到哪一群」的偶然性抹平，得到穩定的機率。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
Python 標準庫：
  os            讀環境變數 PYUPMASK_INDEP_RESAMPLE（M45 加的開關）
第三方套件：
  numpy（np）   np.random.normal（常態亂數，重抽用）、陣列遮罩
  sklearn.preprocessing.StandardScaler
                每一欄減去平均、除以標準差，讓三個維度同一個尺度
                （自行單位 mas/yr、視差單位 mas，數值大小差很多）
  sklearn.decomposition.PCA
                主成分分析：找出資料變化最大的方向，把 3 維投影到前 2 維
本程式的模組：
  inner.loop               內圈：分群＋逐群天空檢定
  GUMM.GUMMProbs           「2D 高斯＋均勻背景」模型，給每顆星屬於高斯成分的機率
  GUMMExtras.GUMMProbCut   決定 GUMM 機率要切在哪裡
  GUMMExtras.lowCIGUMMClean  外圈結束前最後一次 GUMM 清理
  KDEAnalysis.probs        用核密度估計算最後的連續機率

======================================================================
【(b) 用到的參數與意義】
======================================================================
  ID, xy        星的編號、縮放到 [0,1] 的天空座標
  data, data_err  分群用的 pmRA/pmDE/Plx 與誤差（已剔除 5σ 極端值）
  resampleFlag  是否重抽（M45：True）
  PCAflag, PCAdims  是否 PCA、降到幾維（M45：True、2）
  GUMM_flag, GUMM_perc  是否 GUMM 清理、機率切點（M45：True、auto）
  KDEP_flag     是否用 KDE 算連續機率（M45：True）
  IL_runs       內圈最多幾次（M45：25）
  N_membs       每個小群平均星數（M45：25）
  N_cl_max      小群數上限（M45：1000）
  clust_method  分群方法（M45：MiniBatchKMeans）
  Kest          Ripley's K 估計器（pyUPMASK.py 建好傳進來）
  prfl          輸出去向（None = 螢幕）
環境變數：
  PYUPMASK_INDEP_RESAMPLE=1   三個維度各自獨立重抽（M45 加的；run_variant.py
                              --indep-resample 會設它）。不設 = 上游原本的行為

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 107–165 行｜loop() 的內圈迴圈：反覆分群＋檢定，再用 GUMM 清理
  核心 2｜第 167–193 行｜loop() 的收尾：成員記 1、最後 GUMM 清理、KDE 機率
  核心 3｜第 196–228 行｜reSampleData()：用誤差重抽資料並標準化
  核心 4｜第 231–247 行｜dimReduc()：PCA 降維

======================================================================
【(d) 整體流程】
======================================================================
  複製 ID 與座標（避免改到原資料）
    → reSampleData：每顆星的 pmRA/pmDE/Plx 加上「常態亂數 × 誤差」，再標準化
    → dimReduc：PCA 3 維 → 2 維
    → 內圈最多 25 次：
        inner.loop 分群並檢定 → 全部小群都通過 = 收斂，停止
        剩下的星少於 25 顆 → 停止
        否則只留通過的星；再用 GUMM 在天空上擬合「高斯＋均勻」，
        丟掉屬於均勻背景的星（剩下的星夠多才丟）
    → 留到最後的星記 1，其餘記 0
    → 最後一次 GUMM 清理
    → KDE：用原始（未重抽）的位置＋三個維度算連續機率
"""

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import inner
from .GUMM import GUMMProbs
from .GUMMExtras import GUMMProbCut, lowCIGUMMClean
from . import KDEAnalysis


def loop(
    ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
    GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
    clRjctMethod, Kest, C_thresh, cl_method_pars, prfl, KDE_vals,
        standard_scale=True):
    """
    Perform the outer loop: inner loop until all "fake" clusters are rejected
    """

    # Make a copy of the original data to avoid over-writing it
    clust_ID, clust_xy = np.array(list(ID)), np.array(list(xy))

    # Re-sample the data using its uncertainties?
    # ↓ 這一輪的資料版本：原值 + 隨機擾動，再標準化
    clust_data = reSampleData(
        resampleFlag, data, data_err, prfl, standard_scale)

    # Apply PCA and features reduction
    # ↓ 3 維壓成 2 維（分群只在這 2 維上做）
    clust_data = dimReduc(clust_data, PCAflag, PCAdims, prfl)

    # ═══════════════ 核心 1：內圈迴圈＋GUMM 清理 ═══════════════
    # Call the inner loop until all the "fake clusters" are rejected
    for _iter in range(IL_runs):
        print("\n IL iteration {}".format(_iter + 1), file=prfl)

        # Call the Inner Loop (IL)
        # ↓ N_clusts：切出幾個小群；msk_all：通過天空檢定的星（True）；
        #   N_survived：通過的小群數
        N_clusts, msk_all, N_survived, KDE_vals = inner.loop(
            clust_xy, clust_data, N_membs, N_cl_max, clust_method,
            clRjctMethod, KDE_vals, Kest, C_thresh, cl_method_pars, prfl)

        # No clusters were rejected in this iteration of the IL. This means
        # that the method converged. Break
        # ↓ 每個小群都通過 → 剩下的星都「在天空上聚集」，不用再切，收斂
        if N_clusts == N_survived:
            print(" All clusters survived, N={}".format(
                clust_xy.shape[0]), file=prfl)
            break

        # Applying 'msk_all' results in too few stars. Break
        # ↓ 通過的星少於 25 顆 → 再切下去沒意義，停止（這時保留的是上一次的星）
        if msk_all.sum() < N_membs:
            print(" N_stars<{:.0f} Breaking".format(N_membs), file=prfl)
            break
        print(" A total of {} stars survived in {} clusters".format(
            msk_all.sum(), N_survived), file=prfl)

        # Keep only stars identified as members and move on to the next
        # iteration
        # ↓ 只留通過檢定的星，下一次內圈只在它們之間重新分群
        clust_ID, clust_xy, clust_data = clust_ID[msk_all],\
            clust_xy[msk_all], clust_data[msk_all]

        # Clean using GUMM
        if GUMM_flag:
            print(" Performing GUMM analysis...", file=prfl)
            # ↓ 在天空座標上擬合「一個 2D 高斯（星團）＋均勻分布（場星）」，
            #   gumm_p = 每顆星屬於高斯成分的機率
            gumm_p = GUMMProbs(clust_xy)
            # ↓ 機率切點：auto = 找「機率從接近 0 開始爬升」的轉折點
            prob_cut = GUMMProbCut(GUMM_perc, gumm_p)
            # Mark all stars as members
            probs_cl = np.ones(len(clust_xy))
            # Mark as non-members those below 'prob_cut'
            probs_cl[gumm_p <= prob_cut] = 0.

            # Keep only member stars for the next run (if enough stars remain
            # in the list)
            # ↓ 清掉後剩下的星多於 25 顆才真的清；否則這次 GUMM 結果不採用
            msk = probs_cl > 0.
            if msk.sum() > N_membs:
                clust_ID, clust_xy, clust_data = clust_ID[msk], clust_xy[msk],\
                    clust_data[msk]
                print(" Rejected {} stars as non-members".format(
                    len(probs_cl) - msk.sum()), file=prfl)

    if _iter + 1 == IL_runs:
        print("Maximum number of IL runs reached. Breaking", file=prfl)

    # ═══════════════ 核心 2：這一輪的機率 ═══════════════
    # Mark all the stars that survived in 'clust_ID' as members assigning
    # a probability of '1'. All others are field stars and are assigned
    # a probability of '0'.
    # ↓ 留到最後的星記 1，其餘 0（這是「這一輪」的二元判定）
    cl_probs = np.zeros(len(ID))
    for i, st in enumerate(ID):
        if st in clust_ID:
            cl_probs[i] = 1.

    # Perform a final cleaning on the list of stars selected as members.
    # Use the last list of coordinates and IDs from the inner loop.
    # This is only ever used for *very* low contaminated clusters.
    if GUMM_flag:
        print("Performing final GUMM analysis...", file=prfl)
        cl_probs = lowCIGUMMClean(
            N_membs, GUMM_perc, ID, cl_probs, clust_ID, clust_xy, prfl)

    # Estimate probabilities using KDEs for the field stars, and assigned
    # true members.
    # ↓ 把 0/1 變成連續機率：用「位置＋原始 pmRA/pmDE/Plx」（未重抽、未降維）
    #   建成員與場星兩個核密度分布，再對每顆星算後驗機率
    if KDEP_flag:
        print("Performing KDE analysis...", file=prfl)
        cl_probs = KDEAnalysis.probs(xy, data, cl_probs)

    return list(cl_probs), KDE_vals


# ═══════════════ 核心 3：用誤差重抽 ═══════════════
def reSampleData(resampleFlag, data, data_err, prfl, standard_scale=True):
    """
    Re-sample the data given its uncertainties using a normal distribution
    """
    if resampleFlag:
        # Gaussian random sample
        #
        # 原始 pyUPMASK 每顆星只抽「一個」亂數，再套用到所有資料維度上，等於
        # pmRA/pmDE/Plx 的擾動完全相關。設環境變數 PYUPMASK_INDEP_RESAMPLE=1
        # 改成每個維度各自獨立抽樣。保留原行為當預設，才不會偏離已發表的方法。
        if os.environ.get("PYUPMASK_INDEP_RESAMPLE") == "1":
            # ↓ 每顆星、每個維度各抽一個標準常態亂數，形狀 = (星數, 3)
            grs = np.random.normal(0., 1., data.shape)
            sampled_data = data + grs * data_err
        else:
            # ↓ 每顆星只抽一個亂數，形狀 = (星數,)；grs[:, np.newaxis] 把它變成
            #   (星數, 1)，乘上誤差時自動複製到三個維度 → 三個維度往同一個
            #   「幾個 σ」的方向擾動
            grs = np.random.normal(0., 1., data.shape[0])
            sampled_data = data + grs[:, np.newaxis] * data_err
    else:
        sampled_data = np.array(list(data))

    # ↓ 標準化：每一欄減平均、除標準差
    if standard_scale:
        print(
            "Standard scale: removed mean and scaled to unit variance",
            file=prfl)
        sampled_data = StandardScaler().fit(sampled_data).transform(
            sampled_data)

    return sampled_data


# ═══════════════ 核心 4：PCA 降維 ═══════════════
def dimReduc(cl_data, PCAflag, PCAdims, prfl):
    """
    Perform PCA and feature reduction
    """
    if PCAflag:
        # ↓ 找出變化最大的 PCAdims（2）個方向，把資料投影上去
        pca = PCA(n_components=PCAdims)
        cl_data_pca = pca.fit(cl_data).transform(cl_data)
        print(" Selected N={} PCA features".format(PCAdims), file=prfl)
        # ↓ 每個保留下來的方向各解釋了多少比例的變化
        var_r = ["{:.2f}".format(_) for _ in pca.explained_variance_ratio_]
        print(" Variance ratio: ", ", ".join(var_r), file=prfl)
    else:
        cl_data_pca = cl_data

    return cl_data_pca
