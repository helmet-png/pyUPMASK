"""pyUPMASK 主程式：判定一個星場裡哪些星是星團成員。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
pyUPMASK（Pera et al. 2021，A&A 650, A109）是 UPMASK（Krone-Martins &
Moitinho 2014）的 Python 版。核心想法：星團成員在「自行＋視差」空間裡
聚在一起，**而且**在天空上也聚在一起；場星在自行空間偶然湊成一小群時，
它們在天空上的位置仍然是隨機散布的。所以：
  1. 先在自行＋視差空間把星切成很多小群；
  2. 再逐群檢查天空位置是不是「比隨機更集中」；集中的才算成員。
整個過程用天測誤差重抽、重複很多輪，平均起來得到每顆星的成員機率。

在 M45 專案裡的位置：
  prep.py → prepared/m45_raw.dat
    → scripts/drivers/run_variant.py 把它放進 input/、改 params.ini、
      在本資料夾執行 `python pyUPMASK.py`
    → output/m45_raw.dat（多一欄 probs_final）→ results/baseline.dat
    → run_pipeline.py 取 probs_final ≥ 0.7 當成員

本 fork（helmet-png/pyUPMASK，分支 m45-annotated）＝上游 msolpera/pyUPMASK
commit 3602293 ＋ M45 的三處修改（setup/pyupmask_local.patch）＋這些註解：
  - modules/dataIO.py：distutils 在 Python 3.12 移除，自己補 strtobool
  - modules/outer.py：環境變數 PYUPMASK_INDEP_RESAMPLE=1 時三個維度獨立重抽
  - params.ini：M45 用的欄位名與參數（見下方 (b)）

======================================================================
【(a) 引用的外部函式庫】
======================================================================
Python 標準庫：
  os                 os.devnull（verbose=0 時把輸出丟掉）、os.environ（限制執行緒數）
  pathlib.Path       建 output/ 資料夾
  time（簡寫 t）     計時
  multiprocessing    平行跑外圈（M45 設定 parallel=False，沒有用到）
第三方套件：
  numpy（np）        np.random.seed 固定亂數、np.mean 平均各輪機率
  astropy.stats.RipleysKEstimator
                     Ripley's K 函數：量一群點在平面上「比完全隨機更集中」的程度。
                     這裡建一個 [0,1]×[0,1] 的估計器，給內圈逐群檢定用
本程式的模組（modules/）：
  dataIO.readINI     讀 params.ini，回傳一長串參數
  dataIO.readFiles   列出 input/ 裡所有要處理的檔案
  dataIO.dread       讀星表，拆出 ID、天空座標、分群用資料、誤差
  dataIO.dmask       剔除分群資料的極端值（離中位數超過 5σ）
  dataIO.dxynorm     天空座標縮放到 [0, 1]
  dataIO.dwrite      寫出結果檔（多一欄 probs_final）
  outer.loop         一輪外圈：重抽 → PCA → 內圈分群與檢定 → GUMM → KDE 機率

======================================================================
【(b) 用到的參數與意義】（全部來自 params.ini，M45 的值）
======================================================================
[General parameters]
  rnd_seed = 99        亂數種子（run_variant.py --seed 可覆寫）
  verbose = 1          印出全部過程
  parallel = False     不平行跑外圈
  numpy_multi = True   允許 numpy 多執行緒；為 True 時平行旗標會被強制關掉
[Input file's data columns]
  ID = source_id               Gaia 編號
  xy_coords = _x _y            prep.py 做過切平面投影的天空座標
  data = pmRA pmDE Plx         分群用的三個維度：赤經自行、赤緯自行、視差
  uncert = e_pmRA e_pmDE e_Plx 三者的誤差（重抽用）
  oultr_method = stdregion、stdRegion_nstd = 5
                               任一維度離中位數超過 5 個標準差的星先剔除，
                               最後機率記為 −1
[Outer loop]
  OL_runs = 25         外圈輪數；最終機率 = 25 輪平均
  resampleFlag = True  每輪先用誤差重抽資料
  PCAflag = True、PCAdims = 2   三維壓成兩維再分群
  GUMM_flag = True、GUMM_perc = auto
                       每次內圈後，用「2D 高斯＋均勻背景」模型再清一次天空位置
  KDEP_flag = True     最後用核密度估計算連續的成員機率（否則只有 0 或 1）
[Inner loop]
  IL_runs = 25         內圈最多重複幾次
  N_membs = 25         每個小群平均幾顆星 → 小群數 = 星數 ÷ 25
  N_cl_max = 1000      小群數上限
  clust_method = MiniBatchKMeans   分群方法（scikit-learn）
程式內寫死：
  clRjctMethod = 'rkfunc'、C_thresh 由內圈依群大小算（1.68 / N，Dixon 2001 的 1% 臨界值）

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 120–186 行｜main()：逐檔讀資料、剔除極端值、縮放座標、跑所有外圈、平均、寫檔
  核心 2｜第 256–299 行｜dataProcess() 的外圈迴圈：跑 OL_runs 輪 outer.loop()
  核心 3｜第 310–348 行｜程式進入點：讀 params.ini、設定執行緒、呼叫 main()
真正的分群與判定邏輯在 modules/outer.py（外圈）與 modules/inner.py（內圈）。

======================================================================
【(d) 整體流程】
======================================================================
  讀 params.ini
    → 對 input/ 裡每個檔案：
        固定亂數種子
        → 讀星表；三個分群維度都沒有值的星先放一邊（最後機率 −1）
        → 剔除任一維度超過 5σ 的極端星（最後機率 −1）
        → 天空座標縮放到 [0, 1]
        → 外圈 25 輪，每輪（outer.loop）：
            用誤差重抽 pmRA、pmDE、Plx → 標準化 → PCA 壓成 2 維
            → 內圈最多 25 次（inner.loop）：
                MiniBatchKMeans 切成 約 星數÷25 個小群
                → 每個小群算 Ripley's K，天空位置比隨機集中的才保留
                → 全部小群都保留 = 收斂，停止；否則只留保留下來的星再切一次
                → 每次內圈後再用 GUMM 清掉天空上偏離中心的星
            → 留到最後的星這一輪記為 1，其餘 0；最後再做一次 GUMM 清理
            → KDE：用「位置＋自行＋視差」建成員與場星兩個密度，
              機率 = 1 / (1 + 場星密度 / 成員密度)
        → 25 輪機率平均 → 寫到 output/<同檔名>，多一欄 probs_final
"""

import os
from pathlib import Path
import numpy as np
from astropy.stats import RipleysKEstimator
import time as t
from modules import outer
from modules.dataIO import readFiles, readINI, dread, dmask, dxynorm, dwrite
import multiprocessing as mp


# ═══════════════ 核心 1：逐檔處理的主流程 ═══════════════
def main(
    parallel_flag, parallel_procs, rnd_seed, verbose, ID_c, x_c, y_c,
    data_cols, data_errs, oultr_method, stdRegion_nstd, OL_runs, resampleFlag,
    PCAflag, PCAdims, GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs,
        N_cl_max, clust_method, clRjctMethod, C_thresh, cl_method_pars):
    """
    """
    out_folder = "output"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)

    # Process all files inside the '/input' folder
    # ↓ input/ 底下除了 .md 以外的所有檔案（run_variant.py 確保只有一個）
    inputfiles = readFiles()

    for file_path in inputfiles:

        print("\n")
        print("===========================================================")
        print("Processing         : {}".format(file_path.name))
        # Set a random seed for reproducibility
        # ↓ 'None' → 隨機挑一個種子（不可重現）；否則用 params.ini 的值（M45：99）
        if rnd_seed == 'None':
            seed = np.random.randint(100000)
        else:
            seed = int(rnd_seed)
        print("Random seed        : {}".format(seed))
        # ↓ 設定 numpy 全域亂數。之後的重抽、GUMM 初值、MiniBatchKMeans
        #   （沒指定 random_state，會用 numpy 全域亂數）都由這個種子決定
        np.random.seed(seed)

        # Original data
        # ↓ full_data：完整表格（寫檔用）；cl_ID／cl_xy／cl_data／cl_errs：
        #   編號、天空座標、分群用的 pmRA/pmDE/Plx、對應誤差；
        #   data_rjct：三個分群欄位全部缺值的星
        full_data, cl_ID, cl_xy, cl_data, cl_errs, data_rjct = dread(
            file_path, ID_c, x_c, y_c, data_cols, data_errs)

        # Remove outliers
        # ↓ msk_data：True = 保留。任一維度離中位數超過 5σ 的星被剔除
        msk_data, ID, xy, data, data_err = dmask(
            cl_ID, cl_xy, cl_data, cl_errs, oultr_method, stdRegion_nstd)

        # Normalize (x, y) data to [0, 1]
        # ↓ 天空座標各自線性縮放到 [0, 1]，配合 Ripley's K 估計器的 [0,1]² 視窗
        xy01 = dxynorm(xy)

        # ↓ 跑所有外圈；probs_all 形狀 = (OL_runs, 星數)，每列是一輪的機率
        probs_all = dataProcess(
            ID, xy01, data, data_err, verbose, OL_runs,
            parallel_flag, parallel_procs, resampleFlag, PCAflag, PCAdims,
            GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max,
            clust_method, clRjctMethod, C_thresh, cl_method_pars)

        if OL_runs > 1:
            # Obtain the mean of all runs. This are the final probabilities
            # assigned to each star in the frame
            # ↓ 對 25 輪取平均（沿第 0 軸），就是最終的成員機率
            probs_mean = np.mean(probs_all, 0)
        else:
            probs_mean = probs_all[0]

        # Write final data to file
        # ↓ 被剔除的星機率填 −1，寫到 output/<同檔名>
        dwrite(
            out_folder, file_path, full_data, msk_data, data_rjct, probs_mean)


def dataProcess(
    ID, xy, data, data_err, verbose, OL_runs, parallel_flag,
    parallel_procs, resampleFlag, PCAflag, PCAdims, GUMM_flag, GUMM_perc,
    KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method, clRjctMethod,
        C_thresh, cl_method_pars):
    """
    """
    start_t = t.time()

    # TODO this should be handled by the logging() module
    # Set print() according to the 'verbose' parameter
    # ↓ prfl：模組裡 print(..., file=prfl) 的去向；None = 印到螢幕
    if verbose == 0:
        prfl = open(os.devnull, 'w')
    else:
        prfl = None

    # Print input parameters to screen
    if parallel_flag:
        print("Parallel runs      : {}".format(parallel_flag))
        print("Processes          : {}".format(parallel_procs))
    print("Outer loop runs    : {}".format(OL_runs))
    if PCAflag:
        print("Apply PCA          : {}".format(PCAflag))
        print(" PCA N_dims        : {}".format(PCAdims))
    if GUMM_flag:
        print("Apply GUMM         : {}".format(GUMM_flag))
        print(" GUMM percentile   : {}".format(GUMM_perc))
    if KDEP_flag:
        print("Obtain KDE probs   : {}".format(KDEP_flag))

    print("Inner loop runs    : {}".format(IL_runs))
    print("Stars per cluster  : {}".format(N_membs))
    print("Maximum clusters   : {}".format(N_cl_max))
    print("Clustering method  : {}".format(clust_method))
    if cl_method_pars:
        for key, val in cl_method_pars.items():
            print(" {:<17} : {}".format(key, val))
    print("")
    # print("Rejection method   : {}".format(clRjctMethod))
    # if clRjctMethod != 'rkfunc':
    #     print("Threshold          : {:.2f}".format(C_thresh))

    # Define RK test with an area of 1.
    # Kest = None
    # if clRjctMethod == 'rkfunc':
    # ↓ Ripley's K 估計器，視窗 = 縮放後的 [0,1]×[0,1] 天空區域（面積 1）
    Kest = RipleysKEstimator(area=1, x_max=1, y_max=1, x_min=0, y_min=0)
    # if clRjctMethod == 'kdetest' or clust_method == 'rkmeans':
    #     from rpy2.robjects import r
    #     from rpy2.robjects import numpy2ri
    #     from rpy2.robjects.packages import importr
    #     # cat(paste("R version: ",R.version.string,"\n"))
    #     importr('MASS')
    #     r("""
    #     set.seed(12345)
    #     """)
    #     numpy2ri.activate()
    #     r.assign('nruns', 2000)
    #     r.assign('nKde', 50)

    # Arguments for the Outer Loop
    OLargs = (
        ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
        GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
        clRjctMethod, Kest, C_thresh, cl_method_pars, prfl)

    # ═══════════════ 核心 2：外圈迴圈 ═══════════════
    # TODO: Breaks if verbose=0
    # ↓ 平行版（M45 不用：numpy_multi=True 時 parallel 會被強制關掉）
    if parallel_flag is True:
        if parallel_procs == 'None':
            # Use *almost* all the cores
            N_cpu = mp.cpu_count() - 1
        else:
            N_cpu = int(parallel_procs)
        with mp.Pool(processes=N_cpu) as p:
            manager = mp.Manager()
            KDE_vals = manager.dict({})
            probs_all = p.starmap(
                OLfunc, [(OLargs, KDE_vals) for _ in range(OL_runs)])

    # ↓ 循序版（M45 走這裡）：跑 OL_runs 輪，每輪得到每顆星一個機率
    else:
        # ↓ KDE_vals 是舊的 kdetest 檢定用的快取；rkfunc 不用它，只是原樣傳遞
        KDE_vals = {}
        probs_all = []
        for _ in range(OL_runs):
            print("\n--------------------------------------------------------")
            print("OL run {}".format(_ + 1))
            # The KDE_vals dictionary is updated after each OL run
            # ↓ 一輪外圈（見 modules/outer.py）
            probs, KDE_vals = outer.loop(*OLargs, KDE_vals)
            probs_all.append(probs)

            # ↓ 到目前為止的平均機率中，超過 0.5／0.75／0.9／0.95／0.99 的星數，
            #   用來目測各輪累積後是否穩定
            p_dist = [
                (np.mean(probs_all, 0) > _).sum() for _ in
                (.5, .75, .9, .95, .99)]
            print("\nP>(.5, .75, .9, .95, .99): {}, {}, {}, {}, {}".format(
                *p_dist), file=prfl)

    elapsed = t.time() - start_t
    if elapsed > 60.:
        elapsed, ms_id = elapsed / 60., "minutes"
    else:
        ms_id = "seconds"
    print("\nTime consumed: {:.1f} {}".format(elapsed, ms_id))

    return probs_all


def OLfunc(args, KDE_vals):
    """
    Here to handle the parallel runs.
    """
    probs, _ = outer.loop(*args, KDE_vals)
    return probs


# ═══════════════ 核心 3：程式進入點 ═══════════════
if __name__ == '__main__':

    # # Limit numpy's cores used to 1
    # # Source: https://stackoverflow.com/a/58195413/1391441, also
    # # https://stackoverflow.com/q/17053671/1391441

    # parallel_flag, parallel_procs = params[:2]
    # if parallel_flag:
    #     if parallel_procs == 'None':
    #         # Use *almost* all the cores
    #         parallel_procs = mp.cpu_count() - 1
    #     else:
    #         # Never use more than these cores
    #         parallel_procs = min(int(parallel_procs), mp.cpu_count() - 1)
    # else:
    #     parallel_procs = 1

    # Read input parameters.
    # ↓ params.ini 從「目前工作目錄」讀，所以 run_variant.py 要在本資料夾執行
    params = readINI()

    # ↓ params[0] = numpy_multi
    if params[0] is False:
        # Disable numpy's multithreading
        # ↓ 不讓 numpy 多執行緒：把各種數學庫的執行緒數設成 1
        parallel_procs = str(1)
        os.environ["OMP_NUM_THREADS"] = parallel_procs
        os.environ["MKL_NUM_THREADS"] = parallel_procs
        os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
        os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
        os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs
    else:
        # If numpy is allowed to multithread, disable the parallel run
        # ↓ numpy 可多執行緒時，關掉外圈平行（params[1] = parallel_flag）
        params[1] = False

    # ↓ 其餘參數依序傳給 main()
    main(*params[1:])
