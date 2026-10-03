"""讀設定、讀寫星表、前處理。

======================================================================
【這支程式在做什麼】（M45 專案加註，2026-10）
======================================================================
pyUPMASK.py 的輸入輸出全部在這裡：
  readINI()   讀 params.ini（從「目前工作目錄」讀）
  readFiles() 列出 input/ 裡要處理的檔案
  dread()     讀星表，拆出編號、天空座標、分群資料、誤差
  dmask()     剔除極端值（呼叫 outlierRjct.stdRegion）
  dxynorm()   天空座標縮放到 [0, 1]
  dwrite()    寫結果：原表格多一欄 probs_final
M45 修改：distutils 在 Python 3.12 被移除，檔頭自己補了 strtobool()。

======================================================================
【(a) 引用的外部函式庫】
======================================================================
Python 標準庫：
  pathlib.Path    列出 input/、組輸出路徑
  configparser    讀 .ini 設定檔：[區段] 底下「鍵 = 值」
  warnings        發出警告
第三方套件：
  numpy（np）
  astropy.io.ascii       ascii.write 寫純文字表格
  astropy.table.Table    Table.read 讀純文字表格；缺值會變成「遮罩」（masked）
  astropy.table.Column   新增一欄
  astropy.table.vstack   把兩張表上下接起來
  sklearn.preprocessing.MinMaxScaler
                         每一欄線性縮放到 [0, 1]：(x − 最小值) / (最大值 − 最小值)
本程式的模組：
  outlierRjct.stdRegion、sklearnMethod   剔除極端值

======================================================================
【(b) 用到的參數與意義】
======================================================================
params.ini 的各個鍵與 M45 的值見 pyUPMASK.py 檔頭 (b)。
readINI() 回傳的串列順序（pyUPMASK.py 用 params[0]、params[1] 取值，所以順序重要）：
  [numpy_multi, parallel, processes, rnd_seed, verbose, ID, x, y,
   data 欄位, uncert 欄位, oultr_method, stdRegion_nstd, OL_runs, resampleFlag,
   PCAflag, PCAdims, GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs,
   N_cl_max, clust_method, clRjctMethod, C_thresh, cl_method_pars]
寫死的值：clRjctMethod = 'rkfunc'、C_thresh = 1.（之後內圈會依群大小重算）

======================================================================
【(c) 真正在執行操作的核心】（行號以這個版本為準，改程式後要更新）
======================================================================
  核心 1｜第 85–181 行｜readINI()：讀 params.ini
  核心 2｜第 199–238 行｜dread()：讀星表並拆欄位
  核心 3｜第 241–262 行｜dmask()、dxynorm()：剔除極端值、縮放座標
  核心 4｜第 282–301 行｜dwrite()：寫出結果

======================================================================
【(d) 整體流程】
======================================================================
  readINI：逐區段讀值 → 檢查分群方法名稱 → 組成串列回傳
  dread：讀表 → 三個分群欄位全部缺值的星另外放（data_rjct）
    → 拆成 ID、(x, y)、(pmRA, pmDE, Plx)、誤差
  dmask：任一維度超過 ±5σ 的星剔除
  dxynorm：座標縮放到 [0, 1]；x、y 範圍差超過 5% 印警告（視窗不是正方形）
  dwrite：保留的星填平均機率、被剔除的填 −1、全部缺值的星接在表尾也填 −1
    → 寫到 output/<同檔名>
"""

import numpy as np
from pathlib import Path
from astropy.io import ascii
from astropy.table import Column
from astropy.table import Table, vstack
import configparser
import warnings
from sklearn.preprocessing import MinMaxScaler
from .outlierRjct import stdRegion, sklearnMethod


def strtobool(val):
    """distutils 在 Python 3.12 被移除，補一個等效實作（原本 import 自 distutils.util）。"""
    val = val.strip().lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    raise ValueError("invalid truth value {!r}".format(val))


# ═══════════════ 核心 1：讀 params.ini ═══════════════
def readINI():
    """
    Read .ini config file
    """

    # ↓ [Clustering parameters] 的值格式是「型別_值」，例如 int_50 → 50
    def vtype(var):
        tp, v = var.split('_')
        if tp == 'int':
            return int(v)
        elif tp == 'float':
            return float(v)
        elif tp == 'bool':
            return bool(strtobool(v))
        elif tp == 'str':
            return v

    in_params = configparser.ConfigParser()
    # ↓ 相對路徑：從目前工作目錄讀 params.ini
    in_params.read('params.ini')

    # Data columns
    gen_pars = in_params["General parameters"]
    rnd_seed, verbose, parallel_flag, parallel_procs, np_mthread =\
        gen_pars.get('rnd_seed'), gen_pars.getint('verbose'),\
        gen_pars.getboolean('parallel'), gen_pars.get('processes'),\
        gen_pars.getboolean('numpy_multi')
    if parallel_procs not in ('None', 'none', 'NONE'):
        parallel_procs = int(parallel_procs)
        if parallel_procs <= 0:
            raise ValueError("The 'processes' parameter must be >0")

    # Data columns
    data_columns = in_params["Input file's data columns"]
    ID_c = data_columns['ID']
    x_c, y_c = data_columns['xy_coords'].split()
    data_cols = data_columns['data'].split()
    oultr_method = data_columns.get('oultr_method')
    stdRegion_nstd = data_columns.getfloat('stdRegion_nstd')

    # Arguments for the Outer Loop
    outer_loop = in_params['Outer loop']
    OL_runs, resampleFlag,\
        PCAflag, PCAdims, GUMM_flag, KDEP_flag =\
        outer_loop.getint('OL_runs'), outer_loop.getboolean('resampleFlag'),\
        outer_loop.getboolean('PCAflag'), outer_loop.getint('PCAdims'),\
        outer_loop.getboolean('GUMM_flag'), outer_loop.getboolean('KDEP_flag')
    GUMM_perc = outer_loop.get('GUMM_perc')
    if GUMM_perc != 'auto':
        GUMM_perc = float(GUMM_perc)

    # Only read if the code is set to re-sample the data.
    data_errs = []
    if resampleFlag:
        data_errs = data_columns['uncert'].split()

    # Arguments for the Inner Loop
    inner_loop = in_params['Inner loop']
    IL_runs, N_membs, N_cl_max, clust_method = inner_loop.getint('IL_runs'),\
        inner_loop.getint('N_membs'), inner_loop.getint('N_cl_max'),\
        inner_loop.get('clust_method')

    allowed_clust_methods = (
        'KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
        'SpectralClustering', 'AgglomerativeClustering', 'DBSCAN', 'OPTICS',
        'Birch', 'GaussianMixture', 'BayesianGaussianMixture', 'Voronoi',
        'rkmeans', 'kNNdens', 'HDBSCAN')
    if clust_method not in allowed_clust_methods:
        raise ValueError("Unrecognized clustering method '{}'".format(
            clust_method))

    single_run_methods = ('Voronoi', 'rkmeans', 'kNNdens')
    if clust_method in single_run_methods and OL_runs > 1\
            and resampleFlag is False:
        warnings.warn(
            "Single run method selected, only one OL run will be processed")
        OL_runs = 1

    # Only allow the 'rkfunc' method
    # inner_loop.get('clRjctMethod'), inner_loop.getfloat('C_thresh')
    # ↓ 天空檢定方法寫死為 Ripley's K；門檻之後由內圈依群大小重算
    clRjctMethod, C_thresh = 'rkfunc', 1.
    # if clRjctMethod not in ('rkfunc', 'kdetest', 'kdetestpy'):
    #     raise ValueError("'{}' is not a valid choice for clRjctMethod".format(
    #         clRjctMethod))

    cl_method_pars = {}
    for key, val in in_params['Clustering parameters'].items():
        cl_method_pars[key] = vtype(val)

    return [
        np_mthread, parallel_flag, parallel_procs, rnd_seed, verbose,
        ID_c, x_c, y_c, data_cols, data_errs, oultr_method, stdRegion_nstd,
        OL_runs, resampleFlag, PCAflag, PCAdims, GUMM_flag, GUMM_perc,
        KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method, clRjctMethod,
        C_thresh, cl_method_pars]


def readFiles():
    """
    Read files from the input folder
    """
    files = []
    for pp in Path('input').iterdir():
        if not pp.name.endswith(".md"):
            if pp.is_file():
                files += [pp]
            else:
                files += [arch for arch in pp.iterdir()]

    return files


# ═══════════════ 核心 2：讀星表 ═══════════════
def dread(file_path, ID_c, x_c, y_c, data_cols, data_errs):
    """
    """

    # ↓ 讀純文字表格（prep.py 產的 .dat）
    data = Table.read(file_path, format='ascii')
    N_d = len(data)
    print("Stars read         : {}".format(N_d))

    # Remove stars with no valid data
    try:
        # ↓ ~mask = 有值。logical_or：三個分群欄位**至少一個**有值就保留；
        #   三個全部缺值的星放進 data_rjct（最後機率 −1）。
        #   沒有任何缺值時欄位不是遮罩欄，.mask 不存在 → 走 except
        msk = np.logical_or.reduce([~data[_].mask for _ in data_cols])
        data_rjct = data[~msk]
        data = data[msk]
        print("Stars removed      : {}".format(N_d - len(data)))
    except AttributeError:
        # No masked columns
        data_rjct = []
        pass

    # Separate data into groups
    if ID_c == 'None':
        N_d = len(data)
        ID_data = np.arange(1, N_d + 1)
    else:
        ID_data = data[ID_c]
    # ↓ xy_data：(星數, 2) 天空座標；cl_data：(星數, 3) pmRA、pmDE、Plx
    xy_data, cl_data = np.array([data[x_c], data[y_c]]).T,\
        np.array([data[_] for _ in data_cols]).T

    cl_errs = np.array([])
    if data_errs:
        cl_errs = np.array([data[_] for _ in data_errs]).T
    print("Data dimensions    : {}".format(cl_data.shape[1]))

    return data, ID_data, xy_data, cl_data, cl_errs, data_rjct


# ═══════════════ 核心 3：剔除極端值、縮放座標 ═══════════════
def dmask(ID, xy, pdata, perrs, oultr_method, stdRegion_nstd):
    """
    """
    if oultr_method == 'stdregion':
        msk_data = stdRegion(pdata, stdRegion_nstd)
    else:
        msk_data = sklearnMethod(pdata, oultr_method)

    # ↓ 只留沒被剔除的星
    ID_data, xy_data, cl_data = ID[msk_data], xy[msk_data], pdata[msk_data]

    if perrs.any():
        data_err = perrs[msk_data]
    else:
        data_err = np.array([])

    print("Masked outliers    : {}".format((~msk_data).sum()))
    if oultr_method == 'stdregion':
        print(" N_std             : {}".format(stdRegion_nstd))

    return msk_data, ID_data, xy_data, cl_data, data_err


def dxynorm(xy_data):
    """
    """
    # ↓ np.ptp：最大值 − 最小值（x 與 y 各自的範圍）
    _xrange, _yrange = np.ptp(xy_data, 0)
    perc_sq = abs(1. - _xrange / _yrange)
    if perc_sq > .05:
        print((
            "WARNING: (x, y) frame deviates from a square region "
            "by {:.0f}%").format(perc_sq * 100.))
    # ↓ x、y 各自縮放到 [0, 1]（不保持長寬比）
    xy = MinMaxScaler().fit(xy_data).transform(xy_data)
    print("Coordinates scaled : [0, 1]")

    return xy


# ═══════════════ 核心 4：寫出結果 ═══════════════
def dwrite(out_folder, file_path, full_data, msk_data, data_rjct, probs_mean):
    """
    """
    # ↓ input/xxx.dat → output/xxx.dat
    out_path = Path(out_folder, *file_path.parts[1:])

    # Assign probabilities of '-1' to outliers
    # ↓ 先全部填 −1，再把保留的星填上平均機率（四捨五入到小數 4 位）
    pf = np.zeros(len(full_data)) - 1.
    pf[msk_data] = probs_mean
    full_data.add_column(Column(np.round(pf, 4), name='probs_final'))

    # Assign probabilities of '-1' to rejected stars (if any)
    if len(data_rjct) > 0:
        pf = np.zeros(len(data_rjct)) - 1.
        data_rjct.add_column(Column(pf), name='probs_final')
        full_data = vstack([full_data, data_rjct])

    ascii.write(full_data, out_path, overwrite=True)


# def dataNorm(data_arr, err_data=None):
#     """
#     """
#     data_norm, err_norm = [], []
#     for i, arr in enumerate(data_arr.T):
#         min_array, max_array = np.nanmin(arr), np.nanmax(arr)
#         arr_delta = max_array - min_array
#         data_norm.append((arr - min_array) / arr_delta)

#         if err_data is not None:
#             err_norm.append(err_data.T[i] / arr_delta)

#         # # This normalization tends to make things more difficult
#         # mean_arr, std_arr = np.mean(arr[msk_data]), np.std(arr[msk_data])
#         # data_norm.append((arr[msk_data] - mean_arr) / std_arr)

#     return np.array(data_norm).T, np.array(err_norm).T
