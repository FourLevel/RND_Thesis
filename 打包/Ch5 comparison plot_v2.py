# 基本數據處理與分析套件
import pandas as pd
import numpy as np
# 繪圖套件
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# 日期時間處理
import time
from datetime import datetime
# 數學與統計相關套件
from scipy.optimize import minimize
from scipy.stats import norm, genpareto as gpd
from scipy.signal import savgol_filter
from scipy.interpolate import LSQUnivariateSpline
# 系統與工具套件
import nest_asyncio
import warnings
# 自定義套件
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')


''' 比較兩種方法的 RND 擬合結果（日報酬） '''
# 參數設定
initial_i = 1
delta_x = 0.1
BATCH_SIZE = 10  # 每批處理的數量
SAVE_PROGRESS = True  # 是否保存進度
PROGRESS_FILE = 'progress.txt'  # 進度檔案
RESULT_FOLDER = 'daily results'  # 結果儲存資料夾

# 建立結果資料夾（如果不存在）
import os
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# 讀取進度（如果存在）
last_processed_date = None
if SAVE_PROGRESS and os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        last_processed_date = f.read().strip()
    print(f"從上次進度繼續：{last_processed_date}")

# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_day = pd.DataFrame()

# 選擇日期，將 type 為 day, week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[(instruments['type'].isin(['day', 'week', 'quarter', 'year']))]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates).strftime('%Y-%m-%d')

# 如果有上次進度，從該位置開始處理
if last_processed_date:
    start_idx = list(expiration_dates).index(last_processed_date) + 1
    expiration_dates = expiration_dates[start_idx:]

# 將日期分批
date_batches = [expiration_dates[i:i + BATCH_SIZE] for i in range(0, len(expiration_dates), BATCH_SIZE)]

# 對每一批次進行處理
for batch_num, batch_dates in enumerate(date_batches, 1):
    print(f"\n開始處理第 {batch_num} 批次，共 {len(batch_dates)} 個日期")
    
    # 對批次中的每個到期日進行處理
    for expiration_date in batch_dates:
        # 計算觀察日（到期日前 1 天）
        expiry = pd.to_datetime(expiration_date)
        observation_date = (expiry - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n處理中... 觀察日: {observation_date}, 到期日: {expiration_date}")
        
        try:
            # 讀取資料
            call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4, s=None, w=None)
            fit = RND_function(smooth_IV)

            # 方法一：選1個點
            fit1, lower_bound1, upper_bound1 = fit_gpd_tails_use_slope_and_cdf_with_one_point(
                fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
            )

            # 方法二：選2個點
            fit2, lower_bound2, upper_bound2 = fit_gpd_tails_use_pdf_with_two_points(
                fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
            )

            # 比較兩種方法並儲存圖片
            fig = plot_compare_methods_side_by_side(
                fit1, fit2,
                lower_bound1, upper_bound1,
                lower_bound2, upper_bound2,
                observation_date, expiration_date
            )
            
            # 儲存圖片
            fig.canvas.draw()  # 強制更新畫布
            plt.pause(0.1)  # 給予一點時間完成繪圖
            fig.savefig(f'{RESULT_FOLDER}/RND_comparison_{observation_date}_{expiration_date}.png', 
                       bbox_inches='tight', dpi=200)
            plt.close(fig)  # 明確關閉特定的圖形
            
            # 儲存進度
            if SAVE_PROGRESS:
                with open(PROGRESS_FILE, 'w') as f:
                    f.write(expiration_date)
            
            print(f"成功處理 {expiration_date} 的數據")
            
        except Exception as e:
            print(f"處理 {expiration_date} 時發生錯誤: {str(e)}")
            continue
    
    print(f"\n完成第 {batch_num} 批次處理")
    
    # 每批次完成後暫停一下（可選）
    time.sleep(5)  # 暫停 5 秒

print("\n所有日期處理完成")

# 處理完成後刪除進度檔案（可選）
if SAVE_PROGRESS and os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)


''' 比較兩種方法的 RND 擬合結果（週報酬） '''
# 參數設定
initial_i = 1
delta_x = 0.1
BATCH_SIZE = 10  # 每批處理的數量
SAVE_PROGRESS = True  # 是否保存進度
PROGRESS_FILE = 'progress.txt'  # 進度檔案
RESULT_FOLDER = 'weekly results'  # 結果儲存資料夾

# 建立結果資料夾（如果不存在）
import os
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# 讀取進度（如果存在）
last_processed_date = None
if SAVE_PROGRESS and os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        last_processed_date = f.read().strip()
    print(f"從上次進度繼續：{last_processed_date}")

# 讀取 instruments.csv
instruments = pd.read_csv('deribit_data/instruments.csv')

# 建立 DataFrame 存放迴歸資料
df_regression_day = pd.DataFrame()

# 選擇日期，將 type 為 week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = instruments[(instruments['type'].isin(['week', 'quarter', 'year']))]['date'].unique()

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates).strftime('%Y-%m-%d')

# 如果有上次進度，從該位置開始處理
if last_processed_date:
    start_idx = list(expiration_dates).index(last_processed_date) + 1
    expiration_dates = expiration_dates[start_idx:]

# 將日期分批
date_batches = [expiration_dates[i:i + BATCH_SIZE] for i in range(0, len(expiration_dates), BATCH_SIZE)]

# 對每一批次進行處理
for batch_num, batch_dates in enumerate(date_batches, 1):
    print(f"\n開始處理第 {batch_num} 批次，共 {len(batch_dates)} 個日期")
    
    # 對批次中的每個到期日進行處理
    for expiration_date in batch_dates:
        # 計算觀察日（到期日前 7 天）
        expiry = pd.to_datetime(expiration_date)
        observation_date = (expiry - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"\n處理中... 觀察日: {observation_date}, 到期日: {expiration_date}")
        
        try:
            # 讀取資料
            call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4, s=None, w=None)
            fit = RND_function(smooth_IV)

            # 方法一：選1個點
            fit1, lower_bound1, upper_bound1 = fit_gpd_tails_use_slope_and_cdf_with_one_point(
                fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
            )

            # 方法二：選2個點
            fit2, lower_bound2, upper_bound2 = fit_gpd_tails_use_pdf_with_two_points(
                fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
            )

            # 比較兩種方法並儲存圖片
            fig = plot_compare_methods_side_by_side(
                fit1, fit2,
                lower_bound1, upper_bound1,
                lower_bound2, upper_bound2,
                observation_date, expiration_date
            )
            
            # 儲存圖片
            fig.canvas.draw()  # 強制更新畫布
            plt.pause(0.1)  # 給予一點時間完成繪圖
            fig.savefig(f'{RESULT_FOLDER}/RND_comparison_{observation_date}_{expiration_date}.png', 
                       bbox_inches='tight', dpi=200)
            plt.close(fig)  # 明確關閉特定的圖形
            
            # 儲存進度
            if SAVE_PROGRESS:
                with open(PROGRESS_FILE, 'w') as f:
                    f.write(expiration_date)
            
            print(f"成功處理 {expiration_date} 的數據")
            
        except Exception as e:
            print(f"處理 {expiration_date} 時發生錯誤: {str(e)}")
            continue
    
    print(f"\n完成第 {batch_num} 批次處理")
    
    # 每批次完成後暫停一下（可選）
    time.sleep(5)  # 暫停 5 秒

print("\n所有日期處理完成")

# 處理完成後刪除進度檔案（可選）
if SAVE_PROGRESS and os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)


''' 比較兩種方法的 RND 擬合結果（日報酬；已選定日期 2022-04-11, 2023-07-22） '''
# 參數設定
initial_i = 1
delta_x = 0.1
RESULT_FOLDER = 'chosen results'  # 結果儲存資料夾

# 建立結果資料夾（如果不存在）
import os
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# 建立 DataFrame 存放迴歸資料
df_regression_day = pd.DataFrame()

# 選擇日期，將 type 為 day, week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = ['2022-04-11', '2023-07-22']

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates).strftime('%Y-%m-%d')
    
# 對批次中的每個到期日進行處理
for expiration_date in expiration_dates:
    # 計算觀察日（到期日前 1 天）
    expiry = pd.to_datetime(expiration_date)
    observation_date = (expiry - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n處理中... 觀察日: {observation_date}, 到期日: {expiration_date}")
    
    try:
        # 讀取資料
        call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
        F = find_F2()
        get_FTS()
        df_options_mix = mix_cp_function_v2()
        smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4, s=None, w=None)
        fit = RND_function(smooth_IV)
        # 方法一：選1個點
        fit1, lower_bound1, upper_bound1 = fit_gpd_tails_use_slope_and_cdf_with_one_point(
            fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
        )
        # 方法二：選2個點
        fit2, lower_bound2, upper_bound2 = fit_gpd_tails_use_pdf_with_two_points(
            fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
        )
        # 比較兩種方法並儲存圖片
        fig = plot_compare_methods_side_by_side(
            fit1, fit2,
            lower_bound1, upper_bound1,
            lower_bound2, upper_bound2,
            observation_date, expiration_date
        )
        
        # 儲存圖片
        fig.canvas.draw()  # 強制更新畫布
        plt.pause(0.1)  # 給予一點時間完成繪圖
        fig.savefig(f'{RESULT_FOLDER}/Chosen_RND_comparison_{observation_date}_{expiration_date}.png', 
                   bbox_inches='tight', dpi=200)
        plt.close(fig)  # 明確關閉特定的圖形
        
        print(f"成功處理 {expiration_date} 的數據")
        
    except Exception as e:
        print(f"處理 {expiration_date} 時發生錯誤: {str(e)}")
        continue



''' Function '''
# 讀取資料
def read_data_v2(expiration_date):
    call_iv = pd.read_csv(f"deribit_data/iv/call/call_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    put_iv = pd.read_csv(f"deribit_data/iv/put/put_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    df_idx = pd.read_csv(f"deribit_data/BTC-index/BTC_index_{expiration_date}.csv", index_col="Unnamed: 0")

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"deribit_data/BTC-call/call_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    put_price = pd.read_csv(f"deribit_data/BTC-put/put_strike_{expiration_date}.csv", index_col="Unnamed: 0")

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)

    return call_iv, put_iv, call_price, put_price, df_idx

# 找出 F
def find_F1(K, type):
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    def calculate_call_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return ( norm.cdf(d1) - K / F * norm.cdf(d2) ) * S0

    def calculate_put_price(F, K, sigma, T, S0):
        d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return ( K / F * norm.cdf(-d2) - norm.cdf(-d1) ) * S0

    def newton_method(real_price, K, sigma, T, S, type, tolerance=1e-6, max_iterations=1000):
        F = S*0.8
        for _ in range(max_iterations):
            if type=="C":
                guess_price = calculate_call_price(F, K, sigma, T, S)
            elif type=="P":
                guess_price = calculate_put_price(F, K, sigma, T, S)

            F_new = F + abs(guess_price - real_price) * 0.5
            if abs(real_price - guess_price) < tolerance:
                return F_new
            F = F_new
            
        return F
    
    if type=="C":
        price = call_price[K].loc[observation_date]  
        sigma = call_iv[K].loc[observation_date]
    if type=="P":
        price = put_price[K].loc[observation_date]  
        sigma = put_iv[K].loc[observation_date]
    
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days /365 
    S = df_idx["index_price"].loc[observation_date]

    return newton_method(price, K, sigma, T, S, type)

# 找出 F
def find_F2():
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    S = df_idx["index_price"].loc[observation_date]
    df = call_price.loc[observation_date][call_price.loc[observation_date]!=0]
    result_c = df[(df.index >= S*0.9) & (df.index <= S*1.5)]
    result_c = pd.DataFrame(result_c)
    result_c.columns = ["C"]

    df = put_price.loc[observation_date][put_price.loc[observation_date]!=0]
    result_p = df[(df.index >= S*0.8) & (df.index <= S*1.1)]
    result_p = pd.DataFrame(result_p)
    result_p.columns = ["P"]

    F_values_c = [find_F1(K, "C") for K in result_c.index]
    F_values_p =  [find_F1(K, "P") for K in result_p.index]
    F = np.array(F_values_c+F_values_p).mean()
    
    return F

# 找出 F、T、S
def get_FTS():
    global observation_date, expiration_date
    F = find_F2()
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observation_date)).days/365
    S = df_idx["index_price"].loc[observation_date]
    return {"F": F, "T": T, "S": S}

# 定義混合買權、賣權隱含波動率函數
def mix_cp_function_v2():
    global observation_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    mix = pd.concat([call_iv.loc[observation_date], put_iv.loc[observation_date]], axis=1)
    mix.columns = ["C", "P"]
    mix = mix.replace(0, np.nan)

    # atm
    atm = mix.loc[(mix.index <= F*1.1) & (mix.index >= F*0.9)]
    atm["mixIV"] = atm[["C","P"]].mean(axis=1)

    # otm
    otm = pd.DataFrame(pd.concat([mix.loc[mix.index < F*0.9, 'P'], mix.loc[mix.index > F*1.1, 'C'] ], axis=0), columns=["mixIV"])

    # mix
    mix_cp = pd.concat([atm, otm], axis=0).sort_index()
    mix_cp[["C","P"]] = mix
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[:F*2.5]
    
    return mix_cp

# 定義 LSQUnivariateSpline 函數
def UnivariateSpline_function_v3(mix_cp, power=4, s=None, w=None):
    global observation_date, expiration_date, delta_x
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    
    # 在 F 的位置加入 knot
    knots = np.array([F])
    spline = LSQUnivariateSpline(mix_cp.index, mix_cp["mixIV"], knots, k=power)
    
    min_K = 0
    max_K = int(max(mix_cp.index)*1.2)
    dK = delta_x
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame([K_fine, Vol_fine], index=["K", "mixIV"]).T
    smooth_IV["C"] = call.future(F, smooth_IV["K"], T, smooth_IV["mixIV"], S)
    
    return smooth_IV

# RND
def RND_function(smooth_IV):
    # 計算 RND（Risk-neutral Density）
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]
    smooth_IV["RND"] = (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / ((dk)**2)
    smooth_IV = smooth_IV.dropna()

    # RND 平滑
    smooth_IV["RND"] = savgol_filter(smooth_IV["RND"], 301, 2)

    # 計算 CDF，使用中央差分
    smooth_IV["left_cumulative"] = (
        (smooth_IV['C'].shift(-1) - smooth_IV['C'].shift(1)) / 
        (smooth_IV['K'].shift(-1) - smooth_IV['K'].shift(1))
    ) + 1
    smooth_IV["right_cumulative"] = 1 - smooth_IV["left_cumulative"]

    # 找出 RND 最大值的位置
    max_rnd_idx = smooth_IV["RND"].idxmax()
    
    # 從最大值往右檢查，找出第一個負值的位置
    right_cut_idx = None
    for idx in smooth_IV.loc[max_rnd_idx:].index:
        if smooth_IV.loc[idx, "RND"] <= 0:
            right_cut_idx = idx
            break
    
    # 如果找到負值，則截斷資料
    if right_cut_idx is not None:
        smooth_IV = smooth_IV.loc[:right_cut_idx-1]

    # 過濾無效數據
    smooth_IV = smooth_IV[
        # 基本範圍檢查
        (smooth_IV['K'] >= df_options_mix.index.min()) & 
        (smooth_IV['K'] <= df_options_mix.index.max()) &
        # CDF 相關檢查
        (smooth_IV['right_cumulative'].notna()) & 
        (smooth_IV['left_cumulative'].notna()) &
        (smooth_IV['right_cumulative'] < 0.9999) &
        (smooth_IV['right_cumulative'] > 0.0001) &
        (smooth_IV['left_cumulative'] < 0.9999) &
        (smooth_IV['left_cumulative'] > 0.0001) &
        # RND 密度檢查
        (smooth_IV['RND'] > 0)
    ]

    '''
    # 檢查過濾後的資料
    if len(smooth_IV) == 0:
        print("警告：過濾後沒有剩餘資料")
    else:
        print(f"資料點數：{len(smooth_IV)}")
        print("RND 統計資訊：")
        print(smooth_IV['RND'].describe())
    '''

    # 重命名欄位
    smooth_IV = smooth_IV.rename(columns={
        'K': 'strike_price', 
        'mixIV': 'fit_imp_vol', 
        'C': 'fit_call', 
        'RND': 'RND_density'
    })

    return smooth_IV

# 定義擬合 GPD 的函數，選 1 個點，比較斜率與 CDF
def fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95):
    """使用斜率和CDF擬合GPD尾部
    
    Args:
        fit (DataFrame): 包含RND和CDF資料的DataFrame
        initial_i (int): 初始搜尋步數
        delta_x (float): x軸間距
        alpha_1L (float): 左尾接合點的累積機率
        alpha_1R (float): 右尾接合點的累積機率
    """
    # 檢查左尾累積機率
    if fit['left_cumulative'].iloc[0] > alpha_1L:
        alpha_1L = fit['left_cumulative'].iloc[0] + 0.0005
        print(f"警告：left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L，將接合點設為 {alpha_1L}。")

    #--------------------
    # 右尾擬合
    #--------------------
    # 1. 找到接合點
    loc_right = fit.iloc[(fit['left_cumulative'] - alpha_1R).abs().argsort()[:1]]
    right_end = loc_right['strike_price'].values[0]
    right_missing_tail = loc_right['right_cumulative'].values[0]
    right_sigma = right_missing_tail / loc_right['RND_density'].values[0]

    # 2. 尋找斜率變化點
    i = initial_i
    loc_index = fit.index.get_loc(loc_right.index[0])
    while True:
        try:
            if loc_index - i < 0:
                print(f"警告：i={i} 太大，導致索引為負。")
                break
            current_density = fit.iloc[loc_index-i]['RND_density']
            right_slope = (current_density - loc_right['RND_density'].values[0]) / (i * delta_x)
            if right_slope < 0:
                break
            i += 1
        except (KeyError, IndexError):
            print("警告：索引錯誤。")
            break

    # 3. GPD擬合
    def right_objective(x):
        xi, scale = x
        X_alpha = right_end
        slope_error = ((right_missing_tail * gpd.pdf(X_alpha + delta_x, xi, loc=X_alpha, scale=scale) - 
                       loc_right['RND_density'].values[0]) / delta_x - right_slope)
        cdf_error = (gpd.cdf(X_alpha, xi, loc=X_alpha, scale=scale) - loc_right['right_cumulative'].values[0])
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    right_fit = minimize(right_objective, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    # 4. 擴展並填充右尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, 160010, delta_x)}), how='outer')
    fit['right_extra_density'] = right_missing_tail * gpd.pdf(fit['strike_price'], right_xi, loc=right_end, scale=right_sigma)
    fit['full_density'] = fit['RND_density'].fillna(0)
    fit.loc[fit['strike_price'] >= right_end, 'full_density'] = fit.loc[fit['strike_price'] >= right_end, 'right_extra_density'].fillna(0)

    #--------------------
    # 左尾擬合
    #--------------------
    # 1. 準備資料
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc_left = fit.iloc[(fit['left_cumulative'] - alpha_1L).abs().argsort()[:1]]
    left_end = loc_left['strike_price'].values[0]
    left_missing_tail = loc_left['left_cumulative'].values[0]
    left_sigma = left_missing_tail / loc_left['RND_density'].values[0]

    # 2. 尋找斜率變化點
    i = initial_i
    loc_index = fit.index.get_loc(loc_left.index[0])
    max_index = len(fit) - 1
    while True:
        try:
            if loc_index + i > max_index:
                print(f"警告：i={i} 太大，導致索引超出範圍。")
                break
            current_density = fit.iloc[loc_index+i]['RND_density']
            left_slope = (current_density - loc_left['RND_density'].values[0]) / (i * delta_x)
            if left_slope > 0:
                break
            i += 1
        except (KeyError, IndexError):
            print("警告：索引錯誤。")
            break

    # 3. GPD擬合
    def left_objective(x):
        xi, scale = x
        X_alpha = loc_left['reverse_strike'].values[0]
        slope_error = ((left_missing_tail * gpd.pdf(X_alpha + delta_x, xi, loc=X_alpha, scale=scale) - 
                       loc_left['RND_density'].values[0]) / delta_x - left_slope)
        cdf_error = (gpd.cdf(X_alpha, xi, loc=X_alpha, scale=scale) - loc_left['left_cumulative'].values[0])
        return (1e12 * slope_error**2) + (1e12 * cdf_error**2)

    left_fit = minimize(left_objective, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    # 4. 擴展並填充左尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = left_missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, 
                                                           loc=loc_left['reverse_strike'].values[0], scale=left_sigma)
    fit.loc[fit['strike_price'] <= left_end, 'full_density'] = fit.loc[fit['strike_price'] <= left_end, 'left_extra_density'].fillna(0)

    #--------------------
    # 後處理
    #--------------------
    fit = fit.sort_values('strike_price')
    fit['full_density'] = fit['full_density'].interpolate(method='cubic')
    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    return fit, left_end, right_end


# 定義擬合 GPD 的函數，選 2 個點，比較 PDF
def fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98):
    """使用兩點PDF比較來擬合GPD尾部
    
    Args:
        fit (DataFrame): 包含RND和CDF資料的DataFrame
        delta_x (float): x軸間距
        alpha_2L (float): 左尾第二個接合點的累積機率
        alpha_1L (float): 左尾第一個接合點的累積機率
        alpha_1R (float): 右尾第一個接合點的累積機率
        alpha_2R (float): 右尾第二個接合點的累積機率
    """
    #--------------------
    # 右尾擬合
    #--------------------
    # 1. 找到接合點
    loc_right = fit.iloc[(fit['left_cumulative'] - alpha_1R).abs().argsort()[:1]]
    right_missing_tail = loc_right['right_cumulative'].values[0]
    right_sigma = right_missing_tail / loc_right['RND_density'].values[0]
    
    # 2. 找到第二個接合點
    X_alpha_2R = fit.iloc[(fit['right_cumulative'] - alpha_2R).abs().argsort()[:1]]['strike_price'].values[0]

    # 3. GPD擬合
    def right_objective(x):
        xi, scale = x
        X_alpha_1R = loc_right['strike_price'].values[0]
        density_error_1R = (right_missing_tail * gpd.pdf(X_alpha_1R + delta_x, xi, loc=X_alpha_1R, scale=scale) - 
                          loc_right['RND_density'].values[0])
        density_error_2R = (right_missing_tail * gpd.pdf(X_alpha_2R + delta_x, xi, loc=X_alpha_2R, scale=scale) - 
                          loc_right['RND_density'].values[0])
        return (1e12 * density_error_1R**2) + (1e12 * density_error_2R**2)

    right_fit = minimize(right_objective, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    # 4. 擴展並填充右尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, 160010, delta_x)}), how='outer')
    fit['right_extra_density'] = right_missing_tail * gpd.pdf(fit['strike_price'], right_xi, 
                                                             loc=loc_right['strike_price'].values[0], scale=right_sigma)
    fit['full_density'] = np.where(fit['strike_price'] > loc_right['strike_price'].values[0], 
                                  fit['right_extra_density'], fit['RND_density'])

    #--------------------
    # 左尾擬合
    #--------------------
    # 1. 準備資料
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc_left = fit.iloc[(fit['left_cumulative'] - alpha_1L).abs().argsort()[:1]]
    left_missing_tail = loc_left['left_cumulative'].values[0]
    left_sigma = left_missing_tail / loc_left['RND_density'].values[0]
    
    # 2. 找到第二個接合點
    X_alpha_2L = fit.iloc[(fit['left_cumulative'] - alpha_2L).abs().argsort()[:1]]['reverse_strike'].values[0]

    # 3. GPD擬合
    def left_objective(x):
        xi, scale = x
        X_alpha_1L = loc_left['reverse_strike'].values[0]
        density_error_1L = (left_missing_tail * gpd.pdf(X_alpha_1L + delta_x, xi, loc=X_alpha_1L, scale=scale) - 
                          loc_left['RND_density'].values[0])
        density_error_2L = (left_missing_tail * gpd.pdf(X_alpha_2L + delta_x, xi, loc=X_alpha_2L, scale=scale) - 
                          loc_left['RND_density'].values[0])
        return (1e12 * density_error_1L**2) + (1e12 * density_error_2L**2)

    left_fit = minimize(left_objective, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    # 4. 擴展並填充左尾
    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = left_missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, 
                                                           loc=loc_left['reverse_strike'].values[0], scale=left_sigma)
    fit.loc[fit['strike_price'] < loc_left['strike_price'].values[0], 'full_density'] = \
        fit.loc[fit['strike_price'] < loc_left['strike_price'].values[0], 'left_extra_density']

    #--------------------
    # 後處理
    #--------------------
    fit = fit.sort_values('strike_price')
    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    # 找出指定累積機率對應的界限
    lower_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1L).abs().idxmin(), 'strike_price']
    upper_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1R).abs().idxmin(), 'strike_price']

    return fit, lower_bound, upper_bound

# 比較兩種方法的 RND 擬合結果
def plot_compare_methods_side_by_side(fit1, fit2, lower_bound1, upper_bound1, lower_bound2, upper_bound2, observation_date, expiration_date):
    # 取得 F
    basicinfo = get_FTS()
    F = basicinfo["F"]
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=100)
    
    # 分別設定兩個子圖的 y 軸格式
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 左側：方法一（1個點）
    ax1.plot(fit1['strike_price'], fit1['full_density'], 
            label='Empirical RND', 
            color='royalblue')
    
    # 找出 CDF 為 5% 和 95% 的點（方法一）
    left_point1 = fit1.loc[(fit1['left_cumulative'] - 0.05).abs().idxmin()]
    right_point1 = fit1.loc[(fit1['left_cumulative'] - 0.95).abs().idxmin()]
    
    # 繪製左尾GPD
    left_tail1 = fit1[fit1['strike_price'] <= upper_bound1]
    ax1.plot(left_tail1['strike_price'], left_tail1['left_extra_density'],
            label='Left tail GPD',
            color='orange',
            linestyle=':',
            linewidth=2)
    
    # 繪製右尾GPD
    right_tail1 = fit1[fit1['strike_price'] >= lower_bound1]
    ax1.plot(right_tail1['strike_price'], right_tail1['right_extra_density'],
            label='Right tail GPD',
            color='green',
            linestyle=':',
            linewidth=2)
    
    # 在 5% 和 95% 的點加上黑色空心圓圈（方法一）
    ax1.plot(left_point1['strike_price'], left_point1['full_density'], 'o', 
            color='black', fillstyle='none', markersize=10)
    ax1.plot(right_point1['strike_price'], right_point1['full_density'], 'o', 
            color='black', fillstyle='none', markersize=10)
    
    # 添加文字標註（方法一）
    ax1.annotate(r'$\alpha_{1L}=0.05$', 
                xy=(left_point1['strike_price'], left_point1['full_density']),
                xytext=(-65, 5), textcoords='offset points',
                fontsize=12)
    ax1.annotate(r'$\alpha_{1R}=0.95$', 
                xy=(right_point1['strike_price'], right_point1['full_density']),
                xytext=(3, 5), textcoords='offset points',
                fontsize=12)
    
    ax1.set_xlabel('Strike Price', fontsize=16)
    ax1.set_ylabel('Probability', fontsize=16)
    ax1.set_xlim(F*0.5, F*1.5)
    ax1.set_title('Method 1: One Point', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 右側：方法二（2個點）
    ax2.plot(fit2['strike_price'], fit2['full_density'],
            label='Empirical RND',
            color='royalblue')
    
    # 找出 CDF 為 2%, 5%, 95%, 98% 的點（方法二）
    left_point2_1 = fit2.loc[(fit2['left_cumulative'] - 0.02).abs().idxmin()]
    left_point2_2 = fit2.loc[(fit2['left_cumulative'] - 0.05).abs().idxmin()]
    right_point2_1 = fit2.loc[(fit2['left_cumulative'] - 0.95).abs().idxmin()]
    right_point2_2 = fit2.loc[(fit2['left_cumulative'] - 0.98).abs().idxmin()]
    
    # 繪製左尾GPD
    left_tail2 = fit2[fit2['strike_price'] <= upper_bound2]
    ax2.plot(left_tail2['strike_price'], left_tail2['left_extra_density'],
            label='Left tail GPD',
            color='orange',
            linestyle=':',
            linewidth=2)
    
    # 繪製右尾GPD
    right_tail2 = fit2[fit2['strike_price'] >= lower_bound2]
    ax2.plot(right_tail2['strike_price'], right_tail2['right_extra_density'],
            label='Right tail GPD',
            color='green',
            linestyle=':',
            linewidth=2)
    
    # 在四個點加上黑色空心圓圈（方法二）
    for point in [left_point2_1, left_point2_2, right_point2_1, right_point2_2]:
        ax2.plot(point['strike_price'], point['full_density'], 'o',
                color='black', fillstyle='none', markersize=10)
    
    # 添加文字標註（方法二）
    annotations = [
        (left_point2_1, r'$\alpha_{2L}=0.02$', -65),
        (left_point2_2, r'$\alpha_{1L}=0.05$', -65),
        (right_point2_1, r'$\alpha_{1R}=0.95$', 3),
        (right_point2_2, r'$\alpha_{2R}=0.98$', 3)
    ]
    
    for point, text, x_offset in annotations:
        ax2.annotate(text,
                    xy=(point['strike_price'], point['full_density']),
                    xytext=(x_offset, 5), textcoords='offset points',
                    fontsize=12)
    
    ax2.set_xlabel('Strike Price', fontsize=16)
    ax2.set_ylabel('Probability', fontsize=16)
    ax2.set_xlim(F*0.5, F*1.5)
    ax2.set_title('Method 2: Two Points', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 設定整體標題
    fig.suptitle(f'Comparison of RND Fitting Methods\nBTC options on {observation_date} (expired on {expiration_date})', 
                 y=1.03, fontsize=22)
    
    plt.tight_layout()
    return fig  # 返回圖形物件