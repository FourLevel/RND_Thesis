# 基本數據處理與分析套件
import pandas as pd
import numpy as np
# 繪圖套件
import matplotlib.pyplot as plt
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
RESULT_FOLDER = 'results'  # 結果儲存資料夾

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
RESULT_FOLDER = 'results'  # 結果儲存資料夾

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
    # 方法一 直接微分
    smooth_IV["cdf"] = np.gradient(smooth_IV['C'], smooth_IV['K'])+1
    smooth_IV["pdf"] = np.gradient(np.gradient(smooth_IV['C'], smooth_IV['K']), smooth_IV['K'])

    # 方法二
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]
    smooth_IV["RND"] =  (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / ((dk)**2)
    smooth_IV = smooth_IV.dropna()

    # RND 平滑
    smooth_IV["RND"] = savgol_filter(smooth_IV["RND"], 500, 3)

    smooth_IV['right_cumulative'] = 1 - smooth_IV['cdf']

    # 只保存 mix_cp.index.min() <= K <= mix_cp.index.max() 的數據
    smooth_IV = smooth_IV[(smooth_IV['K'] >= df_options_mix.index.min()) & (smooth_IV['K'] <= df_options_mix.index.max())]

    # 過濾無效數據
    smooth_IV = smooth_IV[(smooth_IV['right_cumulative'].notna()) & 
              (smooth_IV['cdf'].notna()) &
              (smooth_IV['right_cumulative'] < 1) & 
              (smooth_IV['right_cumulative'] > 0) &
              (smooth_IV['cdf'] < 1) &
              (smooth_IV['cdf'] > 0)]

    # 將欄位 K 名稱改為 strike_price；將 mixIV 改為 fit_imp_vol；將 C 改為 fit_call；將 cdf 改為 left_cumulative；將 RND 改為 RND_density
    smooth_IV = smooth_IV.rename(columns={'K': 'strike_price', 'mixIV': 'fit_imp_vol', 'C': 'fit_call', 'cdf': 'left_cumulative', 'RND': 'RND_density'})

    fit = smooth_IV

    return fit

# 定義擬合 GPD 的函數，選 1 個點，比較斜率與 CDF
def fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95):
    # 取得 F
    basicinfo = get_FTS()
    F = basicinfo["F"]
    
    # 設定接合位置
    left_tail_point = alpha_1L
    right_tail_point = alpha_1R

    # 檢查並調整 left_cumulative
    if fit['left_cumulative'].iloc[0] > alpha_1L:
        left_tail_point = fit['left_cumulative'].iloc[0] + 0.001
        print(f"警告：left_cumulative[0] ({fit['left_cumulative'].iloc[0]:.4f}) 大於 alpha_1L ({alpha_1L})。將 left_tail_point 設為 {left_tail_point:.4f}")

    # Right-tail
    loc = fit.iloc[(fit['left_cumulative'] - right_tail_point).abs().argsort()[:1]]
    right_end = loc['strike_price'].values[0]
    missing_tail = loc['right_cumulative'].values[0]
    right_sigma = missing_tail / loc['RND_density'].values[0]
    max_strike_index = fit['strike_price'].idxmax()
    end_density = fit.loc[max_strike_index, 'RND_density']

    i = initial_i
    while True:
        try:
            loc_index = fit.index.get_loc(loc.index[0])
            if loc_index - i < 0:
                print(f"警告：i={i} 太大，導致索引為負。調整 initial_i 或檢查數據。")
                break
            loc_slope = (-loc['RND_density'].values[0] + fit.iloc[loc_index-i]['RND_density']) / (i * delta_x)
            if loc_slope < 0:
                break
            i += 1
        except KeyError:
            print(f"警告：loc.index[0]={loc.index[0]} 不在 fit 的索引中。檢查數據一致性。")
            break
        except IndexError:
            print(f"警告：索引 {loc_index-i} 超出範圍。調整 initial_i 或檢查數據。")
            break

    def right_func(x):
        xi, scale = x
        alpha_1R = loc['right_cumulative'].values[0]
        X_alpha_1R = loc['strike_price'].values[0]
        density_error = ((missing_tail * gpd.pdf(loc['strike_price'].values[0] + delta_x, xi, loc=loc['strike_price'].values[0], scale=scale) - loc['RND_density'].values[0]) / delta_x - loc_slope)
        cdf_error = (gpd.cdf(X_alpha_1R, xi, loc=loc['strike_price'].values[0], scale=scale) - alpha_1R)
        return (1e12 * density_error**2) + (1e12 * cdf_error**2)

    right_fit = minimize(right_func, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, F*2, delta_x)}), how='outer')
    fit['right_extra_density'] = missing_tail * gpd.pdf(fit['strike_price'], right_xi, loc=loc['strike_price'].values[0], scale=right_sigma)
    fit['full_density'] = np.where(fit['strike_price'] > loc['strike_price'].values[0], fit['right_extra_density'], fit['RND_density'])

    # Left-tail
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc = fit.iloc[(fit['left_cumulative'] - left_tail_point).abs().argsort()[:1]]
    left_end = loc['strike_price'].values[0]
    missing_tail = loc['left_cumulative'].values[0]
    left_sigma = missing_tail / loc['RND_density'].values[0]

    i = initial_i
    while True:
        loc_slope = (-loc['RND_density'].values[0] + fit.iloc[fit.index.get_loc(loc.index[0])-i]['RND_density']) / (i * delta_x)
        if loc_slope < 0:
            break
        else:
            i += 1

    def left_func(x):
        xi, scale = x
        alpha_1L = loc['left_cumulative'].values[0]
        X_alpha_1L = loc['reverse_strike'].values[0]
        density_error = ((missing_tail * gpd.pdf(loc['reverse_strike'].values[0] + delta_x, xi, loc=loc['reverse_strike'].values[0], scale=scale) - loc['RND_density'].values[0]) / delta_x - loc_slope)
        cdf_error = (gpd.cdf(X_alpha_1L, xi, loc=loc['reverse_strike'].values[0], scale=scale) - alpha_1L)
        return (1e12 * density_error**2) + (1e12 * cdf_error**2)

    left_fit = minimize(left_func, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, loc=loc['reverse_strike'].values[0], scale=left_sigma)
    fit['full_density'] = np.where(fit['strike_price'] < loc['strike_price'].values[0], fit['left_extra_density'], fit['full_density'])

    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    # 找到 CDF 的 Lower Bound and Upper Bound
    lower_bound = left_end
    upper_bound = right_end

    return fit, lower_bound, upper_bound

# 定義擬合 GPD 的函數，選 2 個點，比較 PDF
def fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98):
    # 取得 F
    basicinfo = get_FTS()
    F = basicinfo["F"]
    
    # Right-tail
    loc = fit.iloc[(fit['left_cumulative'] - alpha_1R).abs().argsort()[:1]]
    missing_tail = loc['right_cumulative'].values[0]
    right_sigma = missing_tail / loc['RND_density'].values[0]
    X_alpha_2R = fit.iloc[(fit['right_cumulative'] - alpha_2R).abs().argsort()[:1]]['strike_price'].values[0]

    def right_func(x):
        xi, scale = x
        X_alpha_1R = loc['strike_price'].values[0]
        density_error_1R = (missing_tail * gpd.pdf(X_alpha_1R + delta_x, xi, loc=X_alpha_1R, scale=scale) - loc['RND_density'].values[0])
        density_error_2R = (missing_tail * gpd.pdf(X_alpha_2R + delta_x, xi, loc=X_alpha_2R, scale=scale) - loc['RND_density'].values[0])
        return (1e12 * density_error_1R**2) + (1e12 * density_error_2R**2)

    right_fit = minimize(right_func, [0, right_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    right_xi, right_sigma = right_fit.x

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, F*2, delta_x)}), how='outer')
    fit['right_extra_density'] = missing_tail * gpd.pdf(fit['strike_price'], right_xi, loc=loc['strike_price'].values[0], scale=right_sigma)
    fit['full_density'] = np.where(fit['strike_price'] > loc['strike_price'].values[0], fit['right_extra_density'], fit['RND_density'])

    # Left-tail
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    loc = fit.iloc[(fit['left_cumulative'] - alpha_1L).abs().argsort()[:1]]
    missing_tail = loc['left_cumulative'].values[0]
    left_sigma = missing_tail / loc['RND_density'].values[0]
    X_alpha_2L = fit.iloc[(fit['left_cumulative'] - alpha_2L).abs().argsort()[:1]]['reverse_strike'].values[0]

    def left_func(x):
        xi, scale = x
        X_alpha_1L = loc['reverse_strike'].values[0]
        density_error_1L = (missing_tail * gpd.pdf(X_alpha_1L + delta_x, xi, loc=X_alpha_1L, scale=scale) - loc['RND_density'].values[0])
        density_error_2L = (missing_tail * gpd.pdf(X_alpha_2L + delta_x, xi, loc=X_alpha_2L, scale=scale) - loc['RND_density'].values[0])
        return (1e12 * density_error_1L**2) + (1e12 * density_error_2L**2)

    left_fit = minimize(left_func, [0, left_sigma], bounds=[(-1, 1), (0, np.inf)], method='SLSQP')
    left_xi, left_sigma = left_fit.x

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, loc=loc['reverse_strike'].values[0], scale=left_sigma)
    fit['full_density'] = np.where(fit['strike_price'] < loc['strike_price'].values[0], fit['left_extra_density'], fit['full_density'])

    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    lower_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1L).abs().idxmin(), 'strike_price']
    upper_bound = fit.loc[(fit['full_density_cumulative'] - alpha_1R).abs().idxmin(), 'strike_price']

    return fit, lower_bound, upper_bound

# 比較兩種方法的 RND 擬合結果
def plot_compare_methods_side_by_side(fit1, fit2, lower_bound1, upper_bound1, lower_bound2, upper_bound2, observation_date, expiration_date):
    # 取得 F
    basicinfo = get_FTS()
    F = basicinfo["F"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), dpi=100)
    
    # 左側：方法一（1個點）
    ax1.plot(fit1['strike_price'], fit1['full_density'], 
            label='Empirical RND', 
            color='royalblue')
    
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
    
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Probability')
    ax1.set_xlim(F/2, F*1.5)
    ax1.set_title('Method 1: One Point')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右側：方法二（2個點）
    ax2.plot(fit2['strike_price'], fit2['full_density'],
            label='Empirical RND',
            color='royalblue')
    
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
    
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Probability')
    ax2.set_xlim(F/2, F*1.5)
    ax2.set_title('Method 2: Two Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 設定整體標題
    fig.suptitle(f'Comparison of RND Fitting Methods\nBTC options on {observation_date} (expired on {expiration_date})', 
                 y=1.03)
    
    plt.tight_layout()
    return fig  # 返回圖形物件