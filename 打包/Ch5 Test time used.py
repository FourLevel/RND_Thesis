# 基本數據處理與分析套件
import pandas as pd
import numpy as np
# 數學與統計相關套件
from scipy.optimize import minimize
from scipy.stats import norm, genpareto as gpd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
# 日期時間處理
from datetime import datetime, timedelta
# 系統與工具套件
import warnings
import time
import os
# 自定義套件
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')

initial_i = 1
delta_x = 0.01

# 建立 DataFrame 存放迴歸資料
df_regression_week = pd.DataFrame()

# 選擇日期，將 type 為 week, quarter, year 的 date 選出，作為 expiration_dates
expiration_dates = ['2023/9/22', '2023/9/29', '2023/10/13', '2023/10/20', '2023/10/27', 
                    '2023/11/10', '2023/11/17', '2023/11/24', '2023/12/15', '2023/12/22'
                    ]

# 將 expiration_dates 轉換為 datetime 格式
expiration_dates = pd.to_datetime(expiration_dates)

# 計算 observation_dates，將 expiration_dates 前一日設定為 observation_dates，存入 observation_dates
observation_dates = expiration_dates - pd.Timedelta(days=7)

# 將結果轉回字串格式
observation_dates = observation_dates.strftime('%Y-%m-%d')
expiration_dates = expiration_dates.strftime('%Y-%m-%d')

# 將 expiration_dates 和 observation_dates 設定為 DataFrame 的欄位
df_regression_week['observation_dates'] = observation_dates
df_regression_week['expiration_dates'] = expiration_dates


''' 迴歸分析資料整理_每週_一個點方法 '''
# 儲存每次執行的總時間
total_times = []
execution_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 執行10次
for iteration in range(10):
    print(f"\n執行第 {iteration + 1} 次...")
    
    # 開始計時
    start_time = time.time()

    # 建立儲存統計資料的 list
    stats_data = []

    # 對每一組日期進行計算
    for obs_date, exp_date in zip(df_regression_week['observation_dates'], df_regression_week['expiration_dates']):
        try:
            # 記錄每次迭代的開始時間
            iteration_start_time = time.time()
     
            # 設定全域變數
            global observation_date, expiration_date, df_options_mix
            observation_date = obs_date
            expiration_date = exp_date
            
            # 讀取資料並進行 RND 計算
            call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
            
            F = find_F2()
            
            get_FTS()
            
            df_options_mix = mix_cp_function_v2()
            
            smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
            
            fit = RND_function(smooth_IV)
            
            # 在進行 GPD 尾端擬合前，確保 fit DataFrame 的索引是連續的
            fit = fit.reset_index(drop=True)
            
            # GPD 尾端擬合
            try:
                fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(
                    fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95
                )
            except Exception as e:
                print(f"GPD 擬合失敗：{str(e)}")
                print("嘗試使用備用方法...")
                raise e
            
            # 計算執行時間
            iteration_time = time.time() - iteration_start_time
            
            stats_data.append({
                'observation_date': obs_date,
                'expiration_date': exp_date,
                'execution_time': iteration_time,
                'status': 'success'
            })
            
        except Exception as e:
            stats_data.append({
                'observation_date': obs_date,
                'expiration_date': exp_date,
                'execution_time': time.time() - iteration_start_time,
                'status': 'failed',
                'error': str(e)
            })
            print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
            print(f"錯誤訊息：{str(e)}")
            continue

    # 計算總執行時間
    total_time = time.time() - start_time
    total_times.append(total_time)

    # 將統計資料轉換為 DataFrame
    df_regression_week_stats = pd.DataFrame(stats_data)

# 計算平均執行時間
average_time = sum(total_times) / len(total_times)
print(f"\n10次執行的平均時間：{average_time:.2f} 秒")
print(f"最短執行時間：{min(total_times):.2f} 秒")
print(f"最長執行時間：{max(total_times):.2f} 秒")

# 建立執行時間統計資料
execution_stats = {
    'execution_datetime': execution_datetime,
    'iteration_times': total_times,
    'average_time': average_time,
    'min_time': min(total_times),
    'max_time': max(total_times)
}

# 將統計資料轉換為 DataFrame
df_execution_stats = pd.DataFrame({
    '執行時間': [execution_datetime],
    '平均執行時間(秒)': [average_time],
    '最短執行時間(秒)': [min(total_times)],
    '最長執行時間(秒)': [max(total_times)]
})

# 加入每次迭代的執行時間
for i, time_value in enumerate(total_times, 1):
    df_execution_stats[f'第{i}次執行時間(秒)'] = time_value

# 將結果儲存為 CSV
output_filename = f'execution_times_one_point_{today}.csv'
df_execution_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n執行時間統計資料已儲存至 {output_filename}")


''' 迴歸分析資料整理_每週_兩個點方法 '''
# 開始計時
start_time = time.time()
execution_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 建立儲存統計資料的 list
stats_data = []
total_times = []

# 執行10次
for iteration in range(10):
    print(f"\n執行第 {iteration + 1} 次...")
    
    # 開始計時
    iteration_start_time = time.time()

    # 對每一組日期進行計算
    for obs_date, exp_date in zip(df_regression_week['observation_dates'], df_regression_week['expiration_dates']):
        try:
            # 記錄每次迭代的開始時間
            start_time = time.time()
            
            # 設定全域變數
            global observation_date, expiration_date
            observation_date = obs_date
            expiration_date = exp_date
            
            # 讀取資料並進行 RND 計算
            call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
            
            F = find_F2()
            
            get_FTS()
            
            df_options_mix = mix_cp_function_v2()
            
            smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
            
            fit = RND_function(smooth_IV)
            
            # GPD 尾端擬合
            fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(
                fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98
            )
            
            # 計算執行時間
            iteration_time = time.time() - start_time
            
            stats_data.append({
                'observation_date': obs_date,
                'expiration_date': exp_date,
                'execution_time': iteration_time,
                'status': 'success'
            })
            
        except Exception as e:
            stats_data.append({
                'observation_date': obs_date,
                'expiration_date': exp_date,
                'execution_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            })
            print(f"處理失敗：觀察日 {obs_date}，到期日 {exp_date}")
            print(f"錯誤訊息：{str(e)}")
            continue

    # 計算總執行時間
    total_time = time.time() - iteration_start_time
    total_times.append(total_time)

# 計算平均執行時間
average_time = sum(total_times) / len(total_times)
print(f"\n10次執行的平均時間：{average_time:.2f} 秒")
print(f"最短執行時間：{min(total_times):.2f} 秒")
print(f"最長執行時間：{max(total_times):.2f} 秒")

# 建立執行時間統計資料
execution_stats = {
    'execution_datetime': execution_datetime,
    'iteration_times': total_times,
    'average_time': average_time,
    'min_time': min(total_times),
    'max_time': max(total_times)
}

# 將統計資料轉換為 DataFrame
df_execution_stats = pd.DataFrame({
    '執行時間': [execution_datetime],
    '平均執行時間(秒)': [average_time],
    '最短執行時間(秒)': [min(total_times)],
    '最長執行時間(秒)': [max(total_times)]
})

# 加入每次迭代的執行時間
for i, time_value in enumerate(total_times, 1):
    df_execution_stats[f'第{i}次執行時間(秒)'] = time_value

# 將結果儲存為 CSV
output_filename = f'execution_times_two_points_{today}.csv'
df_execution_stats.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n執行時間統計資料已儲存至 {output_filename}")


''' Function '''
# 讀取資料
def read_data_v2(expiration_date):
    #formatted_date = datetime.strptime(expiration_date, "%Y-%m-%d").strftime("%d%b%y").upper()
    call_iv = pd.read_csv(f"deribit_data/iv/call/call_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    put_iv = pd.read_csv(f"deribit_data/iv/put/put_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    df_idx = pd.read_csv(f"deribit_data/BTC-index/BTC_index_{expiration_date}.csv", index_col="Unnamed: 0")

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"deribit_data/BTC-call/call_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    put_price = pd.read_csv(f"deribit_data/BTC-put/put_strike_{expiration_date}.csv", index_col="Unnamed: 0")

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)
    
    #df_F = find_F_df(call_iv, put_iv, call_price, put_price, df_idx)

    return call_iv, put_iv, call_price, put_price, df_idx#, df_F


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
    smooth_IV["RND"] =  (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / ((dk)**2) #np.exp(r*T) *
    smooth_IV = smooth_IV.dropna()

    # RND 平滑
    smooth_IV["RND"] = savgol_filter(smooth_IV["RND"], 500, 3) # 平滑

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
    # 預先計算常用值
    strike_prices = fit['strike_price'].values
    rnd_density = fit['RND_density'].values
    left_cumulative = fit['left_cumulative'].values
    right_cumulative = fit['right_cumulative'].values
    
    # 設定接合位置
    left_tail_point = alpha_1L
    if left_cumulative[0] > alpha_1L:
        left_tail_point = left_cumulative[0] + 0.001
        print(f"警告：left_cumulative[0] ({left_cumulative[0]:.4f}) 大於 alpha_1L ({alpha_1L})。將 left_tail_point 設為 {left_tail_point:.4f}")

    # Right-tail optimization
    right_idx = np.abs(left_cumulative - alpha_1R).argmin()
    right_end = strike_prices[right_idx]
    missing_tail = right_cumulative[right_idx]
    right_sigma = missing_tail / rnd_density[right_idx]
    
    # 優化斜率計算
    i = initial_i
    while right_idx - i >= 0:
        loc_slope = (-rnd_density[right_idx] + rnd_density[right_idx-i]) / (i * delta_x)
        if loc_slope < 0:
            break
        i += 1
    
    # 右尾部目標函數
    def right_func(params):
        xi, scale = params
        X_alpha_1R = strike_prices[right_idx]
        density_error = ((missing_tail * gpd.pdf(X_alpha_1R + delta_x, xi, loc=X_alpha_1R, scale=scale) 
                         - rnd_density[right_idx]) / delta_x - loc_slope)
        cdf_error = (gpd.cdf(X_alpha_1R, xi, loc=X_alpha_1R, scale=scale) - right_cumulative[right_idx])
        return (1e12 * density_error**2) + (1e12 * cdf_error**2)

    # 使用快速優化方法
    right_result = minimize(right_func, [0, right_sigma], 
                          bounds=[(-1, 1), (0, np.inf)],
                          method='SLSQP',
                          options={'ftol': 1e-8, 'maxiter': 100})
    right_xi, right_sigma = right_result.x

    # Left-tail optimization
    strike_max = strike_prices[-1]
    reverse_strikes = strike_max - strike_prices
    left_idx = np.abs(left_cumulative - left_tail_point).argmin()
    left_end = strike_prices[left_idx]
    missing_tail = left_cumulative[left_idx]
    left_sigma = missing_tail / rnd_density[left_idx]
    
    # 優化左尾部斜率計算
    i = initial_i
    while True:
        loc_slope = (-rnd_density[left_idx] + rnd_density[left_idx-i]) / (i * delta_x)
        if loc_slope < 0:
            break
        i += 1
    
    # 左尾部目標函數
    def left_func(params):
        xi, scale = params
        X_alpha_1L = reverse_strikes[left_idx]
        density_error = ((missing_tail * gpd.pdf(X_alpha_1L + delta_x, xi, loc=X_alpha_1L, scale=scale) 
                         - rnd_density[left_idx]) / delta_x - loc_slope)
        cdf_error = (gpd.cdf(X_alpha_1L, xi, loc=X_alpha_1L, scale=scale) - left_cumulative[left_idx])
        return (1e12 * density_error**2) + (1e12 * cdf_error**2)

    # 使用快速優化方法
    left_result = minimize(left_func, [0, left_sigma], 
                         bounds=[(-1, 1), (0, np.inf)],
                         method='SLSQP',
                         options={'ftol': 1e-8, 'maxiter': 100})
    left_xi, left_sigma = left_result.x

    # 創建擴展範圍
    left_extension = np.arange(0, strike_prices[0] - delta_x, delta_x)
    right_extension = np.arange(strike_prices[-1] + delta_x, 160010, delta_x)
    
    # 計算擴展部分的密度
    reverse_left = strike_max - left_extension
    left_extra_density = missing_tail * gpd.pdf(reverse_left, left_xi, 
                                               loc=reverse_strikes[left_idx], 
                                               scale=left_sigma)
    right_extra_density = missing_tail * gpd.pdf(right_extension, right_xi, 
                                                loc=strike_prices[right_idx], 
                                                scale=right_sigma)

    # 創建所有價格點
    all_strikes = np.concatenate([left_extension, strike_prices, right_extension])
    
    # 創建對應的密度值數組
    all_densities = np.zeros(len(all_strikes))
    
    # 計算各區段的起始和結束位置
    left_end_idx = len(left_extension)
    middle_end_idx = left_end_idx + len(strike_prices)
    
    # 填充左側密度值
    all_densities[:left_end_idx] = left_extra_density
    
    # 填充中間密度值
    all_densities[left_end_idx:middle_end_idx] = rnd_density
    
    # 填充右側密度值
    all_densities[middle_end_idx:] = right_extra_density
    
    # 創建結果 DataFrame
    result = pd.DataFrame({
        'strike_price': all_strikes,
        'full_density': all_densities,
        'reverse_strike': strike_max - all_strikes
    })
    
    # 計算累積密度
    result['full_density_cumulative'] = result['full_density'].cumsum() * delta_x

    return result, left_end, right_end


# 定義擬合 GPD 的函數，選 2 個點，比較 PDF
def fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98):
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

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(fit['strike_price'].max() + delta_x, 160010, delta_x)}), how='outer')
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