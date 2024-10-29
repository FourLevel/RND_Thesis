import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime, timedelta
from scipy.optimize import bisect, minimize
from scipy.stats import norm, genextreme
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, CubicSpline, interp1d
from plotly.subplots import make_subplots
from scipy.stats import genpareto as gpd
from scipy.integrate import quad
import os
import re
import asyncio
import nest_asyncio
import warnings
import calendar
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)


# RND main
initial_i = 1
delta_x = 0.1 
observation_date = "2021-05-12"
expiration_date = "2021-06-25"
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
plot_implied_volatility(df_options_mix)
smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
fit = RND_function(smooth_IV)
plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date)


'''
# 擬合 GPD 的函數，選 1 個點，僅比較斜率，不要用這個方法了
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_with_one_point(fit, initial_i, delta_x)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)
# 計算 RND 曲線統計量並繪製具有分位數的 RND 曲線
stats = calculate_rnd_statistics(fit, delta_x)
quants = list(stats['quantiles'].values())
plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date)
print(f"  平均值: {stats['mean']:.4f}")
print(f"  標準差: {stats['std']:.4f}")
print(f"    偏度: {stats['skewness']:.4f}")
print(f"    峰度: {stats['kurtosis']:.4f}")
print()
'''


''' 擬合 GPD 的函數，選 1 個點，比較斜率與 CDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)
# 計算 RND 曲線統計量並繪製具有分位數的 RND 曲線
stats = calculate_rnd_statistics(fit, delta_x)
quants = list(stats['quantiles'].values())
plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date)
print(f"  平均值     Mean: {stats['mean']:.4f}")
print(f"  標準差      Std: {stats['std']:.4f}")
print(f"    偏度 Skewness: {stats['skewness']:.4f}")
print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
print()


''' 擬合 GPD 的函數，選 2 個點，比較 PDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)
# 計算 RND 曲線統計量並繪製具有分位數的 RND 曲線
stats = calculate_rnd_statistics(fit, delta_x)
quants = list(stats['quantiles'].values())
plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date)
print(f"  平均值     Mean: {stats['mean']:.4f}")
print(f"  標準差      Std: {stats['std']:.4f}")
print(f"    偏度 Skewness: {stats['skewness']:.4f}")
print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
print()


''' 於同一張圖繪製多條 RND 曲線，自訂日期 '''
# observation_dates = input('請輸入觀測日期，以逗點分隔: ').split(',')
# observation_dates = [date.strip() for date in observation_dates]
# observation_dates = ['20240731', '20240801', '20240802', '20240805', '20240806', '20240807']
# expiration_date = input('請輸入到期日期: ')
# expiration_date = '202408'
observation_dates = ['2022-09-06', '2022-10-10','2022-11-09', '2022-12-09', '2023-01-09', '2023-02-09', '2023-03-09']
expiration_date = '2023-03-31'
all_stats, all_rnd_data = process_multiple_dates_one_point(observation_dates, expiration_date)
plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date)

# 印出每個日期的 mean, std, skewness, kurtosis
print("每個日期的統計數據：")
for date in observation_dates:
    stats = all_stats[date]
    print(f"{date}:")
    print(f"  平均值     Mean: {stats['mean']:.4f}")
    print(f"  標準差      Std: {stats['std']:.4f}")
    print(f"    偏度 Skewness: {stats['skewness']:.4f}")
    print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
    print()


''' 於同一張圖繪製多條 RND 曲線，僅需輸入起始日和最終日 '''
# 輸入起始日和最終日
start_date = '2021-04-14'
end_date = '2021-06-09'
expiration_date = '2021-06-25'

# 生成日期列表
observation_dates = generate_dates(start_date, end_date, interval_days=7) # interval_days 可設定間隔天數

# 處理數據並繪圖
all_stats, all_rnd_data = process_multiple_dates_two_points(observation_dates, expiration_date) # 使用不同方法可調整函數
plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date)

# 印出每個日期的統計數據
print("每個日期的統計數據：")
for date in observation_dates:
    if date in all_stats:
        stats = all_stats[date]
        print(f"{date}:")
        print(f"  平均值     Mean: {stats['mean']:.4f}")
        print(f"  標準差      Std: {stats['std']:.4f}")
        print(f"    偏度 Skewness: {stats['skewness']:.4f}")
        print(f"    峰度 Kurtosis: {stats['kurtosis']:.4f}")
        print()
    else:
        print(f"{date}: 無可用數據")
        print()


''' 整理統計數據 '''
# 整理統計數據
stats_data = []
for date in observation_dates:
    if date in all_stats:
        stats = all_stats[date]
        stats_data.append({
            '日期': date,
            '平均值 Mean': f"{stats['mean']:.4f}",
            '標準差 Std': f"{stats['std']:.4f}",
            '偏度 Skewness': f"{stats['skewness']:.4f}",
            '峰度 Kurtosis': f"{stats['kurtosis']:.4f}"
        })
    else:
        stats_data.append({
            '日期': date,
            '平均值 Mean': 'N/A',
            '標準差 Std': 'N/A',
            '偏度 Skewness': 'N/A',
            '峰度 Kurtosis': 'N/A'
        })

# 創建 DataFrame
df_stats = pd.DataFrame(stats_data)

# 匯出成 CSV 檔
csv_filename = f'RND_stats_{start_date}_to_{end_date}_exp_{expiration_date}.csv'
df_stats.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"統計數據已匯出至 {csv_filename}")

# 印出每個日期的統計數據
print("每個日期的統計數據：")
print(df_stats.to_string(index=False))


''' 回推買權價格 '''   
# 計算單個買權價格
strike_price = 40000  # 假設行權價為 100
call_option_price = calculate_call_option_price_discrete(fit, strike_price)
print(f"買權價格 Call Option Price: {call_option_price:.4f}")

# 計算所有大於 future_price 的行權價的買權價格，每隔 1000 個計算一次
future_price = F  # 設定 future_price
call_option_prices = calculate_call_option_prices_above_future_price(fit, future_price, step=1000)

for strike_price, call_price in call_option_prices.items():
    print(f"Strike Price: {strike_price:.2f} 的買權價格: {call_price:.4f}")

# 繪製圖表
plt.figure(figsize=(10, 6), dpi=100)
# 擬合的買權價格
plt.plot(fit['strike_price'], fit['fit_call'], color='orange', label='Fitted Call Price')
# 計算的買權價格
strike_prices = list(call_option_prices.keys())
call_prices = list(call_option_prices.values())
plt.plot(strike_prices, call_prices, linestyle='-', color='blue', label='Calculated Call Price')
plt.title(f'Call Option Prices for Strike Prices Above Future Price (Future Price: {F:.2f})')
plt.xlabel('Strike Price')
plt.ylabel('Call Option Price')
# plt.xlim(30000, 100000)
# plt.ylim(0, 5000)
plt.legend()
plt.grid(True)
plt.show()



----------------------------------------------------------------------------------------------------------------

# 輸入起始日和最終日
start_date = '2021-04-14'
end_date = '2021-06-09'
expiration_date = '2021-06-25'

# 生成日期列表
observation_dates = generate_dates(start_date, end_date, interval_days=7) # interval_days 可設定間隔天數

# 處理數據
all_stats, all_rnd_data, all_call_option_prices = find_call_option_prices_above_future_price_multiple_dates_two_points(observation_dates, expiration_date) # 使用不同方法可調整函數

# 印出每個日期的 call option prices
for date in observation_dates:
    print(f"{date}: {all_call_option_prices[date]}")


# 創建一個空的列表來存儲 DataFrame
dataframes = []

# 填充 DataFrame
for date in observation_dates:
    call_prices = all_call_option_prices[date]
    df = pd.DataFrame(call_prices, index=[date])
    dataframes.append(df)

# 使用 pd.concat() 將所有 DataFrame 合併
df_call_option_prices = pd.concat(dataframes)

# 將 NaN 值填充為空字串，方便顯示
df_call_option_prices.fillna('', inplace=True)

# 打印 DataFrame
print(df_call_option_prices)

# 如果需要，可以將 DataFrame 匯出為 CSV 檔案
# df_call_option_prices.to_csv('call_option_prices.csv', index=True, encoding='utf-8')

# 找尋 call_price 變數中，index 與 df_call_option_prices 的 index 相同的值，並比對 column name，向下增加 row
for date in df_call_option_prices.index:
    if date in call_price.index:
        # 將 call_price 中的值添加到 df_call_option_prices 中的新行
        new_row = call_price.loc[date].reindex(df_call_option_prices.columns)
        df_call_option_prices = pd.concat([df_call_option_prices, pd.DataFrame(new_row).T], ignore_index=False)

# 將 df_call_option_prices 中有 missing value 的 column 刪除
df_call_option_prices = df_call_option_prices.dropna(axis=1)

# 打印更新後的 DataFrame
print("更新後的 df_call_option_prices：")
print(df_call_option_prices)


----------------------------------------------------------------------------------------------------------------




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
    #print(np.array(F_values_c+F_values_p))
    
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


# 定義 UnivariateSpline 函數
def UnivariateSpline_function_v2(mix_cp, power=3, s=None, w=None):
    global observation_date, expiration_date, delta_x
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    spline = UnivariateSpline(mix_cp.index, mix_cp["mixIV"], k=4, s=None, w=None)
    
    min_K = 0 #int(min(oneday2["K"]) * 0.8) # 測試!!!!!!!!!!
    max_K = int(max(mix_cp.index)*1.2)#max(F*2//1000*1000, max(mix_cp.index)) +1
    dK = delta_x
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame([K_fine, Vol_fine], index=["K", "mixIV"]).T
    """
    try:    # IV左邊有往下
        left_US = smooth_IV.query(f"K < {mix_cp.index[0]}")
        idx = left_US[left_US["mixIV"].diff() > 0].index[-1]
        smooth_IV = smooth_IV.loc[idx:].reset_index(drop=True)
    except: # IV左邊沒有往下
        pass
    """
    smooth_IV["C"] = call.future(F, smooth_IV["K"], T, smooth_IV["mixIV"], S)
    
    #smooth_IV = add_other_info(date, oneday2, smooth_IV, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname)
    #smooth_IV.index = smooth_IV["K"].values
    return smooth_IV


# 定義繪製隱含波動率圖表的函數
def plot_implied_volatility(mix_cp):
    global observation_date, expiration_date
    basicinfo = get_FTS()
    futures_price = basicinfo["F"]

    # 繪製買權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for call options')
    plt.show()

    # 繪製賣權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for put options')
    plt.show()

    # 繪製買權與賣權的隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange', label='Call')
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue', label='Put')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5)
    plt.text(futures_price + 150, plt.gca().get_ylim()[0] + 0.1, futures_price, transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


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


# 定義繪製擬合曲線的函數
def plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date):
    # 繪製隱含波動率微笑擬合圖
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(df_options_mix.index, df_options_mix['mixIV'], label='Observed IV')
    plt.plot(fit['strike_price'], fit['fit_imp_vol'], color='orange', label='Fitted IV')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility Smile of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製買權曲線
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fit['strike_price'], fit['fit_call'], color='orange', label='Fitted Call Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Price')
    plt.title(f'Call Curve of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製經驗風險中性密度 (PDF)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fit['strike_price'], fit['RND_density'], color='orange')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.title(f'Empirical Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.show()

    # 繪製經驗風險中性累積分佈函數 (CDF)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fit['strike_price'], fit['left_cumulative'], color='orange')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.show()


'''
# 定義擬合 GPD 的函數，選 1 個點，僅比較斜率，不要用這個方法了
def fit_gpd_tails_use_slope_with_one_point(fit, initial_i, delta_x, alpha1L=0.05, alpha1R=0.95):
    # 設定接合位置
    left_tail_point = alpha1L
    right_tail_point = alpha1R

    # Right-tail
    loc = fit.iloc[(fit['left_cumulative'] - right_tail_point).abs().argsort()[:1]]
    right_end = loc['strike_price'].values[0]
    missing_tail = loc['right_cumulative'].values[0]
    right_sigma = missing_tail / loc['RND_density'].values[0]
    max_strike_index = fit['strike_price'].idxmax()
    end_density = fit.loc[max_strike_index, 'RND_density']

    i = initial_i
    while True:
        loc_slope = (-loc['RND_density'].values[0] + fit.iloc[fit.index.get_loc(loc.index[0])+i]['RND_density']) / (i * delta_x)
        if loc_slope < 0:
            break
        else:
            i += 1

    def right_func(x):
        xi = x[0]
        error = ((missing_tail * gpd.pdf(loc['strike_price'].values[0] + delta_x, xi, loc=loc['strike_price'].values[0], scale=right_sigma) - loc['RND_density'].values[0]) / delta_x - loc_slope)
        return (1e12 * error**2)

    right_fit = minimize(right_func, [-2], bounds=[(-np.inf, np.inf)], method='L-BFGS-B')
    right_xi = right_fit.x[0]

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
        xi = x[0]
        error = (missing_tail * gpd.pdf(loc['reverse_strike'].values[0] + delta_x, xi, loc=loc['reverse_strike'].values[0], scale=left_sigma) - loc['RND_density'].values[0]) / delta_x - loc_slope
        return (1e12 * error**2)

    left_fit = minimize(left_func, [-2], bounds=[(-np.inf, np.inf)], method='L-BFGS-B')
    left_xi = left_fit.x[0]

    fit = pd.merge(fit, pd.DataFrame({'strike_price': np.arange(0, fit['strike_price'].min() - delta_x, delta_x)}), how='outer')
    fit['reverse_strike'] = fit['strike_price'].max() - fit['strike_price']
    fit['left_extra_density'] = missing_tail * gpd.pdf(fit['reverse_strike'], left_xi, loc=loc['reverse_strike'].values[0], scale=left_sigma)
    fit['full_density'] = np.where(fit['strike_price'] < loc['strike_price'].values[0], fit['left_extra_density'], fit['full_density'])

    fit['full_density_cumulative'] = fit['full_density'].cumsum() * delta_x

    # 找到 CDF 的 Lower Bound and Upper Bound
    lower_bound = left_end
    upper_bound = right_end

    return fit, lower_bound, upper_bound
'''



# 定義擬合 GPD 的函數，選 1 個點，比較斜率與 CDF
def fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95):
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
    upper_bound = fit.loc[(fit['full_density_cumulative'] - alpha_2R).abs().idxmin(), 'strike_price']

    return fit, lower_bound, upper_bound


# 定義繪製擬合 GPD 的函數
def plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date):
    # RND
    plt.figure(figsize=(10, 6), dpi=100)
    # 原始 RND
    plt.plot(fit['strike_price'], fit['full_density'], label='Empirical RND', color='royalblue')
    # 左尾 GPD
    left_tail = fit[fit['strike_price'] <= upper_bound]
    plt.plot(left_tail['strike_price'], left_tail['left_extra_density'], 
             label='Left tail GPD', color='orange', linestyle=':', linewidth=2)
    # 右尾 GPD
    right_tail = fit[fit['strike_price'] >= lower_bound]
    plt.plot(right_tail['strike_price'], right_tail['right_extra_density'], 
             label='Right tail GPD', color='green', linestyle=':', linewidth=2)
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義繪製完整密度累積分佈函數的函數
def plot_full_density_cdf(fit, observation_date, expiration_date):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fit['strike_price'], fit['full_density_cumulative'])
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.show()


# 定義計算和繪製 RND 統計量的函數
def calculate_rnd_statistics(fit, delta_x):
    # 計算統計量
    RND_mean = np.sum(fit['strike_price'] * fit['full_density'] * delta_x)
    RND_std = np.sqrt(np.sum((fit['strike_price'] - RND_mean)**2 * fit['full_density'] * delta_x))
    
    fit['std_strike'] = (fit['strike_price'] - RND_mean) / RND_std
    RND_skew = np.sum(fit['std_strike']**3 * fit['full_density'] * delta_x)
    RND_kurt = np.sum(fit['std_strike']**4 * fit['full_density'] * delta_x) - 3

    # 計算分位數
    # fit['left_cumulative'] = np.cumsum(fit['full_density'] * delta_x)
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quants = [fit.loc[(fit['left_cumulative'] - q).abs().idxmin(), 'strike_price'] for q in quantiles]

    # 返回數據
    return {
        'mean': RND_mean,
        'std': RND_std,
        'skewness': RND_skew,
        'kurtosis': RND_kurt,
        'quantiles': dict(zip(quantiles, quants)),
        'rnd_data': fit[['strike_price', 'full_density']]
    }


# 定義繪製 RND 圖形及分位數的函數
def plot_rnd_with_quantiles(fit, quants, observation_date, expiration_date):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fit['strike_price'], fit['full_density'], label='RND')
    for quant in quants:
        plt.axvline(x=quant, linestyle='--', color='gray')
    plt.xlabel('Strike Price')
    plt.ylabel('RND')
    plt.title(f'Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# 定義處理多個日期的函數，使用兩點的方法
def process_multiple_dates_two_points(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_1L=0.02, alpha_2L=0.05, alpha_1R=0.95, alpha_2R=0.98)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data


# 定義處理多個日期的函數，使用一點的方法
def process_multiple_dates_one_point(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data


# 繪製多個日期的 RND
def plot_multiple_rnd(all_rnd_data, observation_dates, expiration_date):
    plt.figure(figsize=(12, 8), dpi=100)
       
    for date in observation_dates:
        fit = all_rnd_data[date]
        plt.plot(fit['strike_price'], fit['full_density'], label=date)
       
    plt.xlabel('Strike Price')
    plt.ylabel('RND')
    plt.title(f'Multiple Dates Risk-Neutral Density Curve (Expiration Date: {expiration_date})')
    plt.legend(title='Observation Date', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# 生成日期列表
def generate_dates(start_date, end_date, interval_days=1):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    return date_list


# 使用積分反推買權價格
def calculate_call_option_price_discrete(fit, strike_price):
    # 假設 'x' 是資產價格，'pdf' 是對應的概率密度
    x_values = fit['strike_price'].values
    pdf_values = fit['full_density'].values
    
    # 計算期權價格的離散積分
    call_payoffs = np.maximum(x_values - strike_price, 0)
    call_price = np.trapz(call_payoffs * pdf_values, x_values)
    
    return call_price


# 計算所有大於 future_price 的行權價的買權價格，每隔 50 個計算一次
def calculate_call_option_prices_above_future_price(fit, future_price, step=50):
    # 假設 'x' 是資產價格，'pdf' 是對應的概率密度
    x_values = fit['strike_price'].values
    pdf_values = fit['full_density'].values
    
    # 找到所有大於 future_price 的行權價
    strike_prices = x_values[x_values > future_price]
    
    # 每隔 step 個行權價計算一次
    selected_strike_prices = [sp for sp in strike_prices if sp % step == 0]
    
    # 計算每個選定行權價的買權價格
    call_option_prices = {}
    for strike_price in selected_strike_prices:
        call_payoffs = np.maximum(x_values - strike_price, 0)
        call_price = np.trapz(call_payoffs * pdf_values, x_values)
        call_option_prices[strike_price] = call_price
    
    return call_option_prices



# 定義處理多個日期的函數，使用兩點的方法
def find_call_option_prices_above_future_price_multiple_dates_two_points(observation_dates, expiration_date):
    global observation_date, call_iv, put_iv, call_price, put_price, df_idx, F, df_options_mix, delta_x
    all_stats = {}
    all_rnd_data = {}
    all_call_option_prices = {}

    # 只讀取一次數據
    call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

    for observation_date in observation_dates:
        try:
            delta_x = delta_x
            F = find_F2()
            get_FTS()
            df_options_mix = mix_cp_function_v2()
            smooth_IV = UnivariateSpline_function_v2(df_options_mix, power=4)
            fit = RND_function(smooth_IV)
            fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_1L=0.02, alpha_2L=0.05, alpha_1R=0.95, alpha_2R=0.98)
            call_option_prices = calculate_call_option_prices_above_future_price(fit, future_price, step=1000)
            stats = calculate_rnd_statistics(fit, delta_x)
            all_stats[observation_date] = stats
            all_rnd_data[observation_date] = fit
            all_call_option_prices[observation_date] = call_option_prices
        except Exception as e:
            print(f"處理日期 {observation_date} 時出錯：{str(e)}")
            continue

    return all_stats, all_rnd_data, all_call_option_prices