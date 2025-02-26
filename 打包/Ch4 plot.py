# 基本數據處理與分析套件
import pandas as pd
import numpy as np
# 繪圖套件
import matplotlib.pyplot as plt
# 數學與統計相關套件
from scipy.optimize import minimize
from scipy.stats import norm, genpareto as gpd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
# 日期時間處理
from datetime import datetime, timedelta
# 系統與工具套件
import warnings
# 自定義套件
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')

# RND main
initial_i = 1
delta_x = 0.1 
observation_date = "2023-09-27"
expiration_date = "2023-09-28"
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
plot_implied_volatility(df_options_mix)
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4, s=None, w=None)
fit = RND_function(smooth_IV)
plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date)


''' 擬合 GPD 的函數，選 1 個點，比較斜率與 CDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_slope_and_cdf_with_one_point(fit, initial_i, delta_x, alpha_1L=0.05, alpha_1R=0.95)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)



''' 擬合 GPD 的函數，選 2 個點，比較 PDF '''
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()
get_FTS()
df_options_mix = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v3(df_options_mix, power=4)
fit = RND_function(smooth_IV)
fit, lower_bound, upper_bound = fit_gpd_tails_use_pdf_with_two_points(fit, delta_x, alpha_2L=0.02, alpha_1L=0.05, alpha_1R=0.95, alpha_2R=0.98)
# 繪製完整 RND 曲線與完整 CDF 曲線
plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date)
plot_full_density_cdf(fit, observation_date, expiration_date)





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


# 定義繪製隱含波動率圖表的函數
def plot_implied_volatility(mix_cp):
    global observation_date, expiration_date
    basicinfo = get_FTS()
    futures_price = basicinfo["F"]

    # 繪製買權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for call options')
    plt.show()

    # 繪製賣權履約價與隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility vs Strike Price for put options')
    plt.show()

    # 繪製買權與賣權的隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mix_cp.index, mix_cp['C'], color='orange', label='Call IV')
    plt.scatter(mix_cp.index, mix_cp['P'], color='blue', label='Put IV')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製買權與賣權的隱含波動率的散布圖，顯示出 0.9F ~ 1.1F 的 mixIV
    plt.figure(figsize=(10, 6), dpi=200)
    # 0.9F 到 1.1F 內的數據
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp['C'][atm_mask], color='orange', label='Call IV')
    plt.scatter(mix_cp.index[atm_mask], mix_cp['P'][atm_mask], color='blue', label='Put IV')
    plt.scatter(mix_cp.index[atm_mask], mix_cp['mixIV'][atm_mask], color='green', label='Mix IV')
    # 0.9F 到 1.1F 外的數據
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['C'][otm_mask_low], color='orange', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['C'][otm_mask_high], color='orange', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['P'][otm_mask_low], color='blue', alpha=0.5, edgecolors='none')
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['P'][otm_mask_high], color='blue', alpha=0.5, edgecolors='none')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製 mix 後之買權與賣權的隱含波動率的散布圖
    plt.figure(figsize=(10, 6), dpi=200)
    # 0.9F 到 1.1F 內的數據
    atm_mask = (mix_cp.index >= futures_price * 0.9) & (mix_cp.index <= futures_price * 1.1)
    plt.scatter(mix_cp.index[atm_mask], mix_cp['mixIV'][atm_mask], color='green', label='Mix IV')
    # 0.9F 到 1.1F 外的數據
    otm_mask_low = mix_cp.index < futures_price * 0.9
    otm_mask_high = mix_cp.index > futures_price * 1.1
    plt.scatter(mix_cp.index[otm_mask_high], mix_cp['C'][otm_mask_high], color='orange', label='Call IV')
    plt.scatter(mix_cp.index[otm_mask_low], mix_cp['P'][otm_mask_low], color='blue', label='Put IV')
    plt.axvline(x=futures_price, color='black', linestyle='--', alpha=0.5, label='Futures Price')
    plt.text(futures_price + 500, plt.gca().get_ylim()[0] + 0.2, f'F = {int(futures_price)}', transform=plt.gca().transData)
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility vs Strike Price for call and put options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


# RND
def RND_function(smooth_IV):
    """
    計算風險中性機率密度 (Risk-Neutral Density) 及相關分配函數
    
    參數:
        smooth_IV: 包含平滑隱含波動率的資料框
    
    回傳:
        處理後的資料框，包含 RND 密度和累積分配函數
    """
    # 計算步長
    dk = smooth_IV["K"].iloc[1] - smooth_IV["K"].iloc[0]
    
    # 計算累積分配函數 (CDF)
    smooth_IV["left_cumulative"] = np.gradient(smooth_IV['C'], smooth_IV['K']) + 1
    smooth_IV["right_cumulative"] = 1 - smooth_IV["left_cumulative"]
    
    # 計算機率密度函數 (PDF)
    smooth_IV["pdf"] = np.gradient(np.gradient(smooth_IV['C'], smooth_IV['K']), smooth_IV['K'])
    
    # 使用有限差分法計算 RND
    smooth_IV["RND_density"] = (smooth_IV["C"].shift(1) + smooth_IV["C"].shift(-1) - 2*smooth_IV["C"]) / (dk**2)
    
    # 移除計算過程中產生的 NaN 值
    smooth_IV = smooth_IV.dropna()
    
    # 使用 Savitzky-Golay 濾波器平滑 RND
    smooth_IV["RND_density"] = savgol_filter(smooth_IV["RND_density"], 500, 3)
    
    # 只保留有效範圍內的資料
    if 'df_options_mix' in globals():
        smooth_IV = smooth_IV[(smooth_IV['K'] >= df_options_mix.index.min()) & 
                              (smooth_IV['K'] <= df_options_mix.index.max())]
    
    # 過濾無效資料
    smooth_IV = smooth_IV[(smooth_IV['right_cumulative'].notna()) & 
                          (smooth_IV['left_cumulative'].notna()) &
                          (smooth_IV['right_cumulative'] < 1) & 
                          (smooth_IV['right_cumulative'] > 0) &
                          (smooth_IV['left_cumulative'] < 1) &
                          (smooth_IV['left_cumulative'] > 0)]
    
    # 重新命名欄位以提高可讀性
    smooth_IV = smooth_IV.rename(columns={
        'K': 'strike_price', 
        'mixIV': 'fit_imp_vol', 
        'C': 'fit_call'
    })
    
    return smooth_IV



# 定義繪製擬合曲線的函數
def plot_fitted_curves(df_options_mix, fit, observation_date, expiration_date):
    # 繪製隱含波動率微笑擬合圖
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(df_options_mix.index, df_options_mix['mixIV'],color='green', label='Mix IV')
    plt.plot(fit['strike_price'], fit['fit_imp_vol'], color='orange', label='Fitted IV')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility Smile of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製買權曲線
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['fit_call'], color='orange', label='Fitted Call Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Price')
    plt.title(f'Call Curve of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製經驗風險中性密度 (PDF)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['RND_density'], color='orange', label='Empirical RND')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.title(f'Empirical Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()

    # 繪製經驗風險中性累積分佈函數 (CDF)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['left_cumulative'], color='orange', label='CDF')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()


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


# 定義繪製擬合 GPD 的函數
def plot_gpd_tails(fit, lower_bound, upper_bound, observation_date, expiration_date):
    # RND
    plt.figure(figsize=(10, 6), dpi=200)
    
    # 設定 y 軸格式為 10^n
    from matplotlib.ticker import ScalarFormatter
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 原始 RND
    plt.plot(fit['strike_price'], fit['full_density'], label='Empirical RND', color='royalblue')
    
    # 找出 CDF 為 5% 和 95% 的點
    left_point = fit.loc[(fit['left_cumulative'] - 0.05).abs().idxmin()]
    right_point = fit.loc[(fit['left_cumulative'] - 0.95).abs().idxmin()]
    
    # 左尾 GPD
    left_tail = fit[fit['strike_price'] <= upper_bound]
    plt.plot(left_tail['strike_price'], left_tail['left_extra_density'], 
             label='Left tail GPD', color='orange', linestyle=':', linewidth=2)
    
    # 右尾 GPD
    right_tail = fit[fit['strike_price'] >= lower_bound]
    plt.plot(right_tail['strike_price'], right_tail['right_extra_density'], 
             label='Right tail GPD', color='green', linestyle=':', linewidth=2)
    
    # 在 5% 和 95% 的點加上黑色空心圓圈
    plt.plot(left_point['strike_price'], left_point['full_density'], 'o', 
             color='black', fillstyle='none', markersize=10)
    plt.plot(right_point['strike_price'], right_point['full_density'], 'o', 
             color='black', fillstyle='none', markersize=10)
    
    # 添加文字標註
    plt.annotate(r'$\alpha_{1L}=0.05$', 
                xy=(left_point['strike_price'], left_point['full_density']),
                xytext=(-65, 5), textcoords='offset points',
                fontsize=12)
    plt.annotate(r'$\alpha_{1R}=0.95$', 
                xy=(right_point['strike_price'], right_point['full_density']),
                xytext=(3, 5), textcoords='offset points',
                fontsize=12)
    
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    # plt.xlim(10000, 30000)
    plt.legend()
    plt.show()


# 定義繪製完整密度累積分佈函數的函數
def plot_full_density_cdf(fit, observation_date, expiration_date):
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['full_density_cumulative'], label='CDF')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability')
    plt.title(f'Empirical Risk-Neutral Probability of BTC options on {observation_date} (expired on {expiration_date})')
    # plt.xlim(F*0.8, F*1.2)
    plt.legend()
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
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fit['strike_price'], fit['full_density'], label='Empirical RND')
    for quant in quants:
        plt.axvline(x=quant, linestyle='--', color='gray')
    plt.xlabel('Strike Price')
    plt.ylabel('RND')
    plt.title(f'Risk-Neutral Density of BTC options on {observation_date} (expired on {expiration_date})')
    plt.legend()
    plt.show()