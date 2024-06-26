# import
# import
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

from scipy.optimize import bisect, minimize
from scipy.stats import norm, genextreme
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, CubicSpline

import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import os
import re

import asyncio
import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.6f}'.format)

from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

hello()



def read_data_v2(expiration_date):

    #formatted_date = datetime.strptime(expiration_date, "%Y-%m-%d").strftime("%d%b%y").upper()

    call_iv = pd.read_csv(f"RND_data/asher_data/For RND/call_iv_{expiration_date}.csv", index_col="Unnamed: 0")
    call_iv.index = pd.to_datetime(call_iv.index)
    call_iv.index = call_iv.index.strftime('%Y-%m-%d')
    put_iv = pd.read_csv(f"RND_data/asher_data/For RND/put_iv_{expiration_date}.csv", index_col="Unnamed: 0")
    put_iv.index = pd.to_datetime(put_iv.index)
    put_iv.index = put_iv.index.strftime('%Y-%m-%d')
    df_idx = pd.read_csv(f"RND_data/asher_data/For RND/TX_index_{expiration_date}.csv", index_col="Unnamed: 0")
    df_idx.index = pd.to_datetime(df_idx.index)
    df_idx.index = df_idx.index.strftime('%Y-%m-%d')
    df_idx['index_price'] = df_idx['index_price'].str.replace(',', '').astype(float)

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"RND_data/asher_data/For RND/call_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    call_price.index = pd.to_datetime(call_price.index)
    call_price.index = call_price.index.strftime('%Y-%m-%d')
    put_price = pd.read_csv(f"RND_data/asher_data/For RND/put_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    put_price.index = pd.to_datetime(put_price.index)
    put_price.index = put_price.index.strftime('%Y-%m-%d')

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)
    
    #df_F = find_F_df(call_iv, put_iv, call_price, put_price, df_idx)

    return call_iv, put_iv, call_price, put_price, df_idx#, df_F





def find_F1(K, type):
    global observe_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
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
        price = call_price[K].loc[observe_date]  
        sigma = call_iv[K].loc[observe_date]
    if type=="P":
        price = put_price[K].loc[observe_date]  
        sigma = put_iv[K].loc[observe_date]
    
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observe_date)).days /365 
    S = df_idx["index_price"].loc[observe_date]

    return newton_method(price, K, sigma, T, S, type)







def find_F2():
    global observe_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    S = df_idx["index_price"].loc[observe_date]
    df = call_price.loc[observe_date][call_price.loc[observe_date]!=0]
    result_c = df[(df.index >= S*0.9) & (df.index <= S*1.5)]
    result_c = pd.DataFrame(result_c)
    result_c.columns = ["C"]

    df = put_price.loc[observe_date][put_price.loc[observe_date]!=0]
    result_p = df[(df.index >= S*0.8) & (df.index <= S*1.1)]
    result_p = pd.DataFrame(result_p)
    result_p.columns = ["P"]

    F_values_c = [find_F1(K, "C") for K in result_c.index]
    F_values_p =  [find_F1(K, "P") for K in result_p.index]
    F = np.array(F_values_c+F_values_p).mean()
    #print(np.array(F_values_c+F_values_p))
    
    return F





def draw_IV_and_Call_v2(smooth_IV):
    global call_iv, put_iv, call_price, put_price, df_idx
    F = find_F2()

    # Fig1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=100)
    ax1.set_title(f"{observe_date} date\n{expiration_date} expiration\nsmooth IV(Call, Put)\nF={int(F)}")
    # (1) call iv scatter   
    call_iv_filt = call_iv.loc[observe_date][call_iv.loc[observe_date] != 0]
    ax1.scatter(call_iv_filt.index, call_iv_filt, label="call iv", marker="o", color="mediumseagreen", s=10)
    # (2) put iv scatter     
    put_iv_filt = put_iv.loc[observe_date][put_iv.loc[observe_date] != 0]
    ax1.scatter(put_iv_filt.index, put_iv_filt, label="put iv", marker="o", color="lightcoral", s=10)
    # (3) F
    ax1.plot([F]*2, [0, 0.5],  ":", color="black", label=f"futures price")
    # (4) smooth iv    
    ax1.plot(smooth_IV["K"], smooth_IV["mixIV"], color="royalblue", label=f"smooth iv")  
    ax1.set_xlim(min(put_iv_filt.index), max(call_iv_filt.index))
    ax1.set_ylim(min(min(call_iv_filt), min(put_iv_filt))-0.1, max(max(call_iv_filt), max(put_iv_filt))+0.1)

     # Fig2
    ax2.set_title("smooth call price")  
    # (1) call price scatter   
    call_price_filt = call_price.loc[observe_date][call_price.loc[observe_date] != 0]
    ax2.scatter(call_price_filt.index, call_price_filt, label="call price", marker="o", color="mediumseagreen", s=10)
    # (2) put price scatter     
    #put_price_filt = put_price.loc[observe_date][put_iv.loc[observe_date] != 0]
    #ax2.scatter(put_price_filt.index, put_price_filt, label="put iv", marker="o", color="lightcoral", s=10)
    # (3) smooth price    
    ax2.plot(smooth_IV["K"], smooth_IV["C"], alpha=0.8, label="smooth call price", color="royalblue")
    ax2.set_xlim(min(put_iv_filt.index), max(call_iv_filt.index))
    ax2.set_ylim(0, max(call_price_filt)*1.1)

    ax1.grid(linestyle='--', alpha=0.3), ax2.grid(linestyle='--', alpha=0.3)
    ax1.legend(), ax2.legend()
    plt.show()






def get_FTS():
    global observe_date, expiration_date
    F = find_F2()
    T = (pd.to_datetime(expiration_date) - pd.to_datetime(observe_date)).days/365
    S = df_idx["index_price"].loc[observe_date]
    return {"F": F, "T": T, "S": S}



def mix_cp_function_v2():
    global observe_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    mix = pd.concat([call_iv.loc[observe_date], put_iv.loc[observe_date]], axis=1)
    mix.columns = ["C", "P"]
    mix = mix.replace(0, np.nan)

    # atm
    atm = mix.loc[(mix.index <= F*1.1) & (mix.index >= F*0.9)]
    atm["mixIV"] = atm[["C","P"]].mean(axis=1)

    # otm
    otm = pd.DataFrame(pd.concat([ mix.loc[mix.index < F*0.9, 'P'], mix.loc[mix.index > F*1.1, 'C'] ], axis=0), columns=["mixIV"])

    # mix
    mix_cp = pd.concat([atm, otm], axis=0).sort_index()
    mix_cp[["C","P"]] = mix
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[:F*2.5]
    
    return mix_cp




def UnivariateSpline_function_v2(mix_cp, power=3, s=None, w=None):
    global observe_date, expiration_date
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    spline = UnivariateSpline(mix_cp.index, mix_cp["mixIV"], k=4, s=None, w=None) #三次样条插值，s=0：插值函数经过所有数据点
    
    min_K = 0 #int(min(oneday2["K"]) * 0.8) # 測試!!!!!!!!!!
    max_K = int(max(mix_cp.index)*1.2)#max(F*2//1000*1000, max(mix_cp.index)) +1
    dK = 1
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





# RND main 
observe_date = "2024-05-02"
expiration_date = "2024-05-22"

call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)

F = find_F2()

mix_cp = mix_cp_function_v2()

smooth_IV = UnivariateSpline_function_v2(mix_cp, power=4)

draw_IV_and_Call_v2(smooth_IV)






basicinfo = get_FTS()
F = basicinfo["F"]
T = basicinfo["T"]
S = basicinfo["S"]
r = 1.61/100

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import matplotlib.pyplot as plt

# 假設已經有 smooth_IV dataframe, 包含 K (行使價) 和 mixIV (隱含波動率)

def bs_price(F, K, T, sigma, S, option_type='C'):
    d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'P':
        return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

# 計算期權價格
smooth_IV['C'] = bs_price(F, smooth_IV['K'], T, smooth_IV['mixIV'], S, option_type='C')

# 檢查數據
print(smooth_IV.head())

# 確認數據沒有異常值或缺失值
smooth_IV = smooth_IV.dropna()

# 計算期權價格的二階導數
spline_C = UnivariateSpline(smooth_IV['K'], smooth_IV['C'], k=4, s=None)
second_derivative = spline_C.derivative(n=2)

# 計算風險中性密度 RND
K_values = smooth_IV['K']
RND = np.exp(r * T) * second_derivative(K_values)

# 可視化 RND
plt.plot(K_values, RND, label='Risk Neutral Density')
plt.xlabel('Strike Price (K)')
plt.ylabel('Density')
plt.title('Risk Neutral Density')
plt.legend()
plt.show()