"""
Black Scholes Model

class:
    d1
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)
        
    d2
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    call
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    put
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    implied_volatility                   # 給定其他參數，找出IV
        call(F, K, T, r, call_price)
        put(F, K, T, r, put_price)
"""


import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, CubicSpline


class d1:
    def spot(S, K, T, r, sigma):
        return ( np.log(S/K) + (r+0.5*sigma**2)*T ) / ( sigma*np.sqrt(T) )
    def future(F, K, T, sigma):
        return ( np.log(F/K) + (0.5*sigma**2)*T ) / ( sigma*np.sqrt(T) )

class d2:
    def spot(S, K, T, r, sigma):
        return d1.spot(S,K,T,r,sigma) - sigma * np.sqrt(T)
    def future(F, K, T, sigma):
        return d1.future(F, K, T, sigma) - sigma * np.sqrt(T)

class call:
    def spot(S, K, T, r, sigma):
        return norm.cdf(d1.spot(S,K,T,r,sigma)) * S - norm.cdf(d2.spot(S,K,T,r,sigma)) * K * np.exp(-r*T)

    def future(F, K, T, sigma):
        return norm.cdf(d1.future(F,K,T,sigma)) * F - norm.cdf(d2.future(F,K,T,sigma)) * K

class put:
    def spot(S, K, T, r, sigma):
        return norm.cdf(-d2.spot(S,K,T,r,sigma)) * K * np.exp(-r * T) - norm.cdf(-d1.spot(S,K,T,r,sigma)) * S

    def future(F, K, T, sigma):
        return norm.cdf(-d2.future(F,K,T,sigma)) * K - norm.cdf(-d1.future(F,K,T,sigma)) * F

class implied_volatility:
    def call(F, K, T, r, call_price):
        def func(iv_guess):
            return call.future(F, K, T, iv_guess) - call_price
        try:
            iv = bisect(func, 0.00001, 5)
            return iv
        except:
            return None
    
    def put(F, K, T, r, put_price):
        def func(iv_guess):
            return put.future(F, K, T, iv_guess) - put_price
        try:
            iv = bisect(func, 0.001, 5)
            return iv
        except:
            return None
        

import pandas as pd
import matplotlib.pyplot as plt

'''Call'''
df = pd.read_csv('TXO_call.csv')

T_col = df['T']
K_col = df['K']
C_col = df['C']
F_col = df['F']
r = 1.61 / 100  # 臺灣 10 年期公債殖利率

# 計算每一個履約價 K 的隱含波動率
implied_vols = []
for i in range(len(df)):
    T = T_col[i]
    K = K_col[i]
    C = C_col[i]
    F = F_col[i]
    
    # 計算隱含波動率
    iv = implied_volatility.call(F, K, T, r, C)
    implied_vols.append(iv)

# 將隱含波動率新增至檔案
df['Implied Volatility'] = implied_vols
df.to_csv('TXO_call_withiv_20240619.csv', index=False, encoding='utf-8')

# 繪製XY散布圖
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(df['K'], df['Implied Volatility'], color='orange')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()




'''Put'''
df = pd.read_csv('TXO_put.csv')

T_col = df['T']
K_col = df['K']
P_col = df['P']
F_col = df['F']
r = 1.61 / 100  # 臺灣 10 年期公債殖利率

# 計算每一個履約價 K 的隱含波動率
implied_vols = []
for i in range(len(df)):
    T = T_col[i]
    K = K_col[i]
    P = P_col[i]
    F = F_col[i]
    
    # 計算隱含波動率
    iv = implied_volatility.put(F, K, T, r, P)
    implied_vols.append(iv)

# 將隱含波動率新增至檔案
df['Implied Volatility'] = implied_vols
df.to_csv('TXO_put_withiv_20240619.csv', index=False, encoding='utf-8')

# 繪製XY散布圖
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(df['K'], df['Implied Volatility'], color='blue')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()


def get_FTS():
    # 此處應替換為實際獲取 F, T 和 S 的邏輯
    return {"F": 20292, "T": 0.0356, "S": 20222}


def mix_cp_function_v3():
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    # 讀取 call 和 put 的隱含波動度資料
    call_iv = pd.read_csv('TXO_call_withiv_20240619.csv', index_col='K')
    put_iv = pd.read_csv('TXO_put_withiv_20240619.csv', index_col='K')

    date_str = '2024-05-02'

    # 將字串轉換為日期時間格式並限制其格式
    observe_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    
    call_iv = call_iv[['Implied Volatility']].rename(columns={'Implied Volatility': 'C'})
    put_iv = put_iv[['Implied Volatility']].rename(columns={'Implied Volatility': 'P'})

    call_iv['observe_date'] = observe_date
    put_iv['observe_date'] = observe_date

    call_iv.reset_index(inplace=True)
    put_iv.reset_index(inplace=True)

    call_iv.set_index(['K', 'observe_date'], inplace=True)
    put_iv.set_index(['K', 'observe_date'], inplace=True)

    # 合併 call 和 put 的隱含波動度
    mix = pd.concat([call_iv, put_iv], axis=1)
    mix = mix.replace(0, np.nan)

    # atm (at the money) 隱含波動度的平均
    atm = mix.loc[(mix.index.get_level_values('K') <= F*1.1) & (mix.index.get_level_values('K') >= F*0.9)]
    atm["mixIV"] = atm[["C", "P"]].mean(axis=1)

    # otm (out of the money) 買權和賣權的隱含波動度
    otm = pd.DataFrame(pd.concat([mix.loc[mix.index.get_level_values('K') < F*0.9, 'P'], mix.loc[mix.index.get_level_values('K') > F*1.1, 'C']], axis=0), columns=["mixIV"])

    # 重設索引以確保唯一性
    atm.reset_index(inplace=True)
    otm.reset_index(inplace=True)

    # 合併 atm 和 otm
    mix_cp = pd.concat([atm, otm], axis=0).sort_values(by='K').reset_index(drop=True)
    mix_cp[["C", "P"]] = mix.reset_index(drop=True)[["C", "P"]]
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[mix_cp['K'] <= F*2.5]

    return mix_cp






'''
def mix_cp_function_v2():
    # 假設 get_FTS 是一個能夠獲取 F, T 和 S 的函數
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    # 讀取 call 和 put 的隱含波動度資料
    call_iv = pd.read_csv('TXO_call_withiv_20240619.csv', index_col='K')
    put_iv = pd.read_csv('TXO_put_withiv_20240619.csv', index_col='K')

    date_str = '2024-05-02'

    # 將字串轉換為日期時間格式並限制其格式
    observe_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
    
    call_iv = call_iv[['Implied Volatility']].rename(columns={'Implied Volatility': 'C'})
    put_iv = put_iv[['Implied Volatility']].rename(columns={'Implied Volatility': 'P'})

    call_iv['observe_date'] = observe_date
    put_iv['observe_date'] = observe_date



    
    
    call_iv.index = pd.to_datetime(call_iv.index)
    put_iv.index = pd.to_datetime(put_iv.index)

    call_iv.index.name = 'observe_date'
    put_iv.index.name = 'observe_date'
    


    
    date_str = "2024-05-02"
    observe_date = pd.to_datetime(date_str)
    call_iv.reset_index(inplace=True)
    put_iv.reset_index(inplace=True)
    call_iv['observe_date'] = observe_date
    put_iv['observe_date'] = observe_date
    call_iv.set_index('observe_date', inplace=True)
    put_iv.set_index('observe_date', inplace=True)
    call_iv.drop(columns=['K'], inplace=True)
    put_iv.drop(columns=['K'], inplace=True)
    

    # 合併 call 和 put 的隱含波動度
    mix = pd.concat([call_iv, put_iv], axis=1)
    mix = mix.replace(0, np.nan)

    # atm (at the money) 隱含波動度的平均
    atm = mix.loc[(mix.index <= F*1.1) & (mix.index >= F*0.9)]
    atm["mixIV"] = atm[["C", "P"]].mean(axis=1)

    # otm (out of the money) 買權和賣權的隱含波動度
    otm = pd.DataFrame(pd.concat([mix.loc[mix.index < F*0.9, 'P'], mix.loc[mix.index > F*1.1, 'C']], axis=0), columns=["mixIV"])

    # 合併 atm 和 otm
    mix_cp = pd.concat([atm, otm], axis=0).sort_index()
    mix_cp[["C", "P"]] = mix
    mix_cp = mix_cp.dropna(subset=["mixIV"])
    mix_cp = mix_cp.loc[:F*2.5]

    return mix_cp
'''

def UnivariateSpline_function_v2(mix_cp, power=3, s=None, w=None):
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    # 建立三次樣條插值模型
    spline = UnivariateSpline(mix_cp.index, mix_cp["mixIV"], k=power, s=s, w=w)

    min_K = 0
    max_K = int(max(mix_cp.index) * 1.2)
    dK = 1
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth_IV = pd.DataFrame({'K': K_fine, 'mixIV': Vol_fine})

    # 計算平滑後的買權價格
    smooth_IV["C"] = call.future(F, smooth_IV["K"], T, smooth_IV["mixIV"])
    
    return smooth_IV







def draw_IV_and_Call_v2(smooth_IV):
    global observe_date, expiration_date, call_iv, put_iv, call_price, put_price
    basicinfo = get_FTS()
    F = basicinfo["F"]

    # Fig1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.set_title(f"{observe_date} date\n{expiration_date} expiration\nsmooth IV(Call, Put)\nF={int(F)}")

    # (1) call iv scatter   
    call_iv_filt = call_iv.loc[observe_date][call_iv.loc[observe_date] != 0]
    ax1.scatter(call_iv_filt.index, call_iv_filt, label="call iv", marker="o", color="mediumseagreen", s=10)
    
    # (2) put iv scatter     
    put_iv_filt = put_iv.loc[observe_date][put_iv.loc[observe_date] != 0]
    ax1.scatter(put_iv_filt.index, put_iv_filt, label="put iv", marker="o", color="lightcoral", s=10)
    
    # (3) F
    ax1.plot([F]*2, [0.3, 1],  ":", color="black", label=f"futures price")
    
    # (4) smooth iv    
    ax1.plot(smooth_IV["K"], smooth_IV["mixIV"], color="royalblue", label=f"smooth iv")
    
    ax1.set_xlim(0, F*2)
    ax1.set_ylim(min(min(call_iv_filt), min(put_iv_filt))-0.1, max(max(call_iv_filt), max(put_iv_filt))+0.1)

    # Fig2
    ax2.set_title("smooth call price")  
    
    # (1) call price scatter   
    call_price_filt = call_price.loc[observe_date][call_price.loc[observe_date] != 0]
    ax2.scatter(call_price_filt.index, call_price_filt, label="call price", marker="o", color="mediumseagreen", s=10)
    
    # (3) smooth price    
    ax2.plot(smooth_IV["K"], smooth_IV["C"], alpha=0.8, label="smooth call price", color="royalblue")
    
    ax2.set_xlim(0, F*2)
    ax2.set_ylim(0, max(call_price_filt)*1.1)

    ax1.grid(linestyle='--', alpha=0.3)
    ax2.grid(linestyle='--', alpha=0.3)
    ax1.legend()
    ax2.legend()
    plt.show()



# RND main 
observe_date = "2024-05-02"
expiration_date = "2024-05-22"

mix_cp = mix_cp_function_v3()

plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(mix_cp['K'], mix_cp['mixIV'], color='orange')
plt.xlabel('Strike Price (K)')
plt.ylabel('mixIV')
plt.title('Mix Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()





smooth_IV = UnivariateSpline_function_v2(mix_cp, power=4)

draw_IV_and_Call_v2(smooth_IV)