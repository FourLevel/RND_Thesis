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
import pandas as pd
import matplotlib.pyplot as plt


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
        


'''Call'''
df_call = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_call_20240502.csv')

T_col = df_call['T']
K_col = df_call['K']
C_col = df_call['C']
F_col = df_call['F']
r = 1.61 / 100  # 臺灣 10 年期公債殖利率

# 計算每一個履約價 K 的隱含波動率
implied_vols = []
for i in range(len(df_call)):
    T = T_col[i]
    K = K_col[i]
    C = C_col[i]
    F = F_col[i]
    
    # 計算隱含波動率
    iv = implied_volatility.call(F, K, T, r, C)
    if iv != 0.00001:
        iv = iv
    else:
        iv = None
    
    implied_vols.append(iv)

# 將隱含波動率新增至檔案
df_call['Implied Volatility'] = implied_vols
df_call.to_csv('RND_data/asher_data/For Implied Volatility/TXO_call_with iv_20240502.csv', index=False, encoding='utf-8')

# 繪製XY散布圖
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(df_call['K'], df_call['Implied Volatility'], color='orange')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()




'''Put'''
df_put = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_put_20240502.csv')

T_col = df_put['T']
K_col = df_put['K']
P_col = df_put['P']
F_col = df_put['F']
r = 1.61 / 100  # 臺灣 10 年期公債殖利率

# 計算每一個履約價 K 的隱含波動率
implied_vols = []
for i in range(len(df_put)):
    T = T_col[i]
    K = K_col[i]
    P = P_col[i]
    F = F_col[i]
    
    # 計算隱含波動率
    iv = implied_volatility.put(F, K, T, r, P)
    if iv != 0.001:
        iv = iv
    else:
        iv = None
    
    implied_vols.append(iv)

# 將隱含波動率新增至檔案
df_put['Implied Volatility'] = implied_vols
df_put.to_csv('RND_data/asher_data/For Implied Volatility/TXO_put_with iv_20240502.csv', index=False, encoding='utf-8')

# 繪製XY散布圖
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(df_put['K'], df_put['Implied Volatility'], color='blue')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()