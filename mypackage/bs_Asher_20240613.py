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
# df.to_csv('F:\\GitHub\\Risk-Neutral-Density\\mypackage\\TXO_call_withiv.csv', index=False, encoding='utf-8')

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
# df.to_csv('F:\\GitHub\\Risk-Neutral-Density\\mypackage\\TXO_put_withiv.csv', index=False, encoding='utf-8')

# 繪製XY散布圖
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(df['K'], df['Implied Volatility'], color='blue')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()