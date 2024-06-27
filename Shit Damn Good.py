import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import minimize
from scipy.stats import genextreme
from scipy.stats import norm

import matplotlib.pyplot as plt

# 讀取數據
df_call = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_call_20240502.csv')
df_put = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_put_20240502.csv')
df_call_with_iv = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_call_with iv_20240502.csv')
df_put_with_iv = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_put_with iv_20240502.csv')

# 計算混合隱含波動率
def mix_cp_function_v3(df_call, df_put):
    F = 20292
    observe_date = '2024-05-02'

    call_iv = df_call[['K', 'Implied Volatility']].rename(columns={'Implied Volatility': 'C'})
    put_iv = df_put[['K', 'Implied Volatility']].rename(columns={'Implied Volatility': 'P'})

    call_iv['observe_date'] = observe_date
    put_iv['observe_date'] = observe_date

    call_iv.set_index(['K', 'observe_date'], inplace=True)
    put_iv.set_index(['K', 'observe_date'], inplace=True)

    mix = pd.concat([call_iv, put_iv], axis=1).replace(0, np.nan)

    atm = mix.loc[(mix.index.get_level_values('K') <= F*1.1) & (mix.index.get_level_values('K') >= F*0.9)]
    atm['mixIV'] = atm[['C', 'P']].mean(axis=1)

    otm = pd.DataFrame(pd.concat([mix.loc[mix.index.get_level_values('K') < F*0.9, 'P'], mix.loc[mix.index.get_level_values('K') > F*1.1, 'C']], axis=0), columns=['mixIV'])

    atm.reset_index(inplace=True)
    otm.reset_index(inplace=True)

    mix_cp = pd.concat([atm, otm], axis=0).sort_values(by='K').reset_index(drop=True)
    mix_cp[['C', 'P']] = mix.reset_index(drop=True)[['C', 'P']]
    mix_cp = mix_cp.dropna(subset=['mixIV'])
    mix_cp = mix_cp.loc[mix_cp['K'] <= F*2.5]

    return mix_cp

# 計算混合隱含波動率
mix_cp = mix_cp_function_v3(df_call_with_iv, df_put_with_iv)

plt.figure(figsize=(10, 6), dpi=200)
plt.scatter(mix_cp['K'], mix_cp['mixIV'], color='orange')
plt.xlabel('Strike Price (K)')
plt.ylabel('mixIV')
plt.title('Mix Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()




# 4 次樣條插值擬合
x = mix_cp['K']
y = mix_cp['mixIV']
knots = [20292]
spline = LSQUnivariateSpline(x, y, knots, k=4)

x_fit = np.linspace(15800, 23000, 125)
y_fit = spline(x_fit)


# 繪製擬合圖
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x, y, 'o', label='original')
plt.plot(x_fit, y_fit, '-', label='fit_curve')
plt.xlabel('K')
plt.ylabel('mix_IV')
plt.legend()
plt.title('4th Order Spline with One Knot')
plt.grid(True)
plt.show()




# 計算買權價格與一二階導數
def call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

call_prices_fit = [call_price(20292, K, 0.0356, 0.0161, vol) for K, vol in zip(x_fit, y_fit)]
call_price_first_derivative = np.gradient(call_prices_fit, x_fit)
call_price_second_derivative = np.gradient(call_price_first_derivative, x_fit)


# 繪製買權價格與履約價的圖
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_fit, call_prices_fit, 'o', label='original')
plt.xlabel('K')
plt.ylabel('Call Price from IV')
plt.legend()
plt.title('Call Price from IV vs Strike Price')
plt.grid(True)
plt.show()





# 計算 RND
RND = np.exp(0.0161 * 0.0356) * call_price_second_derivative

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_fit, RND, '-', label='Risk Neutral Density')
plt.axvline(x=20292, color='k', linestyle='--', label='Strike Price 20292')
plt.xlabel('K')
plt.ylabel('RND')
plt.legend()
plt.title('Risk Neutral Density')
plt.grid(True)






# 創建包含 RND 的 DataFrame
RND2 = pd.DataFrame({'K': x_fit, 'RND': RND})
RND2['RND_quantile'] = RND2['RND'].cumsum() / RND2['RND'].sum()
RND2['cdf'] = RND2['RND'].cumsum()

# GEV 分布擬合函數
def FitGEVRightTail(beta, *args):
    Mean, sigma, phi = beta
    a0, a1, a2, Ka0, Ka1, Ka2, fKa0, fKa1, fKa2, FKa0R = args

    pdf0 = genextreme(phi, loc=Mean, scale=sigma).pdf(Ka0)
    pdf1 = genextreme(phi, loc=Mean, scale=sigma).pdf(Ka1)
    pdf2 = genextreme(phi, loc=Mean, scale=sigma).pdf(Ka2)
    cdf0 = genextreme(phi, loc=Mean, scale=sigma).cdf(Ka0)

    y = (pdf0 - fKa0)**2 + (pdf1 - fKa1)**2 + (pdf2 - fKa2)**2

    return y

def FitGEVLeftTail(beta, *args):
    Mean, sigma, phi = beta
    a0, a1, a2, Ka0, Ka1, Ka2, fKa0, fKa1, fKa2, FKa0L = args

    pdf0 = genextreme(phi, loc=Mean, scale=sigma).pdf(-Ka0)
    pdf1 = genextreme(phi, loc=Mean, scale=sigma).pdf(-Ka1)
    pdf2 = genextreme(phi, loc=Mean, scale=sigma).pdf(-Ka2)
    pdf3 = genextreme(phi, loc=Mean, scale=sigma).pdf(0)
    cdf0 = genextreme(phi, loc=Mean, scale=sigma).cdf(-Ka0)

    y = (pdf0 - fKa0)**2 + (pdf1 - fKa1)**2 + pdf3**2

    if pdf3 <= 0:
        y = 1e100

    return y

def GEV_Right_function(RND2):
    F = 20292
    right_edge = int(F * 3)
    a0R = 0.95
    a1R = 0.98
    a2R = 0.99

    Ka0R = RND2.query(f'RND_quantile>{a0R}').iloc[0]['K']
    Ka1R = RND2.query(f'RND_quantile>{a1R}').iloc[0]['K']
    Ka2R = RND2.query(f'RND_quantile>{a2R}').iloc[0]['K']
    fKa0R = np.interp(Ka0R, RND2['K'], RND2['RND'])
    fKa1R = np.interp(Ka1R, RND2['K'], RND2['RND'])
    fKa2R = np.interp(Ka2R, RND2['K'], RND2['RND'])
    FKa0R = np.interp(Ka0R, RND2['K'], RND2['cdf'])

    start1 = [F, 4000, 0]
    options = {'maxfun': 1e10, 'maxiter': 1e10}
    result1 = minimize(FitGEVRightTail, start1, args=(a0R, a1R, a2R, Ka0R, Ka1R, Ka2R, fKa0R, fKa1R, fKa2R, FKa0R), method='Nelder-Mead', options=options, tol=1e-8)
    
    Mean, sigma, phi = result1.x

    K3 = np.arange(int(Ka0R * 0.8), right_edge, 1)
    GEV_right = [genextreme(phi, loc=Mean, scale=sigma).pdf(k) for k in K3]
    return K3, GEV_right, result1.x, Ka0R

# 找左尾 GEV 參數
def GEV_Left_function(RND2):
    F = 20292
    a0L = 0.01
    a1L = 0.03
    a2L = 0.05

    Ka0L = RND2.query(f'RND_quantile<{a0L}').iloc[-1]['K']
    Ka1L = RND2.query(f'RND_quantile<{a1L}').iloc[-1]['K']
    Ka2L = RND2.query(f'RND_quantile<{a2L}').iloc[-1]['K']
    fKa0L = np.interp(Ka0L, RND2['K'], RND2['RND'])
    fKa1L = np.interp(Ka1L, RND2['K'], RND2['RND'])
    fKa2L = np.interp(Ka2L, RND2['K'], RND2['RND'])
    FKa0L = np.interp(Ka0L, RND2['K'], RND2['cdf'])

    start1 = [-F, 20000, -0.1]
    options = {'maxfun': 1e2, 'maxiter': 1e2}
    result1 = minimize(FitGEVLeftTail, start1, args=(a0L, a1L, a2L, Ka0L, Ka1L, Ka2L, fKa0L, fKa1L, fKa2L, FKa0L), method='Nelder-Mead', options=options, tol=1e-8)
    Mean, sigma, phi = result1.x

    K4 = np.arange(0, int(Ka0L * 2), 1)
    GEV_left = [genextreme(phi, loc=Mean, scale=sigma).pdf(-k) for k in K4]
    return K4, GEV_left, result1.x, Ka0L

# 合併左右尾的 RND
def mix_RND_GEV(RND2, K3, GEV_right, K4, GEV_left, Ka0R, Ka0L):
    df_Body = RND2[['K', 'RND']]

    F = 20292
    right_edge = int(F * 3)

    # Right
    df_Right = pd.DataFrame([K3, GEV_right], index=['K', 'Right']).T
    df_main = df_Body.merge(df_Right, on='K', how='outer')

    # Left
    if isinstance(K4, np.ndarray):
        df_Left = pd.DataFrame([K4, GEV_left], index=['K', 'Left']).T
        df_main = df_main.merge(df_Left, on='K', how='outer')
        df_main['meanLeft'] = df_main.query(f"K<{Ka0L}")[['Left']].mean(axis=1, skipna=True)
    else:
        df_main['meanLeft'] = 0

    df_main = df_main.sort_values('K').reset_index(drop=True)
    df_main.replace([np.inf, -np.inf], np.nan, inplace=True)

    if RND2['K'].max() > right_edge:
        df_main['meanRight'] = df_main.query(f"K>{Ka0R}")[['RND']].mean(axis=1, skipna=True)
    else:
        df_main['meanRight'] = df_main.query(f"K>{Ka0R}")[['Right']].mean(axis=1, skipna=True)

    df_main['RND'] = df_main['meanRight']
    df_main['RND'] = df_main['RND'].fillna(df_main['meanLeft'])
    df_main['RND'] = df_main['RND'].fillna(df_main['RND'])

    df_main['RND'] = np.where(df_main['RND'] < 0, 0, df_main['RND'])
    df_main['RND'] = df_main['RND'].cumsum() / df_main['RND'].sum()

    return df_main

# 右尾
K3, GEV_right, param_right, Ka0R = GEV_Right_function(RND2)

# 左尾
K4, GEV_left, param_left, Ka0L = GEV_Left_function(RND2)

# 合併 RND 和 GEV
df_main = mix_RND_GEV(RND2, K3, GEV_right, K4, GEV_left, Ka0R, Ka0L)

# 繪製 RND 與 GEV 的分布圖
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=200)

ax.set_title(f"GEV \n left: {', '.join([f'{x:.2f}' for x in param_left])} \n right: {', '.join([f'{x:.2f}' for x in param_right])}")
ax.plot(RND2['K'], RND2['RND'], label='Original RND', color='royalblue')
ax.plot(K4, GEV_left, ':', label='GEV for left tail', color='darkorange')
ax.plot(K3, GEV_right, ':', label='GEV for right tail', color='mediumseagreen')
ax.legend()

plt.xlabel('Strike Price (K)')
plt.ylabel('Risk Neutral Density')
plt.xlim(10000, 30000)
plt.show()


# 左邊直接拉到 0，右邊拉到 30,000，要固定。
# Next, 不要有起始值，用論文裡的 Three Conditions，去試第一個條件。
# 研究一下一個點的做法，(1) cdf, (2) pdf, (3) 斜率相等。