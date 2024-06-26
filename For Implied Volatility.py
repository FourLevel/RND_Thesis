import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import bisect


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


F = 20292
T = 0.0356
S = 20222

def mix_cp_function_v3():
    global F, T, S

    # 讀取 call 和 put 的隱含波動度資料
    call_iv = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_call_with iv_20240502.csv', index_col='K')
    put_iv = pd.read_csv('RND_data/asher_data/For Implied Volatility/TXO_put_with iv_20240502.csv', index_col='K')

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


mix_cp = mix_cp_function_v3()
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(mix_cp['K'], mix_cp['mixIV'], color='orange')
plt.xlabel('Strike Price (K)')
plt.ylabel('mixIV')
plt.title('Mix Implied Volatility vs Strike Price')
plt.grid(True)
plt.show()





# 提取履约价和隐含波动率
x = mix_cp['K']
y = mix_cp['mixIV']

# 定义样条节点
knots = [20292]  # 你可以根据数据调整节点


# 创建LSQUnivariateSpline样条
spline = LSQUnivariateSpline(x, y, knots, k=4)

# 创建拟合数据
x_fit = np.linspace(15800, 23000, 125)
y_fit = spline(x_fit)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x, y, "o", label="original")
plt.plot(x_fit, y_fit, "-", label='fit_curve')
plt.xlabel('K')
plt.ylabel('mix_IV')
plt.legend()
plt.title("4th order spline with one knot ")
plt.grid(True)
plt.show()



# 计算期权价格
call_prices = [call.spot(F, K, T, r, vol) for K, vol in zip(x_fit, y_fit)]

# 计算期权价格的二阶导数
call_price_first_derivative = np.gradient(call_prices, x_fit)
call_price_second_derivative = np.gradient(call_price_first_derivative, x_fit)

# 计算风险中性概率密度
RND = np.exp(r * T) * call_price_second_derivative



# 创建包含期权价格和RND的DataFrame
fit_20240522 = pd.DataFrame({
    '履約價': x_fit,
    'fit_cp_IV': y_fit,
    'Call_Price': call_prices,
    'Call_Price_Second_Derivative': call_price_second_derivative,
    'RND': RND
})

# 输出结果
print(fit_20240522.head())


# 计算累计分布函数（CDF）
CDF = np.cumsum(RND) * (x_fit[1] - x_fit[0])

# 绘制RND
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x_fit, RND, "-", label='Risk Neutral Density')
plt.axvline(x=20292, color='k', linestyle='--', label='Strike Price 20292')
plt.xlabel('K')
plt.ylabel('RND')
plt.legend()
plt.title('Risk Neutral Density')
plt.grid(True)

# 設置 X 軸間隔為 
plt.xticks(np.arange(min(x_fit), max(x_fit)+1, 500))

# 設置 Y 軸起點為 0
plt.ylim(bottom=0)

plt.show()