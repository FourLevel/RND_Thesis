# import packages
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



def read_data_v2(expiration_date):

    #formatted_date = datetime.strptime(expiration_date, "%Y-%m-%d").strftime("%d%b%y").upper()

    call_iv = pd.read_csv(f"RND_data/deribit_data/iv/call/call_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    put_iv = pd.read_csv(f"RND_data/deribit_data/iv/put/put_iv_{expiration_date}.csv", index_col="Unnamed: 0")/100
    df_idx = pd.read_csv(f"RND_data/deribit_data/BTC-index/BTC_index_{expiration_date}.csv", index_col="Unnamed: 0")

    call_iv.columns = call_iv.columns.astype(int)
    put_iv.columns = put_iv.columns.astype(int)

    call_price = pd.read_csv(f"RND_data/deribit_data/BTC-call/call_strike_{expiration_date}.csv", index_col="Unnamed: 0")
    put_price = pd.read_csv(f"RND_data/deribit_data/BTC-put/put_strike_{expiration_date}.csv", index_col="Unnamed: 0")

    call_price.columns = call_price.columns.astype(int)
    put_price.columns = put_price.columns.astype(int)
    
    #df_F = find_F_df(call_iv, put_iv, call_price, put_price, df_idx)

    return call_iv, put_iv, call_price, put_price, df_idx#, df_F



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
    # (2) put price scatter     
    #put_price_filt = put_price.loc[observe_date][put_iv.loc[observe_date] != 0]
    #ax2.scatter(put_price_filt.index, put_price_filt, label="put iv", marker="o", color="lightcoral", s=10)
    # (3) smooth price    
    ax2.plot(smooth_IV["K"], smooth_IV["C"], alpha=0.8, label="smooth call price", color="royalblue")
    ax2.set_xlim(0, F*2)
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

    return smooth_IV



# 找出正常區間的RND
def RND2_function(smooth_IV):
    global ITM, vally, der3, peak

    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]

    df = smooth_IV.copy()
    df["RND_diff"] = df["RND"] - df["RND"].shift(1)
    max_idx = df.query(f"K>{F*0.7} & K<{F*1.3}")["RND"].idxmax() # 價平附近的最大RND
    print("maxidx", max_idx)
    # 1. 左尾
    #   (1) 若有三次微分高點
    try:     
        der1 = (df["C"].diff() / df["K"].diff()).fillna(np.nan)
        der2 = der1.diff().fillna(np.nan)
        der3 = der2.diff().fillna(np.nan)
        der3 = savgol_filter( der3.fillna(method='bfill'),35000, 3)
        der3 = pd.Series(der3, index=der1.index)
        
        ITM = der3.loc[:max_idx-5000]
        vally = ITM.copy()
        conditions = []

        for i in range(1, 500):
            conditions.append((ITM.shift(-i) < ITM) & (ITM > ITM.shift(i)))

        for condition in conditions:
            vally = vally[condition]
            
        startidx1 = int(vally.index[-1] ) #*1.2#*0.9 # -5000  #* 0.7 #0.8
    except:  
        startidx1 = 0
    
    #   (2) 翹起
    try:     
        startidx2 = int(df.loc[:max_idx-5000].query(f"RND_diff<0").index[-1] )#*1.2
    except:  
        startidx2 = 0
    
    print("startidx1", startidx1, "startidx2", startidx2)
    if startidx1 > startidx2 and startidx2 !=0:
        startidx = startidx2 + (startidx1 - startidx2) * 0.5
        print("startidx", startidx)
    else:
        startidx = max(startidx1*0.8, startidx2*1.2)
    

    if (df["RND"].loc[startidx:int(F)]<=0).sum() >= 1: #有負值
        startidx = int(df.loc[:int(F)].query("RND<0").index[-1] * 1.05)


    #   右尾
    try:     # 若有翹起
        endidx = df.loc[max_idx+5000:].query(f"RND_diff>0").index[0]
    except:  # 若無翹起
        endidx = len(df)
    #RND2 =  df.loc[startidx-1:endidx-1]

    print(startidx-1,endidx-1000)
    RND2 =  df.loc[startidx-1:endidx-1000]


    # 2. 無負值
    RND2["RND"] = np.where(RND2["RND"] < 0, np.nan, RND2["RND"])
    RND2 = RND2.dropna()
    RND2["CDF"] = RND2["RND"].cumsum()
    RND2["RND_quantile"] = RND2["CDF"]/RND2["CDF"].iloc[-1]
    
    return RND2






# RND main 
observe_date = "2024-01-29"
expiration_date = "2024-03-29"
call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
F = find_F2()

mix_cp = mix_cp_function_v2()
smooth_IV = UnivariateSpline_function_v2(mix_cp, power=4)
draw_IV_and_Call_v2(smooth_IV)

smooth_IV = RND_function(smooth_IV)
RND2 = RND2_function(smooth_IV)    

plt.plot(smooth_IV["pdf"],':')
plt.plot(RND2["RND"],':')
plt.plot(RND2["pdf"],':')

left_cdf = RND2["cdf"].iloc[0] - RND2["pdf"].iloc[0] 
right_cdf = (1-RND2["cdf"].iloc[-1]) - RND2["pdf"].iloc[-1] 
print(left_cdf, right_cdf)

plt.plot(smooth_IV["cdf"])
plt.plot(RND2["cdf"])



def daily_rnd():
    global observe_date, expiration_date, call_iv, put_iv, call_price, put_price, df_idx

    # 讀取資料
    basicinfo = get_FTS()
    F = basicinfo["F"]
    T = basicinfo["T"]
    S = basicinfo["S"]
    
    # 處理市場數據
    mix_cp = mix_cp_function_v2()

    # 平滑 IV
    smooth_IV = UnivariateSpline_function_v2(mix_cp, power=4)

    # 計算RND
    smooth_IV = RND_function(smooth_IV)
    RND2 = RND2_function(smooth_IV)      
    
    # 右尾
    K3, GEV_right, param_right = GEV_Right_function(RND2)

    # 左尾
    K4, GEV_left, param_left = GEV_Left_function(RND2)

    # 合併RND GEV
    df_main = mix_RND_GEV(RND2, K3, GEV_right, K4, GEV_left)
    
    # 測試!!!
    #df_main = df_main.query("CDF<=1")  # 測試: 只取到CDF=1 !!!
    #df_main["RND"] = df_main["RND"] + (1-df_main["CDF"].iloc[-1])/len(df_main)
    #df_main["CDF"] = df_main["RND"].cumsum()


    # 綜合圖表
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))

    rnd2_startK = RND2["K"].iloc[0]
    rnd2_endK = RND2["K"].iloc[-1]
    rnd2_highestIV = RND2["mixIV"].max()
    rnd2_lowestIV = RND2["mixIV"].min()

    call_iv_filt = call_iv.loc[observe_date][call_iv.loc[observe_date] != 0]
    put_iv_filt = put_iv.loc[observe_date][put_iv.loc[observe_date] != 0]

    # Fig1

    ax[0, 0].set_title(f"{observe_date} date\n{expiration_date} expiration\nsmooth IV(Call, Put)\nF={int(F)}")
    ax[0, 0].scatter(call_iv_filt.index, call_iv_filt, label="call iv", marker="o", color="mediumseagreen", s=10)  # call iv點
    ax[0, 0].scatter(put_iv_filt.index, put_iv_filt, label="put iv", marker="o", color="lightcoral", s=10)         # put iv點
    ax[0, 0].plot([F]*2, [rnd2_lowestIV-0.1, rnd2_highestIV+0.1],  ":", color="black", label=f"futures price")            # futures price
    ax[0, 0].plot(smooth_IV["K"], smooth_IV["mixIV"], color="royalblue", label=f"smooth iv")                       # smooth iv線 
    ax[0, 0].scatter(rnd2_startK, RND2["mixIV"].iloc[0], color="black")                                          # 正常 RND 範圍的左點
    ax[0, 0].scatter(rnd2_endK, RND2["mixIV"].iloc[-1], color="black")                                           # 正常 RND 範圍的右點
    ax[0, 0].set_xlim(min(rnd2_startK, put_iv_filt.index[0])-1000, max(rnd2_endK, call_iv_filt.index[0])+1000)   # x軸範圍
    ax[0, 0].set_ylim(rnd2_lowestIV-0.1, rnd2_highestIV+0.1)                                                         # y軸範圍
    
    # Fig2
    call_price_filt = call_price.loc[observe_date][call_price.loc[observe_date] != 0]  
    ax[0, 1].set_title("smooth Call price")                                      
    ax[0, 1].scatter(call_price_filt.index, call_price_filt, label="call price", marker="o", color="mediumseagreen", s=10)   # call iv點
    ax[0, 1].plot(smooth_IV["K"], smooth_IV["C"], alpha=0.8, label="smooth call price", color="royalblue")          # smooth price線  
    ax[0, 1].scatter(rnd2_startK, RND2["C"].iloc[0], color="black")                                              # 正常 RND 範圍的左點
    ax[0, 1].scatter(rnd2_endK, RND2["C"].iloc[-1],  color="black")                                               # 正常 RND 範圍的右點       
    ax[0, 1].set_xlim(min(rnd2_startK, put_iv_filt.index[0])-1000, max(rnd2_endK, call_iv_filt.index[0])+1000)  # x軸範圍
    ax[0, 1].set_ylim(0, RND2["C"].max()*1.2)                                                        # y軸範圍
    
    # Fig3
    ax[0, 2].set_title("DVF RND")
    ax[0, 2].plot(smooth_IV.query(f"K>{rnd2_startK-1000} & K<{rnd2_endK+1000}")["K"], 
                  smooth_IV.query(f"K>{rnd2_startK-1000} & K<{rnd2_endK+1000}")["RND"], label='DVF RND', color="royalblue")  # RND1 線
    ax[0, 2].set_xlabel('Stock price at maturity, S(T)')
    ax[0, 2].set_ylabel('Density', color='royalblue')
    ax[0, 2].scatter(RND2["K"].iloc[0], RND2["RND"].iloc[0], color="black")
    ax[0, 2].scatter(RND2["K"].iloc[-1], RND2["RND"].iloc[-1], color="black")

    cdf = RND2["RND"].cumsum()
    ax_twin = ax[0, 2].twinx()
    ax_twin.plot(RND2["K"], cdf , color="darkorange", label='CDF')
    ax_twin.set_ylabel('CDF', color='darkorange')
    ax_twin.tick_params('y', colors='darkorange')
    cdf_value = round(cdf.iloc[-1],4)
    ax_twin.annotate(f'CDF: {cdf_value}', xy=(RND2["K"].iloc[-1], cdf_value),
                xytext=(RND2["K"].iloc[-1]*0.8, cdf_value - 0.1),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10)
    ax_twin.legend(loc='center right', bbox_to_anchor=(1, 0.4))


    # Fig4
    ax[1, 0].set_title(f"Find the GEV for the Right tail \n {', '.join([f'{x:.2f}' for x in param_right])}")
    ax[1, 0].plot(RND2["K"], RND2["RND"], label='DVF RND', color="royalblue")
    ax[1, 0].plot(K3, GEV_right, ':', label='GEV for right tail', color="darkorange")
    ax[1, 0].plot(Ka0R, fKa0R, 'o', markerfacecolor='none', label='a0R anchor', color="mediumseagreen")
    ax[1, 0].plot(Ka1R, fKa1R, 'o', markerfacecolor='none', label='a1R anchor', color="lightcoral")
    #ax[1, 0].plot(Ka2R, fKa2R, 'o', markerfacecolor='none', label='a1R anchor', color="gray")

    ax[1, 0].set_xlabel('Stock price at maturity, S(T)')
    ax[1, 0].set_ylabel('Risk Neutral Density')

    # Fig5
    ax[1, 1].set_title(f"Find the GEV for the Left tail \n {', '.join([f'{x:.2f}' for x in param_left])}")
    ax[1, 1].plot(RND2["K"], RND2["RND"], label='DVF RND', color="royalblue")
    ax[1, 1].plot(K4, GEV_left, ':', label='GEV for left tail', color="darkorange")
    ax[1, 1].plot(Ka0L, fKa0L, 'o', markerfacecolor='none', label= 'a0 anchor', color="mediumseagreen")
    ax[1, 1].plot(Ka1L, fKa1L, 'o', markerfacecolor='none', label='a1 anchor', color="lightcoral")
    #ax[1, 1].plot(Ka2L, fKa2L, 'o', markerfacecolor='none', label='a2 anchor', color="gray")

    ax[1, 1].set_xlabel('Stock price at maturity, S(T)')
    ax[1, 1].set_ylabel('Risk Neutral Density')


    # Fig6
    ax[1, 2].set_title("Final RND(DVF+GEV)")
    ax[1, 2].plot(df_main["K"], df_main["RND"], color="royalblue", label='RND')
    ax[1, 2].set_xlabel('Stock price at maturity, S(T)')
    ax[1, 2].set_ylabel('Density', color='royalblue')
    ax[1, 2].tick_params('y', colors='royalblue')
    try:
        ax[1, 2].plot(Ka1L, df_main.query(f"K=={Ka1L}")["RND"], 'o', markerfacecolor='none')
        ax[1, 2].plot(Ka0L, df_main.query(f"K=={Ka0L}")["RND"], 'o', markerfacecolor='none')
    except:
        ax[1, 2].plot(Ka1L, RND2.query(f"K=={Ka1L}")["RND"], 'o', markerfacecolor='none')
        ax[1, 2].plot(Ka0L, RND2.query(f"K=={Ka0L}")["RND"], 'o', markerfacecolor='none')
        pass
    try:
        ax[1, 2].plot(Ka1R, df_main.query(f"K=={Ka1R}")["RND"], 'o', markerfacecolor='none')
        ax[1, 2].plot(Ka0R, df_main.query(f"K=={Ka0R}")["RND"], 'o', markerfacecolor='none')
    except:
        ax[1, 2].plot(Ka1R, RND2.query(f"K=={Ka1R}")["RND"], 'o', markerfacecolor='none')
        ax[1, 2].plot(Ka0R, RND2.query(f"K=={Ka0R}")["RND"], 'o', markerfacecolor='none')

    #    futures price
    ax[1, 2].plot([F]*2, [max(df_main["RND"]),min(df_main["RND"])], ":", color="black", label="Futures") 

    ax_twin = ax[1, 2].twinx()
    ax_twin.plot(df_main["K"], df_main["CDF"] , color="darkorange", label='CDF')
    ax_twin.set_ylabel('CDF', color='darkorange')
    ax_twin.tick_params('y', colors='darkorange')
    cdf_value = round(df_main["CDF"].iloc[-1], 4)
    ax_twin.annotate(f'CDF: {cdf_value}', xy=(df_main["K"].iloc[-1], cdf_value),
                xytext=(df_main["K"].iloc[-1]*0.8, cdf_value - 0.1),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10)
    ax_twin.legend(loc='center right', bbox_to_anchor=(1, 0.3))

    # step3: calculate skewness, kurtosis
    weighted_mean, weighted_std, weighted_skew, excess_kurt = get_moment(df_main)


    ax_twin.text(df_main["K"].iloc[-1]*0.8, 0.7, f"skew: {round(weighted_skew,2)}")
    ax_twin.text(df_main["K"].iloc[-1]*0.8, 0.6, f"kurt: {round(excess_kurt,2)}")

    for i in range(2):
        for j in range(3):
            ax[i, j].grid(linestyle='--', alpha=0.4)
            ax[i, j].legend()
    ax[0, 2].legend(loc='center right')
    ax[1, 2].legend(loc='center right', bbox_to_anchor=(1, 0.4))

    plt.tight_layout()
    plt.show()


    atm_iv = smooth_IV.query(f"K > { F } ")["mixIV"].iloc[0]
    #lowest_iv = smooth_IV.query(f"K>{rnd2_startK} & K<{rnd2_endK}")["mixIV"].min()
    lowest_iv = smooth_IV.query(f"K>{F*0.8} & K<{F*1.2}")["mixIV"].min()
    #d25_iv_diff = smooth_IV.query(f"K >= { F*1.15 } ")["mixIV"].iloc[0] - smooth_IV.query(f"K >= { F*0.85 } ")["mixIV"].iloc[0]
 
    delta = df_main.dropna(subset=["mixIV"])

    delta["Cdelta"] = norm.cdf( (np.log(F/delta["K"])+(delta["mixIV"]**2/2)*T) / (delta["mixIV"] * np.sqrt(T)) )
    delta["Pdelta"] = norm.cdf( (np.log(F/delta["K"])+(delta["mixIV"]**2/2)*T) / (delta["mixIV"] * np.sqrt(T)) )-1

    d25_calliv = delta.query("Cdelta>=0.25").iloc[0]["mixIV"]
    d25_putiv = delta.query("Pdelta<=-0.25").iloc[0]["mixIV"]
    d25_iv_diff = d25_calliv-d25_putiv

    q5_moneyness = df_main.query("CDF >= 0.05")["K"].iloc[0] / F
    q95_moneyness = df_main.query("CDF <= 0.95")["K"].iloc[-1] / F

    result = {"df":df_main,
              "skew":round(weighted_skew,2),
              "kurt":round(excess_kurt,2),
              "param_right":param_right,
              "param_left":param_left,
              "atm_iv":atm_iv,
              "lowest_iv":lowest_iv,
              "d25_iv_diff":d25_iv_diff,
              "q5_moneyness":q5_moneyness,
              "q95_moneyness":q95_moneyness,
              }

    return result


observe_date = "2024-01-29"
expiration_date = "2024-03-29"

call_iv, put_iv, call_price, put_price, df_idx = read_data_v2(expiration_date)
result = daily_rnd()

#RND_dict_1[expiration_date][observe_date] =  result["df"]
#skew_dict_1[expiration_date][observe_date] =  result["skew"]
#kurt_dict_1[expiration_date][observe_date] =  result["kurt"]
#right_dict_1[expiration_date][observe_date] =  result["param_right"]
#left_dict_1[expiration_date][observe_date] =  result["param_left"]
#df_ivinfo.loc[observe_date] = [result["atm_iv"], result["lowest_iv"], result["d25_iv_diff"], expiration_date]

result["df"]