# 基本數據處理與分析套件
import pandas as pd
import numpy as np
import statsmodels.api as sm
# 繪圖套件
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
# 日期時間處理
from datetime import datetime, timedelta
import calendar
# 數學與統計相關套件
from scipy.optimize import bisect, minimize
from scipy.stats import norm, genextreme, genpareto as gpd
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, InterpolatedUnivariateSpline, CubicSpline, interp1d
from scipy.integrate import quad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
# 系統與工具套件
import os
import re
import asyncio
import nest_asyncio
import warnings
import itertools
# 自定義套件
from mypackage.bs import *
from mypackage.marketIV import *
from mypackage.moment import *

nest_asyncio.apply()
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)
today = datetime.now().strftime('%Y-%m-%d')

# 判斷顯著性
def get_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''


''' 執行迴歸分析_每天_一個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_一個點_2025-04-28_刪1個極端值.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_day_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv(f'descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std


# 測試固定 Skewness 的迴歸分析
# 定義所有要測試的變數
variables = ['Mean', 'Std', 'Kurtosis', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
skewness_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Skewness_係數',
    'Skewness_p值',
    'Skewness_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Skewness 單獨作為變數
X = df_regression_day_stats_with_returns[['Skewness']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_skew = model.pvalues[1]
sig_skew = get_significance(p_value_skew)

if p_value_skew < min_p_value:
    min_p_value = p_value_skew
    best_combination = "僅 Skewness"

skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
    '變數組合': ["僅 Skewness"],
    'Skewness_係數': [model.params[1]],
    'Skewness_p值': [p_value_skew],
    'Skewness_顯著性': [sig_skew],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Skewness 添加到組合中
        all_vars = ['Skewness'] + list(combo)
        X = df_regression_day_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        skewness_idx = 1  # Skewness 在加入常數項後的索引
        p_value_skew = model.pvalues[skewness_idx]
        sig_skew = get_significance(p_value_skew)
        
        # 組合名稱
        combo_name = "Skewness + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_skew < min_p_value:
            min_p_value = p_value_skew
            best_combination = combo_name
        
        # 儲存結果
        skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Skewness_係數': [model.params[skewness_idx]],
            'Skewness_p值': [p_value_skew],
            'Skewness_顯著性': [sig_skew],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Skewness 的 p 值排序結果
skewness_regression_results = skewness_regression_results.sort_values(by='Skewness_p值')

# 顯示結果
print("\nSkewness 各變數組合的迴歸分析結果：")
print(skewness_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nSkewness 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
skewness_regression_results.to_csv('skewness_regression_results_一個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 skewness_regression_results_一個點.csv")


# 測試固定 Kurtosis 的迴歸分析
# 定義所有要測試的變數（排除 Kurtosis 本身）
variables = ['Mean', 'Std', 'Skewness', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
kurtosis_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Kurtosis_係數',
    'Kurtosis_p值',
    'Kurtosis_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Kurtosis 單獨作為變數
X = df_regression_day_stats_with_returns[['Kurtosis']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_kurt = model.pvalues[1]
sig_kurt = get_significance(p_value_kurt)

if p_value_kurt < min_p_value:
    min_p_value = p_value_kurt
    best_combination = "僅 Kurtosis"

kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
    '變數組合': ["僅 Kurtosis"],
    'Kurtosis_係數': [model.params[1]],
    'Kurtosis_p值': [p_value_kurt],
    'Kurtosis_顯著性': [sig_kurt],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Kurtosis 添加到組合中
        all_vars = ['Kurtosis'] + list(combo)
        X = df_regression_day_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        kurtosis_idx = 1  # Kurtosis 在加入常數項後的索引
        p_value_kurt = model.pvalues[kurtosis_idx]
        sig_kurt = get_significance(p_value_kurt)
        
        # 組合名稱
        combo_name = "Kurtosis + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_kurt < min_p_value:
            min_p_value = p_value_kurt
            best_combination = combo_name
        
        # 儲存結果
        kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Kurtosis_係數': [model.params[kurtosis_idx]],
            'Kurtosis_p值': [p_value_kurt],
            'Kurtosis_顯著性': [sig_kurt],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Kurtosis 的 p 值排序結果
kurtosis_regression_results = kurtosis_regression_results.sort_values(by='Kurtosis_p值')

# 顯示結果
print("\nKurtosis 各變數組合的迴歸分析結果：")
print(kurtosis_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nKurtosis 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
kurtosis_regression_results.to_csv('kurtosis_regression_results_一個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 kurtosis_regression_results_一個點.csv")


''' 執行迴歸分析_每天_兩個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_兩個點_2025-04-28_刪1個極端值.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_day_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv('descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std


# 測試固定 Skewness 的迴歸分析
# 定義所有要測試的變數
variables = ['Mean', 'Std', 'Kurtosis', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
skewness_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Skewness_係數',
    'Skewness_p值',
    'Skewness_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Skewness 單獨作為變數
X = df_regression_day_stats_with_returns[['Skewness']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_skew = model.pvalues[1]
sig_skew = get_significance(p_value_skew)

if p_value_skew < min_p_value:
    min_p_value = p_value_skew
    best_combination = "僅 Skewness"

skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
    '變數組合': ["僅 Skewness"],
    'Skewness_係數': [model.params[1]],
    'Skewness_p值': [p_value_skew],
    'Skewness_顯著性': [sig_skew],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Skewness 添加到組合中
        all_vars = ['Skewness'] + list(combo)
        X = df_regression_day_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        skewness_idx = 1  # Skewness 在加入常數項後的索引
        p_value_skew = model.pvalues[skewness_idx]
        sig_skew = get_significance(p_value_skew)
        
        # 組合名稱
        combo_name = "Skewness + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_skew < min_p_value:
            min_p_value = p_value_skew
            best_combination = combo_name
        
        # 儲存結果
        skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Skewness_係數': [model.params[skewness_idx]],
            'Skewness_p值': [p_value_skew],
            'Skewness_顯著性': [sig_skew],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Skewness 的 p 值排序結果
skewness_regression_results = skewness_regression_results.sort_values(by='Skewness_p值')

# 顯示結果
print("\nSkewness 各變數組合的迴歸分析結果：")
print(skewness_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nSkewness 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
skewness_regression_results.to_csv('skewness_regression_results_兩個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 skewness_regression_results_兩個點.csv")


# 測試固定 Kurtosis 的迴歸分析
# 定義所有要測試的變數（排除 Kurtosis 本身）
variables = ['Mean', 'Std', 'Skewness', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
kurtosis_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Kurtosis_係數',
    'Kurtosis_p值',
    'Kurtosis_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_day_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Kurtosis 單獨作為變數
X = df_regression_day_stats_with_returns[['Kurtosis']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_kurt = model.pvalues[1]
sig_kurt = get_significance(p_value_kurt)

if p_value_kurt < min_p_value:
    min_p_value = p_value_kurt
    best_combination = "僅 Kurtosis"

kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
    '變數組合': ["僅 Kurtosis"],
    'Kurtosis_係數': [model.params[1]],
    'Kurtosis_p值': [p_value_kurt],
    'Kurtosis_顯著性': [sig_kurt],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Kurtosis 添加到組合中
        all_vars = ['Kurtosis'] + list(combo)
        X = df_regression_day_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        kurtosis_idx = 1  # Kurtosis 在加入常數項後的索引
        p_value_kurt = model.pvalues[kurtosis_idx]
        sig_kurt = get_significance(p_value_kurt)
        
        # 組合名稱
        combo_name = "Kurtosis + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_kurt < min_p_value:
            min_p_value = p_value_kurt
            best_combination = combo_name
        
        # 儲存結果
        kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Kurtosis_係數': [model.params[kurtosis_idx]],
            'Kurtosis_p值': [p_value_kurt],
            'Kurtosis_顯著性': [sig_kurt],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Kurtosis 的 p 值排序結果
kurtosis_regression_results = kurtosis_regression_results.sort_values(by='Kurtosis_p值')

# 顯示結果
print("\nKurtosis 各變數組合的迴歸分析結果：")
print(kurtosis_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nKurtosis 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
kurtosis_regression_results.to_csv('kurtosis_regression_results_兩個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 kurtosis_regression_results_兩個點.csv")



''' 執行迴歸分析_每週_一個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_一個點_2025-02-02.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_week_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv(f'descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std


# 測試固定 Skewness 的迴歸分析
# 定義所有要測試的變數
variables = ['Mean', 'Std', 'Kurtosis', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
skewness_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Skewness_係數',
    'Skewness_p值',
    'Skewness_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Skewness 單獨作為變數
X = df_regression_week_stats_with_returns[['Skewness']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_skew = model.pvalues[1]
sig_skew = get_significance(p_value_skew)

if p_value_skew < min_p_value:
    min_p_value = p_value_skew
    best_combination = "僅 Skewness"

skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
    '變數組合': ["僅 Skewness"],
    'Skewness_係數': [model.params[1]],
    'Skewness_p值': [p_value_skew],
    'Skewness_顯著性': [sig_skew],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Skewness 添加到組合中
        all_vars = ['Skewness'] + list(combo)
        X = df_regression_week_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        skewness_idx = 1  # Skewness 在加入常數項後的索引
        p_value_skew = model.pvalues[skewness_idx]
        sig_skew = get_significance(p_value_skew)
        
        # 組合名稱
        combo_name = "Skewness + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_skew < min_p_value:
            min_p_value = p_value_skew
            best_combination = combo_name
        
        # 儲存結果
        skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Skewness_係數': [model.params[skewness_idx]],
            'Skewness_p值': [p_value_skew],
            'Skewness_顯著性': [sig_skew],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Skewness 的 p 值排序結果
skewness_regression_results = skewness_regression_results.sort_values(by='Skewness_p值')

# 顯示結果
print("\nSkewness 各變數組合的迴歸分析結果：")
print(skewness_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nSkewness 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
skewness_regression_results.to_csv('skewness_regression_results_一個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 skewness_regression_results_一個點.csv")


# 測試固定 Kurtosis 的迴歸分析
# 定義所有要測試的變數（排除 Kurtosis 本身）
variables = ['Mean', 'Std', 'Skewness', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
kurtosis_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Kurtosis_係數',
    'Kurtosis_p值',
    'Kurtosis_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Kurtosis 單獨作為變數
X = df_regression_week_stats_with_returns[['Kurtosis']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_kurt = model.pvalues[1]
sig_kurt = get_significance(p_value_kurt)

if p_value_kurt < min_p_value:
    min_p_value = p_value_kurt
    best_combination = "僅 Kurtosis"

kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
    '變數組合': ["僅 Kurtosis"],
    'Kurtosis_係數': [model.params[1]],
    'Kurtosis_p值': [p_value_kurt],
    'Kurtosis_顯著性': [sig_kurt],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Kurtosis 添加到組合中
        all_vars = ['Kurtosis'] + list(combo)
        X = df_regression_week_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        kurtosis_idx = 1  # Kurtosis 在加入常數項後的索引
        p_value_kurt = model.pvalues[kurtosis_idx]
        sig_kurt = get_significance(p_value_kurt)
        
        # 組合名稱
        combo_name = "Kurtosis + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_kurt < min_p_value:
            min_p_value = p_value_kurt
            best_combination = combo_name
        
        # 儲存結果
        kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Kurtosis_係數': [model.params[kurtosis_idx]],
            'Kurtosis_p值': [p_value_kurt],
            'Kurtosis_顯著性': [sig_kurt],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Kurtosis 的 p 值排序結果
kurtosis_regression_results = kurtosis_regression_results.sort_values(by='Kurtosis_p值')

# 顯示結果
print("\nKurtosis 各變數組合的迴歸分析結果：")
print(kurtosis_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nKurtosis 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
kurtosis_regression_results.to_csv('kurtosis_regression_results_一個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 kurtosis_regression_results_一個點.csv")


''' 執行迴歸分析_每週_兩個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_兩個點_2025-02-02.csv')

# 對所有數值變數進行敘述統計
numeric_columns = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']
stats_summary = df_regression_week_stats_with_returns[numeric_columns].describe().T

# 顯示結果
print("\n變數的敘述統計：")
print(stats_summary.round(4))

# 將結果儲存為 CSV
stats_summary.to_csv('descriptive stats.csv', encoding='utf-8-sig')
print(f"\n敘述統計結果已儲存至 descriptive stats.csv")

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std


# 測試固定 Skewness 的迴歸分析
# 定義所有要測試的變數
variables = ['Mean', 'Std', 'Kurtosis', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
skewness_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Skewness_係數',
    'Skewness_p值',
    'Skewness_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Skewness 單獨作為變數
X = df_regression_week_stats_with_returns[['Skewness']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_skew = model.pvalues[1]
sig_skew = get_significance(p_value_skew)

if p_value_skew < min_p_value:
    min_p_value = p_value_skew
    best_combination = "僅 Skewness"

skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
    '變數組合': ["僅 Skewness"],
    'Skewness_係數': [model.params[1]],
    'Skewness_p值': [p_value_skew],
    'Skewness_顯著性': [sig_skew],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Skewness 添加到組合中
        all_vars = ['Skewness'] + list(combo)
        X = df_regression_week_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Skewness 的顯著水準
        skewness_idx = 1  # Skewness 在加入常數項後的索引
        p_value_skew = model.pvalues[skewness_idx]
        sig_skew = get_significance(p_value_skew)
        
        # 組合名稱
        combo_name = "Skewness + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_skew < min_p_value:
            min_p_value = p_value_skew
            best_combination = combo_name
        
        # 儲存結果
        skewness_regression_results = pd.concat([skewness_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Skewness_係數': [model.params[skewness_idx]],
            'Skewness_p值': [p_value_skew],
            'Skewness_顯著性': [sig_skew],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Skewness 的 p 值排序結果
skewness_regression_results = skewness_regression_results.sort_values(by='Skewness_p值')

# 顯示結果
print("\nSkewness 各變數組合的迴歸分析結果：")
print(skewness_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nSkewness 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
skewness_regression_results.to_csv('skewness_regression_results_兩個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 skewness_regression_results_兩個點.csv")


# 測試固定 Kurtosis 的迴歸分析
# 定義所有要測試的變數（排除 Kurtosis 本身）
variables = ['Mean', 'Std', 'Skewness', 'Median', 'Fear and Greed Index', 
             'VIX', 'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

# 建立一個 DataFrame 來儲存迴歸結果
kurtosis_regression_results = pd.DataFrame(columns=[
    '變數組合',
    'Kurtosis_係數',
    'Kurtosis_p值',
    'Kurtosis_顯著性',
    '變數數量',
    'R平方',
    'MSE'
])

# 設定 Y 變數
y = df_regression_week_stats_with_returns['T Return']

# 最小 p 值追蹤
min_p_value = 1.0
best_combination = ""

# 測試 Kurtosis 單獨作為變數
X = df_regression_week_stats_with_returns[['Kurtosis']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
p_value_kurt = model.pvalues[1]
sig_kurt = get_significance(p_value_kurt)

if p_value_kurt < min_p_value:
    min_p_value = p_value_kurt
    best_combination = "僅 Kurtosis"

kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
    '變數組合': ["僅 Kurtosis"],
    'Kurtosis_係數': [model.params[1]],
    'Kurtosis_p值': [p_value_kurt],
    'Kurtosis_顯著性': [sig_kurt],
    '變數數量': [1],
    'R平方': [model.rsquared],
    'MSE': [mse]
})], ignore_index=True)

# 測試不同數量變數的組合
for num_vars in range(1, len(variables) + 1):
    for combo in itertools.combinations(variables, num_vars):
        # 將 Kurtosis 添加到組合中
        all_vars = ['Kurtosis'] + list(combo)
        X = df_regression_week_stats_with_returns[all_vars]
        X = sm.add_constant(X)
        
        # 執行迴歸
        model = sm.OLS(y, X).fit()
        
        # 計算 MSE
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        # 判斷 Kurtosis 的顯著水準
        kurtosis_idx = 1  # Kurtosis 在加入常數項後的索引
        p_value_kurt = model.pvalues[kurtosis_idx]
        sig_kurt = get_significance(p_value_kurt)
        
        # 組合名稱
        combo_name = "Kurtosis + " + " + ".join(combo)
        
        # 追蹤最小的 p 值
        if p_value_kurt < min_p_value:
            min_p_value = p_value_kurt
            best_combination = combo_name
        
        # 儲存結果
        kurtosis_regression_results = pd.concat([kurtosis_regression_results, pd.DataFrame({
            '變數組合': [combo_name],
            'Kurtosis_係數': [model.params[kurtosis_idx]],
            'Kurtosis_p值': [p_value_kurt],
            'Kurtosis_顯著性': [sig_kurt],
            '變數數量': [len(all_vars)],
            'R平方': [model.rsquared],
            'MSE': [mse]
        })], ignore_index=True)

# 依照 Kurtosis 的 p 值排序結果
kurtosis_regression_results = kurtosis_regression_results.sort_values(by='Kurtosis_p值')

# 顯示結果
print("\nKurtosis 各變數組合的迴歸分析結果：")
print(kurtosis_regression_results.round(4))
print("\n顯著水準說明：")
print("*** : p < 0.01")
print("**  : p < 0.05")
print("*   : p < 0.1")

print(f"\nKurtosis 的 p 值最小組合是「{best_combination}」，p 值為 {min_p_value:.4f}")

# 將結果儲存為 CSV
kurtosis_regression_results.to_csv('kurtosis_regression_results_兩個點.csv', index=False, encoding='utf-8-sig')
print(f"\n迴歸分析結果已儲存至 kurtosis_regression_results_兩個點.csv")