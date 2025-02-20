# 基本數據處理與分析套件
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 繪圖套件
import matplotlib.pyplot as plt
import seaborn as sns

# 統計相關套件
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

# 系統套件
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 30)
pd.set_option('display.float_format', '{:.4f}'.format)



''' 執行迴歸分析_每天_一個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_一個點_2025-02-02.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Skewness', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# # 執行OLS迴歸
# model = sm.OLS(y, X_1).fit()

# # 計算 MSE
# y_pred = model.predict(X_1)
# mse = np.mean((y - y_pred) ** 2)

# # 印出迴歸結果
# print("迴歸分析結果：")
# print(model.summary())
# print(f"\nMSE: {mse:.4f}")

# # 建立一個 DataFrame 來儲存迴歸結果
# regression_results = pd.DataFrame(columns=[
#     'Variable',
#     'Coefficient',
#     'Std Error',
#     'T-Stat',
#     'P-Value',
#     'Significance'
# ])

# # 取得模型結果
# variables = ['const', 'Skewness', 'Median', 'T-4 Return']
# coefficients = model.params
# std_errors = model.bse
# t_stats = model.tvalues
# p_values = model.pvalues

# # 判斷顯著性
# def get_significance(p_value):
#     if p_value < 0.01:
#         return '***'
#     elif p_value < 0.05:
#         return '**'
#     elif p_value < 0.1:
#         return '*'
#     return ''

# # 整理結果
# for var in variables:
#     idx = variables.index(var)
#     regression_results = pd.concat([regression_results, pd.DataFrame({
#         'Variable': [var],
#         'Coefficient': [coefficients[idx]],
#         'Std Error': [std_errors[idx]],
#         'T-Stat': [t_stats[idx]],
#         'P-Value': [p_values[idx]],
#         'Significance': [get_significance(p_values[idx])]
#     })], ignore_index=True)

# # 加入模型整體統計量
# model_stats = pd.DataFrame({
#     'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
#     'Value': [
#         model.rsquared,
#         model.rsquared_adj,
#         model.fvalue,
#         model.f_pvalue,
#         model.nobs,
#         mse
#     ]
# })

# # 顯示結果
# print("\n迴歸係數及顯著性：")
# print(regression_results.round(4))
# print("\n模型統計量：")
# print(model_stats.round(4))
# print("\n顯著水準說明：")
# print("*** : p < 0.01")
# print("**  : p < 0.05")
# print("*   : p < 0.1")

# Out-of-sample Analysis
data = pd.concat([y, X_1], axis=1)
T = len(data)
initial_window = int(T * 0.87)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'T-4 Return']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.show()


''' 執行迴歸分析_每天_兩個點方法 '''
# 讀取資料
df_regression_day_stats_with_returns = pd.read_csv('RND_regression_day_stats_all_data_兩個點_2025-02-02.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index','VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_day_stats_with_returns[var].mean()
    std = df_regression_day_stats_with_returns[var].std()
    df_regression_day_stats_with_returns[var] = (df_regression_day_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_1 = df_regression_day_stats_with_returns[[
    'Skewness', 'Median',
    'T-4 Return'
]]
y = df_regression_day_stats_with_returns['T Return']

# 加入常數項
X_1 = sm.add_constant(X_1)

# # 執行OLS迴歸
# model = sm.OLS(y, X_1).fit()

# # 計算 MSE
# y_pred = model.predict(X_1)
# mse = np.mean((y - y_pred) ** 2)

# # 印出迴歸結果
# print("迴歸分析結果：")
# print(model.summary())
# print(f"\nMSE: {mse:.4f}")

# # 建立一個 DataFrame 來儲存迴歸結果
# regression_results = pd.DataFrame(columns=[
#     'Variable',
#     'Coefficient',
#     'Std Error',
#     'T-Stat',
#     'P-Value',
#     'Significance'
# ])

# # 取得模型結果
# variables = ['const', 'Skewness', 'Median', 'T-4 Return']
# coefficients = model.params
# std_errors = model.bse
# t_stats = model.tvalues
# p_values = model.pvalues

# # 判斷顯著性
# def get_significance(p_value):
#     if p_value < 0.01:
#         return '***'
#     elif p_value < 0.05:
#         return '**'
#     elif p_value < 0.1:
#         return '*'
#     return ''

# # 整理結果
# for var in variables:
#     idx = variables.index(var)
#     regression_results = pd.concat([regression_results, pd.DataFrame({
#         'Variable': [var],
#         'Coefficient': [coefficients[idx]],
#         'Std Error': [std_errors[idx]],
#         'T-Stat': [t_stats[idx]],
#         'P-Value': [p_values[idx]],
#         'Significance': [get_significance(p_values[idx])]
#     })], ignore_index=True)

# # 加入模型整體統計量
# model_stats = pd.DataFrame({
#     'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
#     'Value': [
#         model.rsquared,
#         model.rsquared_adj,
#         model.fvalue,
#         model.f_pvalue,
#         model.nobs,
#         mse
#     ]
# })

# # 顯示結果
# print("\n迴歸係數及顯著性：")
# print(regression_results.round(4))
# print("\n模型統計量：")
# print(model_stats.round(4))
# print("\n顯著水準說明：")
# print("*** : p < 0.01")
# print("**  : p < 0.05")
# print("*   : p < 0.1")

# Out-of-sample Analysis
data = pd.concat([y, X_1], axis=1)
T = len(data)
initial_window = int(T * 0.87)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Skewness', 'Median', 'T-4 Return']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.show()


''' 執行迴歸分析_每週_一個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_一個點_2025-02-02.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX', 
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Kurtosis', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# # 執行OLS迴歸
# model = sm.OLS(y, X_4).fit()

# # 計算 MSE
# y_pred = model.predict(X_4)
# mse = np.mean((y - y_pred) ** 2)

# # 印出迴歸結果
# print("迴歸分析結果：")
# print(model.summary())
# print(f"\nMSE: {mse:.4f}")

# # 建立一個 DataFrame 來儲存迴歸結果
# regression_results = pd.DataFrame(columns=[
#     'Variable',
#     'Coefficient',
#     'Std Error',
#     'T-Stat',
#     'P-Value',
#     'Significance'
# ])

# # 取得模型結果
# variables = ['const', 'Kurtosis', 'Median', 'Fear and Greed Index']
# coefficients = model.params
# std_errors = model.bse
# t_stats = model.tvalues
# p_values = model.pvalues

# # 判斷顯著性
# def get_significance(p_value):
#     if p_value < 0.01:
#         return '***'
#     elif p_value < 0.05:
#         return '**'
#     elif p_value < 0.1:
#         return '*'
#     return ''

# # 整理結果
# for var in variables:
#     idx = variables.index(var)
#     regression_results = pd.concat([regression_results, pd.DataFrame({
#         'Variable': [var],
#         'Coefficient': [coefficients[idx]],
#         'Std Error': [std_errors[idx]],
#         'T-Stat': [t_stats[idx]],
#         'P-Value': [p_values[idx]],
#         'Significance': [get_significance(p_values[idx])]
#     })], ignore_index=True)

# # 加入模型整體統計量
# model_stats = pd.DataFrame({
#     'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
#     'Value': [
#         model.rsquared,
#         model.rsquared_adj,
#         model.fvalue,
#         model.f_pvalue,
#         model.nobs,
#         mse
#     ]
# })

# # 顯示結果
# print("\n迴歸係數及顯著性：")
# print(regression_results.round(4))
# print("\n模型統計量：")
# print(model_stats.round(4))
# print("\n顯著水準說明：")
# print("*** : p < 0.01")
# print("**  : p < 0.05")
# print("*   : p < 0.1")

# Out-of-sample Analysis
data = pd.concat([y, X_4], axis=1)
T = len(data)
initial_window = int(T * 0.9)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Kurtosis', 'Median', 'Fear and Greed Index']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.show()


''' 執行迴歸分析_每週_兩個點方法 '''
# 讀取資料
df_regression_week_stats_with_returns = pd.read_csv('RND_regression_week_stats_all_data_兩個點_2025-02-02.csv')

# 將所有數據進行標準化
variables_to_standardize = ['T Return', 'Mean', 'Std', 'Skewness', 'Kurtosis', 'Median', 'Fear and Greed Index', 'VIX',
                            'T-1 Return', 'T-2 Return', 'T-3 Return', 'T-4 Return']

for var in variables_to_standardize:
    mean = df_regression_week_stats_with_returns[var].mean()
    std = df_regression_week_stats_with_returns[var].std()
    df_regression_week_stats_with_returns[var] = (df_regression_week_stats_with_returns[var] - mean) / std

# 準備迴歸變數，用這個模型
X_4 = df_regression_week_stats_with_returns[[
     'Kurtosis', 'Median', 'Fear and Greed Index',
]]
y = df_regression_week_stats_with_returns['T Return']

# 加入常數項
X_4 = sm.add_constant(X_4)

# # 執行OLS迴歸
# model = sm.OLS(y, X_4).fit()

# # 計算 MSE
# y_pred = model.predict(X_4)
# mse = np.mean((y - y_pred) ** 2)

# # 印出迴歸結果
# print("迴歸分析結果：")
# print(model.summary())
# print(f"\nMSE: {mse:.4f}")

# # 建立一個 DataFrame 來儲存迴歸結果
# regression_results = pd.DataFrame(columns=[
#     'Variable',
#     'Coefficient',
#     'Std Error',
#     'T-Stat',
#     'P-Value',
#     'Significance'
# ])

# # 取得模型結果
# variables = ['const', 'Kurtosis', 'Median', 'Fear and Greed Index']
# coefficients = model.params
# std_errors = model.bse
# t_stats = model.tvalues
# p_values = model.pvalues

# # 判斷顯著性
# def get_significance(p_value):
#     if p_value < 0.01:
#         return '***'
#     elif p_value < 0.05:
#         return '**'
#     elif p_value < 0.1:
#         return '*'
#     return ''

# # 整理結果
# for var in variables:
#     idx = variables.index(var)
#     regression_results = pd.concat([regression_results, pd.DataFrame({
#         'Variable': [var],
#         'Coefficient': [coefficients[idx]],
#         'Std Error': [std_errors[idx]],
#         'T-Stat': [t_stats[idx]],
#         'P-Value': [p_values[idx]],
#         'Significance': [get_significance(p_values[idx])]
#     })], ignore_index=True)

# # 加入模型整體統計量
# model_stats = pd.DataFrame({
#     'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Number of Observations', 'MSE'],
#     'Value': [
#         model.rsquared,
#         model.rsquared_adj,
#         model.fvalue,
#         model.f_pvalue,
#         model.nobs,
#         mse
#     ]
# })

# # 顯示結果
# print("\n迴歸係數及顯著性：")
# print(regression_results.round(4))
# print("\n模型統計量：")
# print(model_stats.round(4))
# print("\n顯著水準說明：")
# print("*** : p < 0.01")
# print("**  : p < 0.05")
# print("*   : p < 0.1")

# Out-of-sample Analysis
data = pd.concat([y, X_4], axis=1)
T = len(data)
initial_window = int(T * 0.9)

# 執行分析
results_df, R2_OS = out_of_sample_analysis(
    data=data,
    initial_window=initial_window,  
    target_col='T Return',
    feature_cols=['const', 'Kurtosis', 'Median', 'Fear and Greed Index']
)

# 查看結果
print(f"R²_OS: {R2_OS:.4f}")

# 可視化預測結果
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(results_df['time'], results_df['actual'], label='Actual')
plt.plot(results_df['time'], results_df['predicted'], label='Predicted')
plt.plot(results_df['time'], results_df['historical_mean'], label='Historical Mean')
plt.legend()
plt.title('Out-of-Sample Forecasting Results')
plt.show()



''' Function '''
def out_of_sample_analysis(data, initial_window, target_col, feature_cols):
    """
    執行 out-of-sample 分析
    
    參數：
    data: DataFrame, 包含目標變量和特徵
    initial_window: int, 初始訓練窗口大小 (s₀)
    target_col: str, 目標變量的列名 (R_t+1)
    feature_cols: list, 特徵變量的列名列表 (X_t)
    
    返回：
    DataFrame: 包含實際值、預測值和歷史平均值
    float: R²_OS 值
    """
    
    # 初始化結果儲存
    results = []
    
    # 獲取總樣本長度
    T = len(data)
    
    # 對每個預測時點進行迭代
    for t in range(initial_window, T-1):
        # 獲取訓練數據
        train_data = data.iloc[:t]
        
        # 計算歷史平均值作為基準
        historical_mean = train_data[target_col].mean()
        
        # 擬合線性回歸模型
        model = LinearRegression()
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        model.fit(X_train, y_train)
        
        # 進行預測
        X_predict = data.iloc[t:t+1][feature_cols]
        predicted_value = model.predict(X_predict)[0]
        
        # 獲取實際值
        actual_value = data.iloc[t+1][target_col]
        
        # 儲存結果
        results.append({
            'time': data.index[t+1],
            'actual': actual_value,
            'predicted': predicted_value,
            'historical_mean': historical_mean
        })
    
    # 轉換結果為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 計算 R²_OS
    numerator = np.sum((results_df['actual'] - results_df['predicted'])**2)
    denominator = np.sum((results_df['actual'] - results_df['historical_mean'])**2)
    R2_OS = 1 - numerator/denominator
    
    return results_df, R2_OS