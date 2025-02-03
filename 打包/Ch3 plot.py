import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



option_raw_trades = pd.read_csv("deribit_data/BTC-option/BTC_option_all.csv")
option_raw_trades = option_raw_trades.query("date<'2024-05-01'")
print(len(option_raw_trades))
option_raw_trades.head(2)







option_raw_trades["observe_month"] =  pd.to_datetime(option_raw_trades["date"]).dt.strftime("%Y-%m")
option_raw_trades["days"] = (pd.to_datetime(option_raw_trades["expiration_date"], format='%d%b%y') - pd.to_datetime(option_raw_trades["date"])).dt.days 
option_raw_trades["T"] = option_raw_trades["days"] /365
option_raw_trades["days_category"] = pd.cut(option_raw_trades["days"], bins=[-float("inf"), 14, 30, 90, 180, float("inf")], 
                                            labels=["<=14 days", "15-30", "31-90", "90-180", ">180"], right=False)

option_raw_trades["moneyness"] = option_raw_trades["strike"] / option_raw_trades["index_price"]


call_raw = option_raw_trades.query("type=='Call'")
put_raw = option_raw_trades.query("type=='Put'")

moneyness_labels = [
    '<=0.9', '0.9-1.0', 
    '1.0-1.1', '1.1-1.2', '1.2-1.3', '1.3-1.4', '1.4-1.5', '1.5-1.6', 
    '1.6-1.7', '1.7-1.8', '1.8-1.9', '1.9-2.0', '>2.0'
]
call_raw["moneyness_category"] = pd.cut(call_raw["moneyness"], bins=[-float("inf"), 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 
                                        1.6, 1.7, 1.8, 1.9, 2, float("inf")], labels=moneyness_labels, right=False)

moneyness_labels = [
    '<=0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', 
    '1.0-1.1', '>1.1'
]
put_raw["moneyness_category"] = pd.cut(put_raw["moneyness"], bins=[-float("inf"), 0.7, 0.8, 0.9, 1, 1.1, float("inf")], labels=moneyness_labels, right=False)







tradescount_bytime = option_raw_trades.groupby(["type", "observe_month"]).count()[["volume"]]
tradescount_bytime.rename(columns={"volume": "trades"}, inplace=True)
tradescount_bytime = tradescount_bytime.reset_index()

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

bar_width = 0.35
unique_types = tradescount_bytime['type'].unique()
unique_dates = tradescount_bytime['observe_month'].unique()
index = range(len(unique_dates))

colors = {
    'Call': 'lightsalmon',
    'Put': 'lightskyblue'
}

for i, trade_type in enumerate(unique_types):
    type_data = tradescount_bytime[tradescount_bytime['type'] == trade_type]
    bar_positions = [x + i * bar_width for x in index]
    
    ax.bar(bar_positions, type_data.set_index('observe_month')['trades'], bar_width, label=trade_type, color=colors[trade_type])

ax.set_xlabel('Observe Month')
ax.set_ylabel('Number of Trades')
ax.set_title('Number of Trades by Type')
ax.set_xticks([r + bar_width / 2 for r in range(len(unique_dates))])
ax.set_xticklabels(unique_dates)

ax.legend()
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()







print(option_raw_trades.dtypes)
volume_bytime = (option_raw_trades
                .groupby(["type", "observe_month"], observed=True)
                .agg({"volume": "sum"})
                .reset_index())

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

bar_width = 0.35
unique_types = tradescount_bytime['type'].unique()
unique_dates = tradescount_bytime['observe_month'].unique()
index = range(len(unique_dates))

colors = {
    'Call': 'lightsalmon',
    'Put': 'lightskyblue'
}

for i, trade_type in enumerate(unique_types):
    type_data = volume_bytime[volume_bytime['type'] == trade_type]
    bar_positions = [x + i * bar_width for x in index]
    
    ax.bar(bar_positions, type_data.set_index('observe_month')['volume'], bar_width, label=trade_type, color=colors[trade_type])

ax.set_xlabel('Observe Month')
ax.set_ylabel('Volume (USD)')
ax.set_title('Volume by Type')
ax.set_xticks([r + bar_width / 2 for r in range(len(unique_dates))])
ax.set_xticklabels(unique_dates)

ax.legend()
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()







option_raw_trades[["type","moneyness_category"]]

call_volume = call_raw.groupby(["moneyness_category", "days_category"]).sum()[["volume","amount"]].unstack()
put_volume = put_raw.groupby(["moneyness_category", "days_category"]).sum()[["volume","amount"]].unstack()
put_volume

put_volume["volume"]/1000000

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 繪製 Call 的熱力圖
sns.heatmap(call_volume["volume"]/1000000, annot=True, fmt=".0f", cmap="Reds", cbar_kws={'label': 'Volume'}, ax=ax1)
ax1.set_title('Call Volume (millions)')
ax1.set_xlabel('Days')
ax1.set_ylabel('Moneyness (K/S)')

# 繪製 Put 的熱力圖
sns.heatmap(put_volume["volume"]/1000000, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': 'Volume'}, ax=ax2)
ax2.set_title('Put Volume (millions)')
ax2.set_xlabel('Days')
ax2.set_ylabel('Moneyness (K/S)')

plt.tight_layout()
plt.show()

# 熱力圖偷吃步
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 創建數據
call_data = np.array([
    [447, 561, 731, 524, 354],
    [1205, 611, 865, 276, 137],
    [2072, 1380, 1592, 483, 249],
    [239, 555, 1044, 382, 320],
    [43, 151, 572, 328, 212],
    [12, 61, 307, 216, 191],
    [6, 25, 183, 169, 194],
    [3, 17, 100, 120, 140],
    [1, 9, 70, 76, 120],
    [2, 4, 32, 57, 115],
    [1, 6, 24, 39, 61],
    [0, 3, 20, 25, 45],
    [0, 3, 45, 99, 392]
])

put_data = np.array([
    [12, 28, 120, 133, 157],
    [37, 85, 248, 132, 115],
    [214, 348, 677, 233, 159],
    [1676, 916, 1066, 268, 184],
    [835, 383, 589, 193, 157],
    [586, 171, 480, 821, 390]
])

# 創建標籤
moneyness_call = ['<=0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2', '1.2-1.3', '1.3-1.4', 
                  '1.4-1.5', '1.5-1.6', '1.6-1.7', '1.7-1.8', '1.8-1.9', '1.9-2.0', '>2.0']
moneyness_put = ['<=0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '>1.1']
days = ['<=14 days', '15-30', '31-90', '90-180', '>180']

# 設置全局字體大小
plt.rcParams.update({'font.size': 12})  # 調整軸標籤和刻度的基本字體大小

# 繪製買權熱圖
plt.figure(figsize=(12, 8), dpi=200)
sns.heatmap(call_data, annot=True, fmt='d', cmap='Reds', 
            xticklabels=days, yticklabels=moneyness_call, 
            annot_kws={'fontsize': 12})
plt.title('Call Volume (millions)', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Moneyness (K/S)', fontsize=14)
plt.tight_layout()
plt.show()

# 繪製賣權熱圖
plt.figure(figsize=(12, 8), dpi=200)
sns.heatmap(put_data, annot=True, fmt='d', cmap='Blues', 
            xticklabels=days, yticklabels=moneyness_put, 
            annot_kws={'fontsize': 12})
plt.title('Put Volume (millions)', fontsize=16)
plt.xlabel('Days', fontsize=14)
plt.ylabel('Moneyness (K/S)', fontsize=14)
plt.tight_layout()
plt.show()
