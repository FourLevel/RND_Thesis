import requests
import time

# Deribit API URL
url = "https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time"

# 請根據需要修改參數
instrument_name = "BTC-PERPETUAL"  # 交易對名稱，這裡使用 BTC 永續合約
start_timestamp = 1672531200000  # 2023年1月1日 00:00:00 UTC 的毫秒時間戳
end_timestamp = 1704067199000  # 2023年12月31日 23:59:59 UTC 的毫秒時間戳
max_limit = 1000  # 每次 API 請求最多返回的數據條數

# 存儲所有交易數據
all_trades = []

while start_timestamp < end_timestamp:
    # 構建 API 請求的參數
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": start_timestamp,
        "count": max_limit
    }

    # 發送請求
    response = requests.get(url, params=params)
    data = response.json()

    # 檢查是否成功返回數據
    if "result" in data and len(data['result']['trades']) > 0:
        trades = data['result']['trades']
        all_trades.extend(trades)

        # 更新起始時間，使用最後一條交易的時間作為新的起點
        start_timestamp = trades[-1]['timestamp'] + 1

        # 打印已下載的數據數量
        print(f"Downloaded {len(trades)} trades, total: {len(all_trades)}")

        # 避免頻繁請求，設置一定的延遲
        time.sleep(0.5)
    else:
        print("No more trades or reached the end.")
        break


import pandas as pd

df = pd.DataFrame(all_trades)

# df.to_csv(r"D:\bitcoin_trades.csv", index=False)