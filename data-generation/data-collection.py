from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# === CONFIG ===
TICKERS = ['AAPL', 'MSFT', 'GOOG']
EXCHANGE = 'SMART'
CURRENCY = 'USD'
SEC_TYPE = 'STK'
BAR_SIZE = '1 day'
DURATION = '30 D'
END_DATE = datetime.now()
YEARS_BACK = 10
OUTPUT_DIR = 'ibkr_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CONNECT ===
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=99)

def fetch_chunked_history(symbol):
    print(f"\n▶ Fetching: {symbol}")
    contract = Stock(symbol, EXCHANGE, CURRENCY)
    ib.qualifyContracts(contract)
    df_all = pd.DataFrame()

    end = END_DATE
    total_days = YEARS_BACK * 365
    chunk_days = 30
    iterations = total_days // chunk_days

    for i in range(iterations):
        print(f"  - Pulling: {end.strftime('%Y%m%d')}")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=DURATION,
            barSizeSetting=BAR_SIZE,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        df = util.df(bars)
        if df is None or df.empty:
            print("    ⚠️ No data, skipping this chunk.")
            return
        df_all = pd.concat([df, df_all])
        end = df['date'].min() - timedelta(days=1)
        #time.sleep(1.5)  # avoid pacing violation

    df_all.drop_duplicates(subset='date', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_all.to_csv(f"{OUTPUT_DIR}/{symbol}.csv", index=False)
    print(f"✅ Saved: {symbol}.csv ({len(df_all)} rows)")

for symbol in TICKERS:
    file_path = f"{OUTPUT_DIR}/{symbol}.csv"
    if os.path.exists(file_path):
        print(f"⏩ Skipping {symbol}, file already exists.")
        continue
    fetch_chunked_history(symbol)


ib.disconnect()



import os
import pandas as pd
from ta.trend import *
from ta.momentum import *
from ta.volatility import *
from ta.volume import *

# Folder paths
input_folder = "ibkr_data"
output_folder = "ibkr_features"
os.makedirs(output_folder, exist_ok=True)

# Extended timeframes
timeframes = [1, 3, 5, 7, 10, 14, 20, 26, 30, 50, 52, 60, 90, 100, 120, 150, 180, 200, 250]

def generate_many_features(df):
    df = df.copy()
    for w in timeframes:
        # Trend
        df[f'sma_{w}'] = SMAIndicator(df['close'], window=w).sma_indicator()
        df[f'ema_{w}'] = EMAIndicator(df['close'], window=w).ema_indicator()
        df[f'adx_{w}'] = ADXIndicator(df['high'], df['low'], df['close'], window=w).adx()
        df[f'cci_{w}'] = CCIIndicator(df['high'], df['low'], df['close'], window=w).cci()

        # Momentum
        df[f'rsi_{w}'] = RSIIndicator(df['close'], window=w).rsi()
        df[f'roc_{w}'] = ROCIndicator(df['close'], window=w).roc()
        df[f'willr_{w}'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=w).williams_r()

        # Volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=w)
        df[f'atr_{w}'] = atr.average_true_range()
        df[f'tr_{w}'] = atr.true_range()

        bb = BollingerBands(df['close'], window=w)
        df[f'bb_high_{w}'] = bb.bollinger_hband()
        df[f'bb_low_{w}'] = bb.bollinger_lband()
        df[f'bb_width_{w}'] = bb.bollinger_wband()
        df[f'bb_percent_{w}'] = bb.bollinger_pband()

        # Volume
        df[f'obv_{w}'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df[f'mfi_{w}'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=w).money_flow_index()

        # Rolling stats
        df[f'roll_mean_{w}'] = df['close'].rolling(window=w).mean()
        df[f'roll_std_{w}'] = df['close'].rolling(window=w).std()
        df[f'return_{w}'] = df['close'].pct_change(periods=w)

    df = df.dropna().reset_index(drop=True)
    return df

# Process all CSVs
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        symbol = file.replace(".csv", "")
        print(f"▶ Generating features for: {symbol}")
        df_raw = pd.read_csv(os.path.join(input_folder, file))
        df_feat = generate_many_features(df_raw)
        df_feat.to_csv(os.path.join(output_folder, f"{symbol}_features.csv"), index=False)
        print(f"✅ Saved: {symbol}_features.csv with {df_feat.shape[1]} columns")

# Confirm how many features were generated in one file
sample_file = os.listdir(output_folder)[0]
sample_df = pd.read_csv(os.path.join(output_folder, sample_file))
feature_count = sample_df.shape[1]

feature_count

