import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def validate_moving_averages(df):
    """Validate moving average calculations."""
    results = []
    
    # Get all MA columns
    ma_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'wma', 'tema', 'trima'])]
    
    for col in ma_cols:
        # MAs should be between min and max of close price
        valid_range = df['close'].min() <= df[col].max() <= df[col].max()
        
        # MAs should have high correlation with close price
        correlation = df[col].corr(df['close'])
        
        # MAs should be less volatile than price
        ma_std = df[col].std()
        price_std = df['close'].std()
        
        results.append({
            'indicator': col,
            'valid_range': valid_range,
            'correlation': correlation,
            'relative_volatility': ma_std / price_std,
            'null_pct': (df[col].isnull().sum() / len(df)) * 100
        })
    
    return pd.DataFrame(results)

def validate_oscillators(df):
    """Validate oscillator calculations."""
    results = []
    
    # RSI validation
    rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
    for col in rsi_cols:
        valid_range = (0 <= df[col].min() <= 100) and (0 <= df[col].max() <= 100)
        results.append({
            'indicator': col,
            'valid_range': valid_range,
            'min_value': df[col].min(),
            'max_value': df[col].max(),
            'null_pct': (df[col].isnull().sum() / len(df)) * 100
        })
    
    # Stochastic validation
    stoch_cols = [col for col in df.columns if 'stoch' in col.lower()]
    for col in stoch_cols:
        valid_range = (0 <= df[col].min() <= 100) and (0 <= df[col].max() <= 100)
        results.append({
            'indicator': col,
            'valid_range': valid_range,
            'min_value': df[col].min(),
            'max_value': df[col].max(),
            'null_pct': (df[col].isnull().sum() / len(df)) * 100
        })
    
    return pd.DataFrame(results)

def validate_volume_indicators(df):
    """Validate volume-based indicators."""
    results = []
    
    vol_cols = [col for col in df.columns if any(x in col.lower() for x in ['obv', 'cmf', 'mfi', 'pvol', 'pvt'])]
    
    for col in vol_cols:
        # Check correlation with volume
        vol_corr = df[col].corr(df['volume'])
        
        # Check for extreme values
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        extreme_values = (z_scores > 3).sum() / len(z_scores)
        
        results.append({
            'indicator': col,
            'volume_correlation': vol_corr,
            'extreme_values_pct': extreme_values * 100,
            'null_pct': (df[col].isnull().sum() / len(df)) * 100
        })
    
    return pd.DataFrame(results)

def validate_trend_indicators(df):
    """Validate trend indicators."""
    results = []
    
    # MACD validation
    macd_cols = [col for col in df.columns if 'macd' in col.lower()]
    for col in macd_cols:
        # MACD should cross zero line multiple times
        zero_crosses = ((df[col] > 0) != (df[col].shift(1) > 0)).sum()
        
        results.append({
            'indicator': col,
            'zero_crosses': zero_crosses,
            'null_pct': (df[col].isnull().sum() / len(df)) * 100
        })
    
    return pd.DataFrame(results)

def validate_statistical_properties(df):
    """Validate statistical properties of all indicators."""
    results = []
    
    # Skip price, volume, and date/time columns
    skip_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'time', 'timestamp']
    
    for col in df.columns:
        # Skip non-numeric columns and known skip columns
        if col.lower() in skip_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
            
        try:
            # Basic statistics
            stats_dict = {
                'indicator': col,
                'mean': series.mean(),
                'std': series.std(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis(),
                'null_pct': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': series.nunique(),
                'constant': series.nunique() == 1
            }
            
            # Stationarity (basic check)
            split_point = len(series) // 2
            first_half_mean = series[:split_point].mean()
            second_half_mean = series[split_point:].mean()
            stats_dict['mean_shift'] = abs(first_half_mean - second_half_mean)
            
            results.append(stats_dict)
        except Exception as e:
            print(f"Warning: Could not calculate statistics for {col}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def run_validation(input_folder):
    """Run all validations on the generated indicators."""
    print("🔍 Starting indicator validation...")
    
    for file in os.listdir(input_folder):
        if not file.endswith('_pta_features.csv'):
            continue
            
        print(f"\n📊 Analyzing {file}")
        df = pd.read_csv(os.path.join(input_folder, file))
        
        # Run validations
        print("\n1️⃣ Moving Average Validation:")
        ma_results = validate_moving_averages(df)
        print(ma_results[ma_results['valid_range'] == False])
        
        print("\n2️⃣ Oscillator Validation:")
        osc_results = validate_oscillators(df)
        print(osc_results[osc_results['valid_range'] == False])
        
        print("\n3️⃣ Volume Indicator Validation:")
        vol_results = validate_volume_indicators(df)
        print(vol_results[vol_results['extreme_values_pct'] > 10])
        
        print("\n4️⃣ Trend Indicator Validation:")
        trend_results = validate_trend_indicators(df)
        print(trend_results[trend_results['zero_crosses'] < 5])
        
        print("\n5️⃣ Statistical Validation:")
        stat_results = validate_statistical_properties(df)
        
        # Report problematic indicators
        print("\n⚠️ Potential Issues:")
        
        # Constant or near-constant indicators
        constant_indicators = stat_results[stat_results['constant']]
        if not constant_indicators.empty:
            print("\nConstant indicators (no variation):")
            print(constant_indicators['indicator'].tolist())
        
        # High null percentage
        high_null = stat_results[stat_results['null_pct'] > 20]
        if not high_null.empty:
            print("\nIndicators with >20% null values:")
            print(high_null['indicator'].tolist())
        
        # Extreme values
        extreme_stats = stat_results[
            (stat_results['skew'].abs() > 5) | 
            (stat_results['kurtosis'].abs() > 30)
        ]
        if not extreme_stats.empty:
            print("\nIndicators with extreme statistical properties:")
            print(extreme_stats['indicator'].tolist())
        
        print("\n✅ Validation complete!")

if __name__ == "__main__":
    output_folder = "ibkr_features_pta_multi"  # Same as in generate script
    run_validation(output_folder)