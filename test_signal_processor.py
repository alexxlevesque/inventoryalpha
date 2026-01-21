import numpy as np
from src.brain.signal_processor import SignalProcessor

def test_signal_processor():
    # 1. Create a synthetic signal with known periodicity
    # 7-day cycle (weekly) and 30-day cycle (monthly)
    t = np.arange(365 * 2) # 2 years of daily data
    
    # Seasonality: Weekly (7 days) -> freq = 1/7
    # Seasonality: Monthly (30 days) -> freq = 1/30
    signal = (
        10 * np.sin(2 * np.pi * t / 7) + 
        5 * np.sin(2 * np.pi * t / 30) + 
        np.random.normal(0, 2, len(t)) + # Noise
        100 # Baseline demand
    )
    
    processor = SignalProcessor()
    
    print('Analyzing synthetic signal...')
    seasonality = processor.detect_seasonality(signal, top_k=5)
    
    print('Detected Seasonality:')
    for item in seasonality:
        print(f'- Period: {item["period"]} days (Mag: {item["magnitude"]:.2f})')
        
    # Check if 7 and 30 are roughly detected
    periods = [s['period'] for s in seasonality]
    has_weekly = any(6.8 < p < 7.2 for p in periods)
    has_monthly = any(29.0 < p < 31.0 for p in periods)
    
    if has_weekly and has_monthly:
        print('\nSUCCESS: Detected both weekly and monthly patterns!')
    else:
        print('\nWARNING: Failed to strictly detect expected patterns.')

if __name__ == '__main__':
    test_signal_processor()
