import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RealizedVariance:
    """
    Calculate realized variance from high-frequency data as benchmark
    """
    
    @staticmethod
    def calculate_daily_rv(high_freq_data, frequency='5min'):
        """
        Calculate daily realized variance from high-frequency data
        
        Parameters:
        -----------
        high_freq_data : pd.DataFrame
            High-frequency OHLC data
        frequency : str
            Sampling frequency ('5min', '15min', '1H')
            
        Returns:
        --------
        pd.Series : Daily realized variance
        """
        # Calculate returns
        returns = np.log(high_freq_data['Close'] / high_freq_data['Close'].shift(1)).dropna()
        
        # Group by day and sum squared returns
        daily_rv = returns.groupby(returns.index.date).apply(
            lambda x: np.sum(x ** 2)
        )
        
        return daily_rv
    
    @staticmethod
    def annualize_rv(daily_rv, trading_days=252):
        """Annualize realized variance"""
        return np.sqrt(daily_rv * trading_days)