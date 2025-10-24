import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
import warnings
import numpy as np

class VolatilityEstimators:
    def __init__(self, window: int = 30, trading_days: int = 252):
        self.window = window
        self.trading_days = trading_days

    def close_to_close(self, closes):
        """Classical close-to-close estimator"""
        returns = np.log(closes / closes.shift(1)).dropna()
        recent_returns = returns.iloc[-self.window:]
        variance = np.var(recent_returns, ddof=1)
        return np.sqrt(variance * self.trading_days)

    def parkinson(self, highs, lows):
        """Parkinson (1980) - High/Low range estimator"""
        # Formula: σ_P(T) = √(1/T) * √(1/(4ln2) * Σ(ln(H_i/L_i))²)
        log_hl_ratio = np.log(highs / lows)
        recent_ratios = log_hl_ratio.iloc[-self.window:]
        
        sum_squared_log_ratios = np.sum(recent_ratios ** 2)
        variance = (1 / (4 * np.log(2))) * (1 / self.window) * sum_squared_log_ratios
        return np.sqrt(variance * self.trading_days)

    def garman_klass(self, opens, highs, lows, closes):
        """Garman-Klass (1980) - OHLC estimator"""
        # Formula: σ_GK(T) = √(1/T) * √[Σ(0.5*(ln(H_i/L_i))² - (2ln2-1)*(ln(C_i/O_i))²)]
        log_hl_ratio = np.log(highs / lows)
        log_co_ratio = np.log(closes / opens)
        
        recent_hl = log_hl_ratio.iloc[-self.window:]
        recent_co = log_co_ratio.iloc[-self.window:]
        
        term1 = 0.5 * (recent_hl ** 2)
        term2 = (2 * np.log(2) - 1) * (recent_co ** 2)
        variance = (1 / self.window) * np.sum(term1 - term2)
        return np.sqrt(variance * self.trading_days)

    def rogers_satchell(self, opens, highs, lows, closes):
        """Rogers-Satchell (1991) estimator for non-zero mean returns"""
        # Formula: σ_RS(T) = √(1/T) * √[Σ(ln(H_i/C_i)*ln(H_i/O_i) + ln(L_i/C_i)*ln(L_i/O_i))]
        log_open = np.log(opens)
        log_high = np.log(highs)
        log_low = np.log(lows)
        log_close = np.log(closes)
        
        o = log_open.iloc[-self.window:]
        h = log_high.iloc[-self.window:]
        l = log_low.iloc[-self.window:]
        c = log_close.iloc[-self.window:]
        
        term1 = (h - c) * (h - o)  # ln(H_i/C_i) * ln(H_i/O_i)
        term2 = (l - c) * (l - o)  # ln(L_i/C_i) * ln(L_i/O_i)
        variance = (1 / self.window) * np.sum(term1 + term2)
        return np.sqrt(variance * self.trading_days)

    def yang_zhang(self, opens, highs, lows, closes):
        """Yang-Zhang (2000) - Overnight/Intraday volatility estimator"""
        # Formula: σ_YZ(T) = √[σ_co² + kσ_oc² + (1-k)σ_RS²]
        log_open = np.log(opens)
        log_close = np.log(closes)
        
        # Close-to-open returns (overnight)
        close_to_open = log_open - log_close.shift(1)
        co_returns = close_to_open.iloc[-self.window:]
        
        # Open-to-close returns (intraday)
        open_to_close = log_close - log_open
        oc_returns = open_to_close.iloc[-self.window:]
        
        # Calculate components with proper variance (not squared yet)
        var_co = np.var(co_returns, ddof=1)  # σ_co²
        var_oc = np.var(oc_returns, ddof=1)  # σ_oc²
        
        # Rogers-Satchell component (already squared in the method)
        rs_variance = self.rogers_satchell(opens, highs, lows, closes) ** 2 / self.trading_days
        
        # k parameter
        n = len(oc_returns)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        
        # Yang-Zhang variance
        variance = var_co + k * var_oc + (1 - k) * rs_variance
        return np.sqrt(variance * self.trading_days)

class PerformanceMetrics:
    """Quantify estimator performance using statistical metrics"""
    
    @staticmethod
    def bias(estimates, true_value):
        """Calculate bias: E[θ_hat] - θ"""
        return np.mean(estimates - true_value)
    
    @staticmethod
    def mse(estimates, true_value):
        """Calculate Mean Squared Error: E[(θ_hat - θ)²]"""
        return np.mean((estimates - true_value) ** 2)
    
    @staticmethod
    def rmse(estimates, true_value):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(PerformanceMetrics.mse(estimates, true_value))
    
    @staticmethod
    def efficiency(estimates, benchmark_estimates):
        """Calculate relative efficiency: Var(benchmark) / Var(estimator)"""
        return np.var(benchmark_estimates) / np.var(estimates)
    
    @staticmethod
    def calculate_all_metrics(estimator_results, true_volatility, benchmark='Close-to-Close'):
        """Calculate all performance metrics for multiple estimators"""
        metrics_data = []
        
        for estimator_name, estimates in estimator_results.items():
            if len(estimates) > 0 and not np.all(np.isnan(estimates)):
                metrics_data.append({
                    'Estimator': estimator_name,
                    'Bias': PerformanceMetrics.bias(estimates, true_volatility),
                    'MSE': PerformanceMetrics.mse(estimates, true_volatility),
                    'RMSE': PerformanceMetrics.rmse(estimates, true_volatility),
                    'Efficiency': PerformanceMetrics.efficiency(estimates, estimator_results[benchmark]),
                    'Std_Dev': np.std(estimates),
                    'Mean_Estimate': np.mean(estimates)
                })
        
        return pd.DataFrame(metrics_data)

class DataSimulator:
    """Simulate price paths with known true volatility"""
    
    def __init__(self, S0=100, mu=0.08, sigma=0.2, seed=42):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        np.random.seed(seed)
    
    def simulate_gbm(self, n_days=252, dt=1/252):
        """Geometric Brownian Motion with true integrated variance"""
        mu_daily = self.mu * dt
        sigma_daily = self.sigma * np.sqrt(dt)
        true_variance = sigma_daily ** 2
        
        # Generate price path
        returns = np.random.normal(mu_daily, sigma_daily, n_days)
        prices = self.S0 * np.exp(np.cumsum(returns))
        prices = np.concatenate([[self.S0], prices])
        
        # Generate OHLC data
        ohlc = self._generate_ohlc(prices, sigma_daily)
        return ohlc, true_variance
    
    def _generate_ohlc(self, prices, daily_vol):
        """Generate realistic OHLC data from closing prices"""
        n_days = len(prices) - 1
        opens, highs, lows, closes = [], [], [], []
        
        for i in range(1, len(prices)):
            close_prev = prices[i-1]
            close_today = prices[i]
            
            # Realistic OHLC generation
            open_today = close_prev * (1 + np.random.normal(0, daily_vol/3))
            daily_range = np.abs(np.random.normal(0, daily_vol * 1.5))
            
            high_today = max(open_today, close_today) * (1 + daily_range)
            low_today = min(open_today, close_today) * (1 - daily_range)
            
            # Ensure high > low
            high_today = max(high_today, open_today, close_today)
            low_today = min(low_today, open_today, close_today)
            
            opens.append(open_today)
            highs.append(high_today)
            lows.append(low_today)
            closes.append(close_today)
        
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        return pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes}, index=dates)