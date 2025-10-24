# Volatility-Estimator
# Description
A Python package that provides advanced volatility estimators for financial data,
including close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, and Yang–Zhang methods. It provides comprehensive implementation and comparison of all five volatility estimators
Useful for quantitative research, portfolio optimization, and risk management.


## Table of Contents
- [Installation](#installation)
- [Executive Summary](#executive_summary)
- [Notebook Overview](#notebook_overview)
- [Classes](#Classes)
- [Examples](#examples)
- [License](#license)  
- [Author's Info](#author's_info) 



## Installation

```bash
pip install Volatility-Estimator

or

```bash
git clone https://github.com/Eishita0506/Volatility-Estimator.git
cd volatility-estimators
pip install -e .

## Executive Summary
This analysis evaluates 5 prominent volatility estimators across simulated and real market data.
Key findings:

1. YANG-ZHANG emerges as the most robust estimator overall, handling both drift and overnight jumps effectively. 14 times more accurate than close-to-close

2. ROGERS-SATCHELL performs best in trending markets and is drift-independent. 6 times more efficient than close-to-close

3. PARKINSON offers good efficiency with only high/low data. 5.2 times more efficient than close-to-close

4. GARMAN-KLASS provides maximum efficiency with full OHLC data but is sensitive to drift. 7.4 times more efficient than close-to-close

5. CLOSE-TO-CLOSE remains useful for legacy data but is inefficient.

RECOMMENDATIONS:
---------------
PRIMARY RECOMMENDATION: Use Yang-Zhang estimator for most applications.

SCENARIO-SPECIFIC GUIDANCE:

1. RISK MANAGEMENT & OPTIONS PRICING:
   • Primary: Yang-Zhang
   • Alternative: Rogers-Satchell

2. HIGH-FREQUENCY TRADING:
   • Primary: Parkinson  
   • Alternative: Garman-Klass

3. ACADEMIC RESEARCH:
   • Primary: Yang-Zhang
   • Alternative: Close-to-Close (for comparison)

4. LEGACY DATA ANALYSIS:
   • Primary: Close-to-Close
   • Only option when only closing prices available

DATA REQUIREMENTS:
-----------------
• Close-to-Close: Close prices only
• Parkinson: High/Low prices  
• Garman-Klass: Full OHLC data
• Rogers-Satchell: Full OHLC data
• Yang-Zhang: Full OHLC data

IMPLEMENTATION NOTES:
--------------------
- All estimators implemented with 30-day window as standard
- Results are annualized assuming 252 trading days
- Yang-Zhang requires minimum 31 days of data
- Consider realized variance from high-frequency data as benchmark when available

## Notebook Overview
Notebook 1: Implementation & Validation
- Implements 5 volatility estimators (Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
- Tests on simulated data with known true volatility
- Validates estimator accuracy and functionality

Notebook 2: Real Market Data Analysis
- Downloads real OHLC data across multiple asset classes
- Uses high-frequency realized variance as benchmark
- Compares estimator performance on real-world data

Notebook 3: Real Market Data Analysis
- Applies volatility estimators to real OHLC market data using `yfinance`.  
- Compares Parkinson, Garman–Klass, Rogers–Satchell, and Yang–Zhang methods against realized variance, with visual insights on estimator performance and stability.

Notebook 4: Robustness Testing
- Tests estimators under different market regimes (normal, high volatility, trending, jumps)
- Measures performance stability across market conditions
- Identifies most robust estimators

Notebook 5: Recommendation Framework
- Intelligent recommender system for estimator selection
- Decision trees based on data availability and market conditions
- Practical guidance for different use cases

Notebook 6: Example Usage
- Example of a real time use case for reference

## Classes
- `VolatilityEstimators`: Implements various volatility formulas
- `PerformanceMetrics`: Evaluates estimator accuracy
- `DataSimulator`: Generates synthetic financial data
- `RealizedVariance`: Calculates realized variance from OHLC data
- `VolatilityRecommender`: Suggests the most efficient estimator


## Examples
Check out the Jupyter notebook in "D:\New Recordings\DS ML Bootcamp\Volatility Estimator\6.Example_Usage.ipynb" for a full demo.


## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author's Info
**Eishita Bordoloi**  
📧 eishita2005@gmail.com  
🔗 [GitHub](https://github.com/Eishita0506)


