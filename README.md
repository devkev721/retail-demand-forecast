# Retail Demand Forecast & Inventory Risk Model

## Overview
Machine learning project to forecast retail demand and compare performance against a baseline model.

## Problem
Inaccurate demand forecasting leads to stockouts or overstock.

## Solution
Built a forecasting model using Python and compared it with a baseline approach.

## Results
- ML Error: 2.3M
- Baseline Error: 13.1M

## Dashboard
![Dashboard](dashboard/dashboard.png)

## Key Insights
- ML model reduces forecast error significantly compared to baseline
- Demand shows strong seasonal peaks → risk of stockouts
- Baseline model fails to capture variability in demand

## How to Run
1. Install dependencies:
   pip install pandas scikit-learn matplotlib

2. Run script:
   python src/forecast.py

## Impact
Reduced forecast error by ~80%, enabling better inventory planning decisions.

## Tools
- Python
- Power BI

## Data
Dataset sourced from Walmart Sales Forecasting dataset on Kaggle.
