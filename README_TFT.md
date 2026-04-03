# Walmart Weekly Sales Forecasting

This project analyzes Walmart weekly sales data and builds a time-series forecasting model using Temporal Fusion Transformer (TFT).

## Project Files

- `walmart_forecasting_weekly_sales.ipynb` - main notebook with data loading, exploratory data analysis, feature engineering, model training, evaluation, and forecasting.
- `Walmart_Sales.csv` - dataset used for analysis and model training.

## What the Notebook Covers

1. Data loading and inspection
2. Exploratory data analysis
3. Visual analysis of sales patterns
4. Feature engineering for time-series modeling
5. TFT model training
6. Model evaluation against a baseline
7. Future weekly sales forecasting

## Key Findings

- Weekly sales show clear seasonal and store-level variation.
- Holiday weeks tend to behave differently from non-holiday weeks.
- Unemployment, fuel price, and CPI show weak to moderate visual relationships with weekly sales.
- The TFT model performs better than a simple baseline model on the validation set.

## Model Evaluation

The notebook reports validation metrics for the trained TFT model and compares them with a baseline model.

Example metrics from the current run:

- TFT MAE: 37,932.52
- TFT RMSE: 53,172.69
- Baseline MAE: 43,988.99
- Baseline RMSE: 63,596.68

## Future Forecasting

The notebook also generates a 5-week forecast.

Note:
- CPI, unemployment, fuel price, and temperature are not known in advance.
- The forecast currently uses the last observed values for those variables as placeholders.
- For a more realistic forecast, replace those placeholders with actual future values or scenario-based assumptions.

## Requirements

Typical packages used in the notebook include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- lightning
- pytorch-forecasting
- statsmodels
- ydata_profiling

## How to Run

1. Open `walmart_forecasting_weekly_sales.ipynb` in VS Code or Jupyter.
2. Install the required packages if needed.
3. Run the notebook cells from top to bottom.
4. Review the evaluation metrics and forecast plots at the end.

## Notes

- The notebook uses log-transformed weekly sales for TFT training.
- Some plots and metrics depend on executing earlier cells in order.
- If you change the feature engineering or forecast assumptions, rerun the training and evaluation cells.
