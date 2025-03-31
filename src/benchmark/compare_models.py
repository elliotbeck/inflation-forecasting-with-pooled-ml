"""
Model Benchmarking Module

This module runs multiple time series forecasting models — Random Walk, Auto ARIMA,
Random Forest, and Pooled Random Forest — on a panel of inflation data, computes
month-over-month (MoM) inflation, and evaluates models using RMSE.
"""

import pandas as pd

from src.evaluation.metrics import forecast_rmse
from src.features.transforms import compute_mom_inflation, winsorize_series
from src.models.auto_arima import rolling_auto_arima_forecast
from src.models.pooled_random_forest import rolling_rf_pooled_forecast
from src.models.random_forest import rolling_rf_forecast
from src.models.random_walk import rolling_rw_forecast


def compare_models(data: pd.DataFrame) -> pd.DataFrame:
    """
    Benchmarks multiple forecasting models on a panel of inflation series.

    This function computes month-over-month (MoM) inflation, applies winsorization to smooth
    extreme values, and evaluates the following models for each country:

        - Random Walk (RW)
        - Auto ARIMA
        - Country-specific Random Forest (RF)
        - Pooled Random Forest with multitarget forecasting (RF_Pooled)

    RMSE is computed for each model and country, and the results are returned as a summary table.

    Parameters:
        data (pd.DataFrame): Wide-format DataFrame of price-level series
                             (rows = time, columns = countries or indicators)

    Returns:
        pd.DataFrame: Summary table with RMSEs per model, indexed by country/indicator
                      Columns: ['RW_RMSE', 'AUTO_ARIMA_RMSE', 'RF_RMSE', 'RF_Pooled_Multitarget_RMSE']
    """
    results = {}
    mom_data = data.apply(compute_mom_inflation)
    mom_data_winsorized = mom_data.apply(winsorize_series)

    print("Running pooled RF...", flush=True)
    pooled_forecasts = rolling_rf_pooled_forecast(mom_data_winsorized)

    for col in data.columns:
        mom = mom_data[col]
        mom_win = mom_data_winsorized[col]

        print(f"Running RW, ARIMA, RF for {col}...", flush=True)
        forecast_rw = rolling_rw_forecast(mom)
        forecast_auto_arima = rolling_auto_arima_forecast(mom)
        forecast_rf = rolling_rf_forecast(mom_win)
        forecast_rf_pooled = pooled_forecasts[col]

        rmse_rw = forecast_rmse(mom, forecast_rw)
        rmse_auto_arima = forecast_rmse(mom, forecast_auto_arima)
        rmse_rf = forecast_rmse(mom, forecast_rf)
        rmse_rf_pooled = forecast_rmse(mom, forecast_rf_pooled)

        results[col] = {
            "RW_RMSE": rmse_rw,
            "AUTO_ARIMA_RMSE": rmse_auto_arima,
            "RF_RMSE": rmse_rf,
            "RF_Pooled_RMSE": rmse_rf_pooled,
        }

        print(pd.DataFrame(results).T, flush=True)

    return pd.DataFrame(results).T
