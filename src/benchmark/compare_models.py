"""
Model Benchmarking Module

This module runs multiple time series forecasting models — Random Walk, Auto ARIMA,
Random Forest, and Pooled Random Forest — on a panel of inflation data, computes
month-over-month (MoM) inflation, and evaluates models using RMSE.
"""

import pandas as pd

from evaluation.metrics import forecast_rmse
from features.transforms import compute_mom_inflation
from models.auto_arima import rolling_auto_arima_forecast
from models.pooled_random_forest import rolling_rf_pooled_forecast
from models.random_forest import rolling_rf_forecast
from models.random_walk import rolling_rw_forecast
from models.pooled_ffnn import rolling_ffnn_pooled_forecast
from models.pooled_elastic_net import rolling_elastic_net_pooled_forecast
from features.create_dummies import create_regional_dummies


def compare_models(data: pd.DataFrame, data_tabular: pd.DataFrame) -> pd.DataFrame:
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
        data_tabular (pd.DataFrame): Tabular DataFrame with lagged features and target values
                             (rows = countries x time, columns = features)

    Returns:
        pd.DataFrame: Summary table with RMSEs per model, indexed by country/indicator
                      Columns: ['RW_RMSE', 'AUTO_ARIMA_RMSE', 'RF_RMSE', 'RF_Pooled_Multitarget_RMSE']
    """
    results = {}
    mom_data = data.apply(compute_mom_inflation)
    data_tabular_pooled = create_regional_dummies(
        data_tabular, "data/country_continent_map.csv"
    )
    # One-hot encode countries
    country_dummies = pd.get_dummies(data_tabular_pooled["Country"], prefix="country")
    data_tabular_pooled = pd.concat([data_tabular_pooled, country_dummies], axis=1)

    print("Running pooled LR...", flush=True)
    pooled_forecasts_lr = rolling_elastic_net_pooled_forecast(data_tabular_pooled)

    print("Running pooled RF...", flush=True)
    pooled_forecasts_rf = rolling_rf_pooled_forecast(data_tabular_pooled)

    print("Running FFNN...", flush=True)
    pooled_forecasts_ffnn = rolling_ffnn_pooled_forecast(data_tabular_pooled)

    for col in data.columns:
        mom = mom_data[col]

        print(f"Running RW, ARIMA, RF for {col}...", flush=True)
        forecast_auto_arima = rolling_auto_arima_forecast(mom)
        forecast_rw = rolling_rw_forecast(mom, forecast_auto_arima.dropna().index)
        forecast_rf = rolling_rf_forecast(data_tabular, country=col)
        forecast_rf_pooled = pooled_forecasts_rf[col]
        forecast_lr_pooled = pooled_forecasts_lr[col]
        forecast_ffnn_pooled = pooled_forecasts_ffnn[col]

        rmse_rw = forecast_rmse(mom, forecast_rw)
        rmse_auto_arima = forecast_rmse(mom, forecast_auto_arima)
        rmse_rf = forecast_rmse(mom, forecast_rf)
        rmse_rf_pooled = forecast_rmse(mom, forecast_rf_pooled)
        rmse_lr_pooled = forecast_rmse(mom, forecast_lr_pooled)
        rmse_ffnn_pooled = forecast_rmse(mom, forecast_ffnn_pooled)

        results[col] = {
            "RW_RMSE": rmse_rw,
            "AUTO_ARIMA_RMSE": rmse_auto_arima,
            "RF_RMSE": rmse_rf,
            "RF_Pooled_RMSE": rmse_rf_pooled,
            "LR_Pooled_RMSE": rmse_lr_pooled,
            "FFNN_Pooled_RMSE": rmse_ffnn_pooled,
        }
        print(pd.DataFrame(results).round(4).T, flush=True)

    return pd.DataFrame(results)
