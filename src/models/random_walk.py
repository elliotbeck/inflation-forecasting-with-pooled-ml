import pandas as pd
from src.config import MIN_TRAIN_OBSERVATIONS, NUM_LAGS


def rolling_rw_forecast(
    yoy_series: pd.Series,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    num_lags: int = NUM_LAGS,
) -> pd.Series:
    """
    Rolling one-step-ahead Random Walk forecast.
    Forecast at time t is simply the value at t-1.
    Only forecasts are returned if at least `min_train_observations` are available before t.

    Parameters:
        yoy_series: pd.Series of YoY inflation
        min_train_observations: minimum number of observations required before making forecasts
        num_lags: number of lags to consider (not used in RW, but kept for consistency)

    Returns:
        pd.Series of forecasts aligned with input index
    """
    forecast = yoy_series.shift(1)

    valid_obs_count = yoy_series.notna().cumsum()
    min_required_obs = min_train_observations + num_lags + 1
    forecast[valid_obs_count < min_required_obs] = pd.NA
    return forecast
