import pandas as pd


def rolling_rw_forecast(
    series: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Rolling one-step-ahead Random Walk forecast.
    Forecast at time t is simply the value at t-1.
    Only forecasts are returned if at least `min_train_observations` are available before t.

    Parameters:
        series: pd.Series of inflation
        index: pd.DatetimeIndex of forecast dates

    Returns:
        pd.Series of forecasts aligned with input index
    """
    forecast = series.shift(1)
    forecast = forecast[forecast.index.isin(index)]
    return forecast
