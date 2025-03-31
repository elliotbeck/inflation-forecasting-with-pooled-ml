import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_mom_inflation(series: pd.Series) -> pd.Series:
    return 100 * (np.log(series) - np.log(series.shift(1)))


def forecast_rmse(actual: pd.Series, forecast: pd.Series) -> float:
    valid = actual.notna() & forecast.notna()
    return np.sqrt(mean_squared_error(actual[valid], forecast[valid]))


def make_lagged_features(series: pd.Series, lags: int = 4) -> pd.DataFrame:
    return pd.DataFrame({f"lag_{i}": series.shift(i) for i in range(1, lags + 1)})


def winsorize_series(
    series: pd.Series, lower_quantile=0.005, upper_quantile=0.995
) -> pd.Series:
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)
