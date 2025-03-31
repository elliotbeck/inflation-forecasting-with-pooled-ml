import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def forecast_rmse(actual: pd.Series, forecast: pd.Series) -> float:
    valid = actual.notna() & forecast.notna()
    return np.sqrt(mean_squared_error(actual[valid], forecast[valid]))
