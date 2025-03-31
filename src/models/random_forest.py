import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from src.config import MIN_TRAIN_OBSERVATIONS, NUM_LAGS, ROLLING_WINDOW_SIZE
from src.features.transforms import make_lagged_features


def forecast_rf_step(
    t: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    rolling_window: int,
    min_train_observations: int,
) -> tuple[int, float | None]:
    X_train = X_full.iloc[t - rolling_window : t].dropna()
    y_train = y_full.loc[X_train.index]
    X_test = X_full.iloc[[t]]

    if X_train.shape[0] < min_train_observations or X_test.isnull().values.any():
        return t, None

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
        max_features=int(np.floor(len(X_train.columns) / 3)),
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)[0]
    return t, prediction


def rolling_rf_forecast(
    yoy_series: pd.Series,
    lags: int = NUM_LAGS,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    n_jobs: int = 30,
) -> pd.Series:
    """
    Parallel rolling one-step-ahead forecast using Random Forest (non-pooled).

    Parameters:
        yoy_series: pd.Series of YoY inflation (indexed by date)
        lags: number of lagged features
        min_train_observations: minimum number of training observations required
        n_jobs: number of parallel jobs (default: all cores)

    Returns:
        forecast: pd.Series with one-step-ahead forecasts aligned with input index
    """
    forecast = pd.Series(index=yoy_series.index, dtype=float)
    X_full = make_lagged_features(yoy_series, lags)
    y_full = yoy_series

    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast_rf_step)(
            t, X_full, y_full, ROLLING_WINDOW_SIZE, min_train_observations
        )
        for t in range(ROLLING_WINDOW_SIZE, len(yoy_series))
    )

    for t, pred in results:
        if pred is not None:
            forecast.iloc[t] = pred

    return forecast
