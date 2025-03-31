import pandas as pd
from joblib import Parallel, delayed
from pmdarima import auto_arima

from src.config import MIN_TRAIN_OBSERVATIONS, NUM_LAGS, ROLLING_WINDOW_SIZE


def forecast_arima_step(
    t: int,
    series: pd.Series,
    num_lags: int,
    rolling_window: int,
    min_train_observations: int,
) -> tuple[int, float | None]:
    train_series = series.iloc[t - rolling_window : t].dropna()

    if train_series.shape[0] < (min_train_observations + num_lags):
        return t, None

    try:
        model = auto_arima(
            train_series,
            start_p=num_lags,
            max_p=num_lags,
            d=0,
            start_q=0,
            max_q=0,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=1,
            n_jobs=1,
        )
        pred = model.predict(n_periods=1).iloc[0]
        return t, pred
    except Exception:
        return t, None


def rolling_auto_arima_forecast(
    yoy_series: pd.Series,
    num_lags: int = NUM_LAGS,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    n_jobs: int = 30,
) -> pd.Series:
    """
    Parallel rolling one-step-ahead ARIMA forecast using auto_arima.

    Parameters:
        yoy_series: pd.Series of YoY inflation
        num_lags: AR order for ARIMA (AR(num_lags))
        min_train_observations: Minimum required observations to fit model
        n_jobs: Number of parallel workers (-1 = all cores)

    Returns:
        forecast: pd.Series with predictions aligned to input index
    """
    forecast = pd.Series(index=yoy_series.index, dtype=float)

    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast_arima_step)(
            t,
            yoy_series,
            num_lags,
            ROLLING_WINDOW_SIZE,
            min_train_observations,
        )
        for t in range(ROLLING_WINDOW_SIZE, len(yoy_series))
    )

    for t, pred in results:
        if pred is not None:
            forecast.iloc[t] = pred

    return forecast
