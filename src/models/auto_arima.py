import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from config import MIN_TRAIN_OBSERVATIONS, NUM_LAGS, ROLLING_WINDOW_SIZE
from features.create_dummies import create_seasonal_dummies


def fit_ar_with_seasonal_dummies(
    train_series: pd.Series, max_p: int = NUM_LAGS
) -> float | None:
    """
    Fit AR(p) models with seasonal (monthly) dummies as exogenous regressors,
    and select the model with the lowest BIC. Forecast one step ahead.

    Parameters:
        train_series (pd.Series): Time series to fit, must have a DatetimeIndex.
        max_p (int): Maximum AR order to try (p from 1 to max_p).

    Returns:
        float or None: One-step-ahead forecast, or None if fitting fails.
    """
    if not isinstance(train_series.index, pd.DatetimeIndex):
        raise ValueError(
            "train_series must have a DatetimeIndex for seasonal dummies to work"
        )

    train_series = train_series.asfreq("MS")
    seasonal_dummies = create_seasonal_dummies(train_series.index)
    seasonal_dummies.index = train_series.index
    best_bic = np.inf
    best_model = None

    # TODO: Uncomment this block to enable model selection
    # for p in range(1, max_p + 1):
    #     try:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", ConvergenceWarning)
    #             model = ARIMA(
    #                 train_series, order=(p, 0, 0), exog=seasonal_dummies
    #             ).fit()
    #         if model.bic < best_bic:
    #             best_bic = model.bic
    #             best_model = model
    #     except Exception:
    #         continue

    # if best_model is None:
    #     return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        best_model = ARIMA(
            train_series, order=(max_p, 0, 0), exog=seasonal_dummies
        ).fit()

    # Create seasonal dummies for the next period (t+1)
    last_date = train_series.index[-1]
    next_month = last_date + pd.DateOffset(months=1)
    next_month_dummies = create_seasonal_dummies(pd.DatetimeIndex([next_month]))

    # Align dummy columns to training set (to avoid missing ones)
    next_exog = next_month_dummies.reindex(
        columns=seasonal_dummies.columns, fill_value=0
    )

    # Forecast next value
    return best_model.forecast(steps=1, exog=next_exog).iloc[0]


def forecast_arima_step(
    t: int,
    series: pd.Series,
    num_lags: int,
    rolling_window: int,
    min_train_observations: int,
) -> tuple[int, float | None]:
    """
    Perform a one-step-ahead AR(p) forecast with seasonal dummies for a given time point t.

    This function:
    - Extracts a rolling window ending at time t from the input series
    - Ensures the window has sufficient non-missing data
    - Fits AR(p) models (for p in 1 to `num_lags`) with 12 seasonal dummies as exogenous regressors
    - Selects the model with the lowest BIC
    - Returns the one-step-ahead forecast

    Parameters:
        t (int): Time index (used for windowing, not datetime).
        series (pd.Series): The time series data with a DatetimeIndex.
        num_lags (int): Maximum AR order (p) to consider for model selection.
        rolling_window (int): Size of the rolling training window (in time steps).
        min_train_observations (int): Minimum required non-missing observations to attempt model fitting.

    Returns:
        tuple[int, float | None]: The time index `t`, and the forecast value.
                                  If not enough data is available or fitting fails, returns `None` for the prediction.
    """
    train_series = series.iloc[t - rolling_window : t].dropna()

    if train_series.shape[0] < (min_train_observations + num_lags):
        return t, None

    try:
        pred = fit_ar_with_seasonal_dummies(train_series, max_p=num_lags)
        return t, pred
    except Exception:
        return t, None


def rolling_auto_arima_forecast(
    series: pd.Series,
    num_lags: int = NUM_LAGS,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    rolling_window: int = ROLLING_WINDOW_SIZE,
    n_jobs: int = 30,
) -> pd.Series:
    """
    Parallel rolling one-step-ahead ARIMA forecast using auto_arima.

    Parameters:
        series: pd.Series of inflation
        num_lags: AR order for ARIMA (AR(num_lags))
        min_train_observations: Minimum required observations to fit model
        rolling_window: Size of the rolling training window (in time steps)
        n_jobs: Number of parallel workers (-1 = all cores)

    Returns:
        forecast: pd.Series with predictions aligned to input index
    """
    forecast = pd.Series(index=series.index, dtype=float)

    # Only forecast up to the last valid observation in the series
    last_valid_index = series.last_valid_index()
    end_t = series.index.get_loc(last_valid_index)

    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast_arima_step)(
            t,
            series,
            num_lags,
            rolling_window,
            min_train_observations,
        )
        for t in range(rolling_window + num_lags + 1, end_t + 1)
    )

    for t, pred in results:
        if pred is not None:
            forecast.at[series.index[t]] = pred

    return forecast
