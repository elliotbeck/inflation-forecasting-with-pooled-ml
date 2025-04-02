import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from config import MIN_TRAIN_OBSERVATIONS, ROLLING_WINDOW_SIZE


def forecast_rf_step(
    t: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    rolling_window: int,
    min_train_observations: int,
) -> tuple[int, float | None]:
    X_train = X_full.iloc[max(t - rolling_window, 0) : t].dropna()
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
    data: pd.DataFrame,
    country: str,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    n_jobs: int = 30,
) -> pd.Series:
    """
    Parallel rolling one-step-ahead forecast using Random Forest (non-pooled).

    Parameters:
        data: wide-format DataFrame of price-level series
        country: country to forecast
        min_train_observations: minimum number of training observations required
        n_jobs: number of parallel jobs (default: all cores)

    Returns:
        forecast: pd.Series with one-step-ahead forecasts aligned with input index
    """
    forecast = pd.Series(index=sorted(data["Date"].unique()), dtype=float)

    data = data[data["Country"] == country]
    data = data.sort_values("Date")

    X_full = data.drop(columns=["Date", "Country", "Target"]).set_index(data["Date"])
    y_full = data["Target"]
    y_full.index = data["Date"]

    # Only forecast up to the last valid observation in the series
    try:
        start_t = X_full.index.get_loc(forecast.index[ROLLING_WINDOW_SIZE])
    except KeyError:
        start_t = 0
    end_t = y_full.index.get_loc(y_full.last_valid_index())

    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast_rf_step)(
            t,
            X_full,
            y_full,
            rolling_window=ROLLING_WINDOW_SIZE,
            min_train_observations=min_train_observations,
        )
        for t in range(start_t, end_t + 1)
    )

    for t, pred in results:
        if pred is not None:
            forecast.at[y_full.index[t]] = pred

    return forecast
