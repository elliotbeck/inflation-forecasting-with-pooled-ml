from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from config import MIN_TRAIN_OBSERVATIONS, ROLLING_WINDOW_SIZE


def _forecast_step_multitarget(
    t: int,
    panel_df: pd.DataFrame,
    all_dates: pd.Index,
    feature_cols: list[str],
    rolling_window: int,
    min_train_observations: int,
) -> tuple[pd.Timestamp, dict[str, float]]:
    """
    Internal helper function for forecasting all countries at a given time step using a pooled
    Linear Regression.

    This function builds one model using the full training window and predicts
    one-step-ahead values for all countries that have enough training history.

    Parameters:
        t (int): Current time step index.
        panel_df (pd.DataFrame): Long-format panel data with columns: ['Date', 'Country', 'YoY', lags, dummies].
        all_dates (pd.Index): Ordered list of all time indices.
        feature_cols (list[str]): List of feature column names (lags + country dummies).
        country_list (list[str]): List of all country column names in the original wide DataFrame.
        rolling_window (int): Number of months to include in the training window.
        min_train_observations (int): Minimum samples required for a country to be predicted.
        max_features (int): Number of features to consider at each split in the Linear Regression.

    Returns:
        tuple: (current_date, {country: prediction, ...}) with predicted values for all eligible countries.
    """
    current_date = all_dates[t]
    train_window = all_dates[t - rolling_window : t]

    train_df = panel_df[panel_df["Date"].isin(train_window)].dropna(
        subset=feature_cols + ["Target"]
    )
    test_df = panel_df[panel_df["Date"] == current_date].dropna(subset=feature_cols)

    if train_df.empty or test_df.empty:
        return current_date, {}

    # Only include countries with enough training data
    country_counts = train_df["Country"].value_counts()
    eligible_countries = country_counts[country_counts >= min_train_observations].index
    test_df = test_df[test_df["Country"].isin(eligible_countries)]

    if test_df.empty:
        return current_date, {}

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]
    X_test = test_df[feature_cols]

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        # alphas=np.logspace(-4, 1, 20),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        max_iter=10000,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return current_date, dict(zip(test_df["Country"], y_pred))


def rolling_elastic_net_pooled_forecast(
    data: pd.DataFrame,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    max_features: Optional[int] = None,
    n_jobs: int = 30,
) -> pd.DataFrame:
    """
    Forecasts one-step-ahead YoY inflation for all countries using a pooled Linear Regression,
    training a single model at each time step and predicting simultaneously for all eligible countries.

    Parameters:
        data (pd.DataFrame): Wide-format DataFrame with time as index and columns as countries (inflation).
        min_train_observations (int): Minimum number of training samples required per country to be included in prediction.
        max_features (Optional[int]): Number of features to consider at each split in the RF.
                                      If None, defaults to one-third of the total features.
        n_jobs (int): Number of parallel jobs to run (default: -1 = use all available cores).

    Returns:
        pd.DataFrame: Forecasted values with the same shape and index as input `data`.
                      Missing values will remain NaN if the country couldn't be predicted.
    """
    forecast = pd.DataFrame(index=sorted(data["Date"].unique()), dtype=float)

    # One-hot encode countries
    # country_dummies = pd.get_dummies(data["Country"], prefix="country")
    # panel_df = pd.concat([data, country_dummies], axis=1)
    feature_cols = data.drop(columns=["Date", "Country", "Target"]).columns.tolist()

    all_dates = data["Date"].sort_values().unique()

    if max_features is None:
        max_features = int(np.floor((len(feature_cols)) / 3))

    # Parallel time-step forecasting
    results = Parallel(n_jobs=n_jobs)(
        delayed(_forecast_step_multitarget)(
            t,
            data,
            all_dates,
            feature_cols,
            ROLLING_WINDOW_SIZE,
            min_train_observations,
        )
        for t in range(ROLLING_WINDOW_SIZE, len(all_dates))
    )

    # Fill in forecast matrix
    for date, preds in results:
        for country, value in preds.items():
            forecast.loc[date, country] = value

    return forecast
