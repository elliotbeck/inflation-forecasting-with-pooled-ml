import pandas as pd


def create_seasonal_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate seasonal dummy variables for each month from a datetime index.

    Parameters:
        index (pd.DatetimeIndex): Index of dates (usually from a time series).

    Returns:
        pd.DataFrame: DataFrame of month dummies with one column per month (11 total, drop-first encoding).
    """
    return pd.get_dummies(index.month, prefix="month", drop_first=True).astype(float)
