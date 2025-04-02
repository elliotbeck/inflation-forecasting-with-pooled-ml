import pandas as pd


def has_internal_nans(series: pd.Series) -> bool:
    """
    Check if a series has internal NaNs (i.e., NaNs that are not at the
    beginning or end of the series).
    Parameters:
    series (pd.Series): The series to check for internal NaNs.
    Returns:
    bool: True if the series has internal NaNs, False otherwise.
    """
    # Get first and last non-NaN index
    notna = series.notna()
    if notna.any():
        start, end = notna.idxmax(), notna[::-1].idxmax()
        # Slice between first and last non-NaN, and check if there are any NaNs
        return series.loc[start:end].isna().any()
    return False  # all NaNs is fine
