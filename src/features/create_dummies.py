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


def create_regional_dummies(
    data: pd.DataFrame, country_continent_mapping: str
) -> pd.DataFrame:
    """
    Adds continent-level dummy variables to the input DataFrame based on a country-to-continent mapping.

    Parameters:
    -----------
    data : pd.DataFrame
        The main DataFrame containing a 'Country' column.

    country_continent_mapping : str
        Path to a CSV file with columns 'Country' and 'Continent' used for mapping countries to continents.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with continent dummy variables (e.g., Continent_Africa, Continent_Europe, etc.)
        appended as additional columns, and the original 'Continent' column removed.
    """
    country_mapping = pd.read_csv(country_continent_mapping)

    return (
        data.merge(country_mapping, on="Country", how="left")
        .pipe(
            lambda df: df.assign(
                **pd.get_dummies(df["Continent"], prefix="Continent", drop_first=True)
            )
        )
        .drop(columns="Continent")
    )
