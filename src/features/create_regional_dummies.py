import pandas as pd


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


if __name__ == "__main__":
    from src.data.global_inflation_loader import (
        load_global_inflation_data,
        get_lagged_features_and_target,
    )

    data = load_global_inflation_data("data/Inflation-data.xlsx")
    data = get_lagged_features_and_target(data, n_lags=12)
    data = create_regional_dummies(
        data,
        "data/country_continent_map.csv",
    )
    data.head()
