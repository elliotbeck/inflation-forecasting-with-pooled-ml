import pandas as pd

from utils.utils import has_internal_nans
from features.transforms import compute_mom_inflation, winsorize_series


def load_global_inflation_data(file_path: str) -> pd.DataFrame:
    """
    Load global inflation data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing global inflation
    data.
    Returns:
    pd.DataFrame: A DataFrame containing the global inflation data.
    """
    # Read the excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name="hcpi_m")

    # Drop unnecessary columns and last row
    df = df.drop(
        columns=[
            "Country Code",
            "IMF Country Code",
            "Indicator Type",
            "Series Name",
            "Data source",
            "Note",
        ]
    )
    df = df.drop(df.index[-1])

    # First, set 'Country' as the index so it becomes column headers after transpose
    df = df.set_index("Country").transpose()
    df.index = pd.to_datetime(df.index, format="%Y%m")

    # Keep only columns that do NOT have internal NaNs
    df = df.loc[:, ~df.apply(has_internal_nans)]

    # Convert the entire DataFrame to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def get_lagged_features_and_target(df: pd.DataFrame, n_lags: int = 12) -> pd.DataFrame:
    """
    Create a DataFrame with lagged features, seasonal dummies, and target values
    for each country and time step.

    Each row contains:
    - Country
    - Date (target date)
    - Target (value at that date)
    - Lag_1 to Lag_n (previous values)
    - month_2 to month_12 (11 dummy variables for seasonality)

    Parameters:
        df (pd.DataFrame): DataFrame of price levels, with datetime index and country columns.
        n_lags (int): Number of lags to include as features.

    Returns:
        pd.DataFrame: Dataset for supervised learning (one row per country-date).
    """
    df = df.apply(compute_mom_inflation)
    df = df.apply(winsorize_series)

    # Map countries to continents
    continent_map = (
        pd.read_csv("data/country_continent_map.csv")
        .set_index("Country")["Continent"]
        .to_dict()
    )

    # Create a new DataFrame where each column is a continent
    df_continent = df.copy()
    df_continent.columns = [continent_map.get(c) for c in df.columns]
    continent_means = df_continent.T.groupby(level=0).mean().T

    lagged_data = []

    for country in df.columns:
        country_series = df[country]

        for t in range(n_lags, len(country_series)):
            lags = country_series.iloc[t - n_lags : t].values[::-1]
            target = country_series.iloc[t]
            date = country_series.index[t]
            continent = continent_map.get(country)

            prev_date = country_series.index[t - 1]
            continent_mean_lag1 = continent_means.at[prev_date, continent]

            if not pd.isna(lags).any() and not pd.isna(target):
                # Get month from date and create dummy columns (1-12 → 2-12 as dummies)
                month = date.month
                seasonal_dummies = {
                    f"month_{m}": 1 if month == m else 0 for m in range(2, 13)
                }

                row = {
                    "Country": country,
                    "Date": date,
                    "Target": target,
                    **{f"Lag_{i + 1}": lags[i] for i in range(n_lags)},
                    **seasonal_dummies,
                    "Regional_Inflation": continent_mean_lag1,
                }
                lagged_data.append(row)

    return pd.DataFrame(lagged_data)


if __name__ == "__main__":
    # Example usage
    file_path = "data/Inflation-data.xlsx"
    df = load_global_inflation_data(file_path)
    lagged_df = get_lagged_features_and_target(df, n_lags=12)
    print(lagged_df.tail())
