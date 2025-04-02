import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from src.config import START_DATE, END_DATE


def get_fred_api():
    """
    Function to get the FRED API key from environment variables and create a Fred object.

    Returns:
        Fred: An instance of the Fred class with the API key.
    """
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")

    fred = Fred(api_key=FRED_API_KEY)
    return fred


def load_fred_data(
    targets: list, start_date: str = START_DATE, end_date=END_DATE
) -> pd.DataFrame:
    """
    Function to load data from FRED.

    Args:
        targets (list): List of FRED series IDs to load data for.

    Returns:
        pd.DataFrame: DataFrame containing the data in columns.
    """
    fred = get_fred_api()
    data = {}
    for target in targets:
        series = fred.get_series(target)
        data[target] = series
    data = pd.DataFrame(data)
    data = data.loc[start_date:end_date]

    return data
