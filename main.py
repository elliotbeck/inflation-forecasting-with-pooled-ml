from src.benchmark.compare_models import compare_models
from src.data.global_inflation_loader import (
    load_global_inflation_data,
    get_lagged_features_and_target,
)
from src.config import NUM_LAGS

data = load_global_inflation_data("data/Inflation-data.xlsx")
data_tabular = get_lagged_features_and_target(df=data, n_lags=NUM_LAGS)

results = compare_models(data, data_tabular)

results.to_csv("results/results.csv")
