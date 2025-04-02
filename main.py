from src.benchmark.compare_models import compare_models
from src.data.global_inflation_loader import (
    load_global_inflation_data,
    get_lagged_features_and_target,
)
from src.features.create_dummies import create_regional_dummies
from src.config import NUM_LAGS

# Load and prepare data
data = load_global_inflation_data("data/Inflation-data.xlsx")
data_tabular = get_lagged_features_and_target(df=data, n_lags=NUM_LAGS)
data_tabular = create_regional_dummies(data_tabular, "data/country_continent_map.csv")

# Run comparison
results = compare_models(data, data_tabular)

# Save the results to csv
results.to_csv("results/results.csv")
