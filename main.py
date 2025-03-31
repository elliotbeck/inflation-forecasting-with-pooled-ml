from src.benchmark.compare_models import compare_models  # noqa
from src.data.fred_loader import load_fred_data  # noqa
import yaml

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data = load_fred_data(config["countries"])
results = compare_models(data)

results.to_csv("results/results.csv")
