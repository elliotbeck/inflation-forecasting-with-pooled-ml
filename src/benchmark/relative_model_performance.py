import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("results/results.csv", index_col=0)

# # Drop the last four rows
# data = data.iloc[:-4]

# Count and print how many times RMSE_Pooled_RF is lower than RMSE_RF
count = (data["RF_Pooled_RMSE"] < data["RF_RMSE"]).sum()
print(
    f"RMSE of pooled RF is lower than RMSE of RF in {count} out of {data.shape[0]} cases."
)

# Compute relative RMSE
relative_rmse = data.div(data["RW_RMSE"], axis=0)

# Plot boxplots per column
plot_data = relative_rmse.drop(columns=["RW_RMSE"])
plt.figure(figsize=(10, 6))
plot_data.boxplot()
plt.title("Relative RMSE by Method")
plt.ylabel("Relative RMSE")
plt.xticks(rotation=45)
plt.savefig("results/relative_rmse_boxplot.pdf")

# Append a row with the column-wise means
relative_rmse.loc["mean"] = relative_rmse.mean()

# TODO: 1. Implement HRF for pooled RF
# TODO: 2. Think of how to show the relative performance of pooled RF vs RF
# TODO: 3. Think about which countries should be inluded, for now a lot of redundancies?
# TODO: 4. Think about other features to include in the RFs, for now only the lagged target.
# TODO: Regional dummies?
