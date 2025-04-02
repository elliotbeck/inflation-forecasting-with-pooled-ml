import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("results/results.csv", index_col=0)

# Count and print how many times RMSE_Pooled_RF is lower than RMSE_RF
count = (data["RF_Pooled_RMSE"] < data["RF_RMSE"]).sum()
print(
    f"RMSE of pooled RF is lower than RMSE of RF in {count} out of {data.shape[0]} cases."
)

# Compute relative RMSE
relative_rmse = data.div(data["RW_RMSE"], axis=0)

# Plot boxplots per column
plot_data = relative_rmse.drop(columns=["RW_RMSE", "AUTO_ARIMA_RMSE"])
plt.figure(figsize=(10, 6))
plot_data.boxplot()
plt.title("Relative RMSE by Method")
plt.ylabel("Relative RMSE")
plt.xticks(rotation=45)
plt.savefig("results/relative_rmse_boxplot.pdf")

# Append a row with the column-wise means
relative_rmse.loc["mean"] = relative_rmse.mean()
relative_rmse.loc["median"] = relative_rmse.median()
