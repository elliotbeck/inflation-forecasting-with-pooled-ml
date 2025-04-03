from pyexpat import model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
data = pd.read_csv("results/results.csv", index_col=0).T

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
plt.savefig("results/relative_rmse_boxplot.pdf")
plt.close()

# Get results by continent
continent_mapping = pd.read_csv("data/country_continent_map.csv")
plot_data = plot_data.join(continent_mapping.set_index("Country"), on="Country")

melted = plot_data.melt(
    id_vars=["Country", "Continent"],
    value_vars=[
        "RF_RMSE",
        "RF_Pooled_RMSE",
        "LR_Pooled_RMSE",
        "FFNN_Pooled_RMSE",
    ],
    var_name="Model",
    value_name="Relative_RMSE",
)

continents = melted["Continent"].unique()
n_continents = len(continents)

fig, axes = plt.subplots(
    nrows=(n_continents + 2) // 3, ncols=3, figsize=(15, 5 * ((n_continents + 2) // 3))
)
axes = axes.flatten()

for i, continent in enumerate(continents):
    ax = axes[i]
    subset = melted[melted["Continent"] == continent]

    subset.boxplot(column="Relative_RMSE", by="Model", ax=ax, grid=False)
    ax.set_title(continent)
    ax.set_xlabel("Model")
    ax.set_ylabel("Relative RMSE")
    ax.tick_params(axis="x", rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("")
plt.tight_layout()
plt.savefig("results/relative_rmse_boxplot_by_continent.pdf")
plt.close()

# Append a row with the column-wise means
relative_rmse.loc["mean"] = relative_rmse.mean()
relative_rmse.loc["median"] = relative_rmse.median()
