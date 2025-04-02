import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

from config import MIN_TRAIN_OBSERVATIONS, ROLLING_WINDOW_SIZE


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _forecast_step_multitarget_ffnn(
    t: int,
    panel_df: pd.DataFrame,
    all_dates: pd.Index,
    feature_cols: list[str],
    rolling_window: int,
    min_train_observations: int,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    max_epochs: int = 500,
    patience: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
) -> tuple[pd.Timestamp, dict[str, float]]:
    """
    Forecasts one-step-ahead targets for all countries at a given time step using a pooled feedforward neural network.

    Parameters:
        t (int): Current time step index.
        panel_df (pd.DataFrame): Long-format panel with columns ['Date', 'Country', 'Target', features...].
        all_dates (pd.Index): All sorted time indices.
        feature_cols (list[str]): List of feature column names.
        rolling_window (int): Number of past periods to use for training.
        min_train_observations (int): Minimum samples required for a country to be included.
        hidden_dim (int): Hidden layer size in the feedforward net.
        dropout (float): Dropout probability.
        max_epochs (int): Max training epochs.
        patience (int): Patience for early stopping.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        device (str): "cpu" or "cuda" if using GPU.

    Returns:
        tuple[pd.Timestamp, dict[str, float]]: Date and country-wise predictions.
    """
    current_date = all_dates[t]
    train_window = all_dates[t - rolling_window : t]

    train_df = panel_df[panel_df["Date"].isin(train_window)].dropna(
        subset=feature_cols + ["Target"]
    )
    train_df = train_df.sort_values(["Date", "Country"])

    test_df = panel_df[panel_df["Date"] == current_date].dropna(subset=feature_cols)

    if train_df.empty or test_df.empty:
        return current_date, {}

    country_counts = train_df["Country"].value_counts()
    eligible_countries = country_counts[country_counts >= min_train_observations].index
    test_df = test_df[test_df["Country"].isin(eligible_countries)]

    if test_df.empty:
        return current_date, {}

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["Target"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Chronological 80/20 split
    split_idx = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Move to device
    X_tr = torch.tensor(X_tr).to(device)
    y_tr = torch.tensor(y_tr).to(device)
    X_val = torch.tensor(X_val).to(device)
    y_val = torch.tensor(y_val).to(device)
    X_test_tensor = torch.tensor(X_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = FeedforwardNet(
        input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopping
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = [loss_fn(model(xb), yb).item() for xb, yb in val_loader]
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # Early stop

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Predict on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()

    return current_date, dict(zip(test_df["Country"], y_pred))


def rolling_ffnn_pooled_forecast(
    data: pd.DataFrame,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    n_jobs: int = 30,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Forecasts one-step-ahead YoY inflation for all countries using a pooled feedforward neural network,
    training a single model at each time step and predicting simultaneously for all eligible countries.

    Parameters:
        data (pd.DataFrame): Long-format panel with columns ['Date', 'Country', 'Target', features...].
        min_train_observations (int): Minimum number of training samples required per country to be included in prediction.
        n_jobs (int): Number of parallel jobs to run.
        hidden_dim (int): Number of hidden units in the FFNN.
        dropout (float): Dropout rate.
        max_epochs (int): Max number of training epochs.
        patience (int): Early stopping patience.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        device (str): "cpu" or "cuda".

    Returns:
        pd.DataFrame: Forecasted values indexed by Date and columns as Country names.
                      Missing values will remain NaN if the country couldn't be predicted.
    """
    forecast = pd.DataFrame(index=sorted(data["Date"].unique()), dtype=float)
    all_dates = data["Date"].sort_values().unique()
    feature_cols = data.drop(columns=["Date", "Country", "Target"]).columns.tolist()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_forecast_step_multitarget_ffnn)(
            t,
            data,
            all_dates,
            feature_cols,
            ROLLING_WINDOW_SIZE,
            min_train_observations,
            hidden_dim,
            dropout,
            max_epochs,
            patience,
            batch_size,
            lr,
            device,
        )
        for t in range(ROLLING_WINDOW_SIZE, len(all_dates))
    )

    for date, preds in results:
        for country, value in preds.items():
            forecast.loc[date, country] = value

    return forecast
