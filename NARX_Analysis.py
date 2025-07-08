import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load CSVs
df_mcp = pd.read_csv("ACTUAL_MCP_6MONTHS.csv")
df_weather = pd.read_csv("weather_6months.csv")
df_weather_july = pd.read_csv("weather_july.csv")
df_mcp_july = pd.read_csv("mcp_july.csv")

mcp_arr = np.array(df_mcp).flatten()
weather_arr = np.array(df_weather)
weather_july_arr = np.array(df_weather_july)
mcp_july_arr = np.array(df_mcp_july).flatten()

# Scaling
mcp_scaler = MinMaxScaler()
mcp_scaled = mcp_scaler.fit_transform(mcp_arr.reshape(-1, 1)).flatten()

weather_scaler = MinMaxScaler()
weather_scaled = weather_scaler.fit_transform(weather_arr)
weather_july_scaled = weather_scaler.transform(weather_july_arr)

# Add time features
def add_time_features(arr, offset=0):
    n = arr.shape[0]
    hour = (np.arange(offset, offset + n) % 24) / 23.0
    dayofweek = ((np.arange(offset, offset + n) // 24) % 7) / 6.0
    return np.concatenate([arr, hour.reshape(-1, 1), dayofweek.reshape(-1, 1)], axis=1)

weather_full = add_time_features(weather_scaled)
weather_july_full = add_time_features(weather_july_scaled, offset=len(mcp_arr))

# Parameters
lags_list = [24, 48, 168]
epochs = 100
hidden_size = 64
batch_size = 32
learning_rate = 0.0005

# Dataset
class NARXDataset(Dataset):
    def __init__(self, y, x_exog, lags):
        self.X = []
        self.y = []
        max_lag = max(lags)
        for t in range(max_lag, len(y)):
            y_lags = np.array([y[t - lag] for lag in lags], dtype=np.float32)
            x_now = x_exog[t]
            self.X.append(np.concatenate([y_lags, x_now]))
            self.y.append(y[t])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model
class NARXModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NARXModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# Train
def train_model(model, dataset, num_epochs, batch_size, lr):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(dataset):.4f}")

# Forecast
def recursive_forecast(model, y_history, x_future, lags, steps):
    model.eval()
    preds = []
    y_history = list(y_history)
    for t in range(steps):
        y_lags = np.array([y_history[-lag] for lag in lags], dtype=np.float32)
        x_input = x_future[t]
        model_input = torch.tensor(np.concatenate([y_lags, x_input]), dtype=torch.float32).unsqueeze(0)
        next_y = model(model_input).item()
        preds.append(next_y)
        y_history.append(next_y)
    return np.array(preds)

# Train the model
train_dataset = NARXDataset(mcp_scaled, weather_full, lags_list)
input_size = train_dataset[0][0].shape[0]
model = NARXModel(input_size, hidden_size)
train_model(model, train_dataset, epochs, batch_size, learning_rate)

# Predict July
y_history = mcp_scaled  # most recent training MCP
steps = len(weather_july_full)
pred_scaled = recursive_forecast(model, y_history, weather_july_full, lags_list, steps)
pred = mcp_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# Evaluate
MAPE = np.mean(np.abs((mcp_july_arr - pred) / np.clip(mcp_july_arr, 1e-2, None))) * 100
print(f"\nFinal MAPE on July: {MAPE:.2f}%")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(mcp_july_arr, label="Actual July MCP", linewidth=2, marker='o')
plt.plot(pred, label="Predicted MCP", linewidth=2, marker='x')
plt.title("Recursive Multi-step MCP Forecast (NARX)")
plt.xlabel("Hour")
plt.ylabel("MCP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
