import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ------------------ Data Loading ------------------
df_mcp = pd.read_csv("ACTUAL_MCP_6MONTHS.csv")
df_weather = pd.read_csv("weather_6months.csv")
df_weather_july = pd.read_csv("weather_july.csv")
df_mcp_july = pd.read_csv("mcp_july.csv")

mcp_arr = np.array(df_mcp).flatten()
weather_arr = np.array(df_weather)
weather_july_arr = np.array(df_weather_july)
mcp_july_arr = np.array(df_mcp_july).flatten()

# ------------------ Scaling ------------------
mcp_scaler = MinMaxScaler()
mcp_scaled = mcp_scaler.fit_transform(mcp_arr.reshape(-1, 1)).flatten()

weather_scaler = MinMaxScaler()
weather_scaled = weather_scaler.fit_transform(weather_arr)
weather_july_scaled = weather_scaler.transform(weather_july_arr)

# ------------------ Add Time Features ------------------
def add_time_features(arr, offset=0):
    n = arr.shape[0]
    hour = (np.arange(offset, offset + n) % 24) / 23.0
    dayofweek_raw = ((np.arange(offset, offset + n) // 24) % 7)
    dayofweek = dayofweek_raw / 6.0
    holiday = (dayofweek_raw == 6).astype(np.float32).reshape(-1, 1)  # Sunday = holiday
    return np.concatenate([arr, hour.reshape(-1, 1), dayofweek.reshape(-1, 1), holiday], axis=1)

weather_full = add_time_features(weather_scaled)
weather_july_full = add_time_features(weather_july_scaled, offset=len(mcp_arr))

# ------------------ Parameters ------------------
lags_list = [24, 48, 168]
epochs = 75
hidden_size = 32
batch_size = 16
learning_rate = 0.0005
num_stacked_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Dataset Class ------------------
class LSTMDataset(Dataset):
    def __init__(self, y, x_exog, lags):
        self.X = []
        self.y = []
        max_lag = max(lags)
        for t in range(max_lag, len(y)):
            y_lags = np.array([y[t - lag] for lag in lags], dtype=np.float32)
            x_now = x_exog[t]
            features = np.concatenate([y_lags, x_now])
            self.X.append(features)
            self.y.append(y[t])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(1)  # (N, 1, input_size)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------ LSTM Model ------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last time step
        return out

# ------------------ Train Function ------------------
def train_model(model, dataset, num_epochs, batch_size, lr):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(dataset):.4f}")

# ------------------ Forecast ------------------
def recursive_forecast(model, y_history, x_future, lags, steps):
    model.eval()
    preds = []
    y_history = list(y_history)
    for t in range(steps):
        y_lags = np.array([y_history[-lag] for lag in lags], dtype=np.float32)
        x_input = x_future[t]
        model_input = np.concatenate([y_lags, x_input], axis=0).reshape(1, 1, -1)
        model_input = torch.tensor(model_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            next_y = model(model_input).item()
        preds.append(next_y)
        y_history.append(next_y)
    return np.array(preds)

# ------------------ Run Training and Prediction ------------------
train_dataset = LSTMDataset(mcp_scaled, weather_full, lags_list)
input_size = train_dataset[0][0].shape[-1]
model = LSTMModel(input_size, hidden_size, num_stacked_layers)
train_model(model, train_dataset, epochs, batch_size, learning_rate)

steps = len(weather_july_full)
pred_scaled = recursive_forecast(model, mcp_scaled, weather_july_full, lags_list, steps)
pred = mcp_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# ------------------ Evaluate ------------------
MAPE = np.mean(np.abs((mcp_july_arr - pred) / np.clip(mcp_july_arr, 1e-2, None))) * 100
print(f"\nFinal MAPE on July: {MAPE:.2f}%")

# ------------------ Plot ------------------
plt.figure(figsize=(12, 5))
plt.plot(mcp_july_arr, label="Actual July MCP", linewidth=2, marker='o')
plt.plot(pred, label="Predicted MCP", linewidth=2, marker='x')
plt.title("Recursive Multi-step MCP Forecast (LSTM)")
plt.xlabel("Hour")
plt.ylabel("MCP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()