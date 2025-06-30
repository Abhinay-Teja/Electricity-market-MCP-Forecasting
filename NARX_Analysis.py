import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df_mcp = pd.read_csv("../mcp_6months.csv")
df_weather = pd.read_csv("../weather_6months.csv")
df_weather_july = pd.read_csv("../weather_july.csv")
df_mcp_july = pd.read_csv("../mcp_july.csv")
mcp_arr = np.array(df_mcp)
weather_arr = np.array(df_weather)
weather_july_arr = np.array(df_weather_july)
mcp_july_arr = np.array(df_mcp_july)

mcp_scaler = StandardScaler()
mcp_scaled = mcp_scaler.fit_transform(mcp_arr)

weather_scaler = StandardScaler()
weather_6months_scaled = weather_scaler.fit_transform(weather_arr)
weather_july_scaled = weather_scaler.transform(weather_july_arr)

lags_list = [24,48,168]
epoches = 100
hidde_size = 64
batch_size = 64
learning_rate = 0.0005

class NARXDataset(Dataset):
    def __init__(self,y,x_exog,lags):
        self.X= []
        self.y= []
        max_lag = max(lags)

        for t in range(max_lag,len(y)):
            y_lags = np.array([y[t - lag] for lag in lags], dtype=np.float32).flatten()
            x_now = x_exog[t]
            self.X.append(np.concatenate([y_lags,x_now]))
            self.y.append(y[t])
        self.X = torch.tensor(np.array(self.X),dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y),dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]


class NARXModel(nn.Module):
    def __init__(self, input_size,hidden_size,):
        super(NARXModel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)


def train_model(model,dataset,no_of_epoches,batch_size,lr):
    loader = DataLoader(dataset,batch_size,shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_fn = nn.MSELoss()

    for epoch in range(no_of_epoches):
        total_loss = 0
        for X_batch , y_batch in loader:
            y_predicted = model(X_batch)
            loss = loss_fn(y_predicted,y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*X_batch.size(0)
            optimizer.zero_grad()
        average_loss = total_loss/len(dataset)
        print(f"Epoch {epoch+1}/{no_of_epoches} Loss: {average_loss:.4f}")

def recursive_forecast(model,y_latest,x_future,lags,steps):
    model.eval()
    prediction = []
    y_latest = list(y_latest.flatten())

    for t in range(steps):
        y_lagged = np.array([y_latest[-lag] for lag in lags], dtype=np.float32)
        x_now = x_future[t]
        x_model_input = torch.tensor(np.concatenate([y_lagged,x_now]),dtype = torch.float32).unsqueeze(0)#since input for model is of shape [batch_size,input_size]
        y_next = model(x_model_input).item()
        prediction.append(y_next)
        y_latest.append(y_next)
    return np.array(prediction)

train_data = NARXDataset(mcp_scaled,weather_6months_scaled,lags_list)
input_size = train_data[0][0].shape[0]

model = NARXModel(input_size = input_size , hidden_size = hidde_size)
train_model(model, train_data, epoches,batch_size = batch_size,lr = learning_rate)

x_future = weather_july_scaled  # future weather
steps = len(x_future)

prediction_scaled = recursive_forecast(model, mcp_scaled, x_future, lags_list , steps)
prediction = mcp_scaler.inverse_transform(prediction_scaled.reshape(-1,1))

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(mcp_july_arr,label = "Actual July MCP", linewidth = 2, marker = "o")
plt.plot( prediction, label="Predicted MCP", linewidth=2, marker = "x")
plt.title("Recursive Multi-step Forecast using NARX")
plt.xlabel("Time step")
plt.ylabel("MCP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

