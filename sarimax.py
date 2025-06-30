import pandas as pd
import matplotlib.pyplot as plt
df_mcp = pd.read_csv("../mcp_6months.csv")
df_weather = pd.read_csv("../weather_6months.csv")

assert len(df_mcp) == len(df_weather), "Row count mismatch!"


df = pd.concat([df_mcp, df_weather], axis=1)


df = df.dropna()
y = df["MCP_5"]


exog = df[["Temperature", " Wind Speed(m/s)", "Solar Irradiance(W/m^2)"]]
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y,
    exog=exog,
    order=(1, 0, 2),              # ARIMA(p,d,q)
    seasonal_order=(1, 1, 1, 24), # SARIMA(P,D,Q,s)
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit()
print(result.summary())
df_july1_weather = pd.read_csv("../weather_july.csv")

exog_future = df_july1_weather[["Temperature", " Wind Speed(m/s)", "Solar Irradiance(W/m^2)"]]

forecast = result.forecast(steps=24, exog=exog_future)
# Placeholder
forecast.index = range(y.index[-1] + 1, y.index[-1] + 1 + len(forecast))


forecast = result.forecast(steps=24, exog=exog_future)
plt.plot(y.index[-100:], y[-100:], label="Actual")
plt.plot(forecast.index, forecast, label="Forecast", color='red')
print(forecast)
print(forecast.values.var())

forecast = result.forecast(steps=24, exog=exog_future)
plt.legend()
plt.title("MCP Forecast")
plt.show()