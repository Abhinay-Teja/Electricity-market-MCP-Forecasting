import pandas as pd
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
mcp_data = pd.read_csv("mcp_data.csv")
weather_data = pd.read_csv("weatherdata.csv")
from statsmodels.graphics.tsaplots import plot_ccf,plot_acf

mcp = (mcp_data["mcp"] - mcp_data["mcp"].mean()) / mcp_data["mcp"].std()
temp = (weather_data["Temperature"] - weather_data["Temperature"].mean()) / weather_data["Temperature"].std()
wind = (weather_data[" Wind Speed(m/s)"] - weather_data[" Wind Speed(m/s)"].mean()) / weather_data[" Wind Speed(m/s)"].std()
solar = (weather_data["Solar Irradiance(W/m^2)"] - weather_data["Solar Irradiance(W/m^2)"].mean()) / weather_data["Solar Irradiance(W/m^2)"].std()
# Plot CCF between MCP and Temp
plot_ccf(temp, mcp, lags=100)
plt.title("CCF: Temp vs MCP")
plt.show()

# Repeat for wind and solar
plot_ccf(wind, mcp, lags=100)
plt.title("CCF: Wind vs MCP")
plt.show()

plot_ccf(solar, mcp, lags=100)
plt.title("CCF: Solar vs MCP")
plt.show()