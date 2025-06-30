import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("mcp_data.csv")
series = data['mcp']

result = seasonal_decompose(series, model='additive', period=24)

deseasonalized = series - result.seasonal
detrended = series - result.trend

residual_1 = series - result.trend - result.seasonal
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(series, label='Original Series')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(deseasonalized, label='Deseasonalized', color='green')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(residual_1, label='Detrended', color='orange')
plt.legend()

plt.tight_layout()
plt.show()