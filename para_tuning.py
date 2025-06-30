import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
df_mcp = pd.read_csv("../mcp_6months.csv")
df_weather = pd.read_csv("../weather_6months.csv")

assert len(df_mcp) == len(df_weather), "Row count mismatch!"

df = pd.concat([df_mcp, df_weather], axis=1)

df = df.dropna()
y = df["mcp"]

exog = df[["Temperature", " Wind Speed(m/s)", "Solar Irradiance(W/m^2)"]]

p = d = q = range(0, 3)
P = D = Q = range(0, 2)
s = 24

pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, [s]))

best_aic = np.inf
best_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(y, exog=exog, order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, seasonal_param)
                print(f"New best AIC: {best_aic} for order={param} seasonal_order={seasonal_param}")
        except:
            continue
f_mcp = pd.read_csv("../mcp_6months.csv")
df_weather = pd.read_csv("../weather_6months.csv")
