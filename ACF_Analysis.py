import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

dataframe = pd.read_csv("../mcp_6months.csv")
plot_acf(dataframe, lags=170)
plt.xlabel("Lag")
plt.ylabel("Auto correlation")
plt.title("Auto correlation Function (ACF)")
plt.grid(True)
plt.show()