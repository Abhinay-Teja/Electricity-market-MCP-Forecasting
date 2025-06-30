import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df_mcp = pd.read_csv("../mcp_6months.csv")
df_weather = pd.read_csv("../weather_6months.csv")

assert len(df_mcp) == len(df_weather), "Row count mismatch!"


df = pd.concat([df_mcp, df_weather], axis=1)
df = df.dropna()

features_for_clustering = ['mcp', 'Temperature','Solar Irradiance(W/m^2)',' Wind Speed(m/s)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_for_clustering])

kmeans = KMeans(n_clusters=4, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
summary = df.groupby('Cluster')[['mcp', 'Temperature', ' Wind Speed(m/s)', 'Solar Irradiance(W/m^2)']].mean().round(2)
print(summary)

label_map = {
    0: 'Hot Breezy Days(avg:4503)',
    1: 'Cold Windy Days(avg:4531)',
    2: 'Moderate off-peak Days(avg:4160)',
    3: 'Cool Breezy Peak days(avg:10254)'
}
colors = ['red', 'green', 'blue', 'orange']
for cluster_id, label in label_map.items():
    cluster_data = df[df['Cluster'] == cluster_id]
    ax.scatter(
        cluster_data['Temperature'],
        cluster_data[' Wind Speed(m/s)'],
        cluster_data['Solar Irradiance(W/m^2)'],
        label=label,
        color=colors[cluster_id],
        alpha=0.6,
        s=20
    )
ax.set_xlabel('Temperature')
ax.set_ylabel('Wind Speed')
ax.set_zlabel('Solar Irradiance')
plt.title("3D Cluster Plot (Temp, Wind, Solar)")
plt.legend(title='Cluster Label')
plt.show()