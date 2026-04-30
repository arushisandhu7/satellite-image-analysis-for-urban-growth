import rasterio
import numpy as np
import pandas as pd
import os

def extract_stats(file):
    city, year = file.replace(".tif","").split("_")

    with rasterio.open(file) as src:
        ndvi = src.read(1)
        ndbi = src.read(2)

        ndvi = np.where(ndvi == 0, np.nan, ndvi)
        ndbi = np.where(ndbi == 0, np.nan, ndbi)

        urban = np.sum(ndbi > -0.02)
        total = np.sum(~np.isnan(ndbi))

        return {
            "City": city,
            "Year": int(year),
            "NDVI": np.nanmean(ndvi),
            "NDBI": np.nanmean(ndbi),
            "Urban_Area_%": (urban/total)*100
        }

data = []

for file in os.listdir():
    if file.endswith(".tif"):
        data.append(extract_stats(file))

df = pd.DataFrame(data)
df.to_csv("urban_growth.csv", index=False)

print(df) # -----------------------------------
# 🌍 FINAL URBAN GROWTH SYSTEM
# -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------
# STEP 1: LOAD DATA
# -----------------------------------
df = pd.read_csv("urban_growth.csv")

print("Original Data:\n", df)

# -----------------------------------
# STEP 2: ADD CLIMATE DATA
# -----------------------------------
df['Temperature'] = df.apply(lambda row: {
    ("Delhi", 2020): 34,
    ("Delhi", 2024): 36,
    ("Mumbai", 2020): 31,
    ("Mumbai", 2024): 33,
    ("Bangalore", 2020): 27,
    ("Bangalore", 2024): 29
}.get((row['City'], row['Year']), 30), axis=1)

# -----------------------------------
# STEP 3: NORMALIZE URBAN %
# -----------------------------------
df['Urban_Area_%'] = df.groupby('City')['Urban_Area_%'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) * 20 + 40
)

df['Urban_Area_%'] = df['Urban_Area_%'].round(2)

# -----------------------------------
# STEP 4: ADD GROWTH RATE
# -----------------------------------
df['Growth_Rate_%'] = df.groupby('City')['Urban_Area_%'].pct_change() * 100

# -----------------------------------
# STEP 5: CATEGORY + RISK
# -----------------------------------
def classify(x):
    if x > 55:
        return "High Urban"
    elif x > 45:
        return "Moderate"
    else:
        return "Low Urban"

def risk_level(x):
    if x > 65:
        return "High Risk"
    elif x > 50:
        return "Moderate Risk"
    else:
        return "Low Risk"

df['Category'] = df['Urban_Area_%'].apply(classify)
df['Risk_Level'] = df['Urban_Area_%'].apply(risk_level)

# -----------------------------------
# STEP 6: DECISION SUPPORT
# -----------------------------------
def recommendation(row):
    if row['Urban_Area_%'] > 60:
        return "⚠ High urbanization – increase green cover"
    elif row['NDVI'] < 0.2:
        return "🌱 Low vegetation – afforestation needed"
    elif row['Growth_Rate_%'] > 5:
        return "📈 Rapid growth – monitor infrastructure"
    else:
        return "✅ Balanced development"

df['Recommendation'] = df.apply(recommendation, axis=1)

# -----------------------------------
# STEP 7: ML MODEL
# -----------------------------------
X = df[['Year', 'NDVI', 'NDBI', 'Temperature']]
y = df['Urban_Area_%']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------------
# STEP 8: FUTURE PREDICTIONS
# -----------------------------------
future = pd.DataFrame({
    'Year': [2030, 2035],
    'NDVI': [0.18, 0.15],
    'NDBI': [0.05, 0.08],
    'Temperature': [36, 38]
})

predictions = model.predict(future)

print("\n🔮 Future Predictions:")
for year, value in zip([2030, 2035], predictions):
    print(f"{year}: {value:.2f}% urban area")

# -----------------------------------
# STEP 9: SCENARIO FORECASTING
# -----------------------------------
scenarios = {
    "Low Growth": 0.85,
    "Medium Growth": 1.0,
    "High Growth": 1.15
}

scenario_results = []

for i, year in enumerate([2030, 2035]):
    base = predictions[i]

    for scenario in scenarios:
        scenario_results.append({
            "Year": year,
            "Scenario": scenario,
            "Urban_Area_%": round(base * scenarios[scenario], 2)
        })

scenario_df = pd.DataFrame(scenario_results)

print("\nScenario Forecasts:\n", scenario_df)

# -----------------------------------
# STEP 10: VISUALIZATION
# -----------------------------------
plt.figure()

for city in df['City'].unique():
    subset = df[df['City'] == city]
    plt.plot(subset['Year'], subset['Urban_Area_%'], marker='o', label=city)

plt.xlabel("Year")
plt.ylabel("Urban Area %")
plt.title("Urban Growth Comparison")
plt.legend()
plt.grid()
plt.show()

# -----------------------------------
# STEP 11: SCENARIO GRAPH
# -----------------------------------
plt.figure()

for scenario in scenario_df['Scenario'].unique():
    subset = scenario_df[scenario_df['Scenario'] == scenario]
    plt.plot(subset['Year'], subset['Urban_Area_%'], marker='o', label=scenario)

plt.title("Scenario-Based Forecast")
plt.xlabel("Year")
plt.ylabel("Urban Area %")
plt.legend()
plt.grid()
plt.show()

# -----------------------------------
# STEP 12: FEATURE IMPORTANCE
# -----------------------------------
importance = model.feature_importances_

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.show()

# -----------------------------------
# STEP 13: SPATIAL ANALYSIS
# -----------------------------------
try:
    with rasterio.open("Delhi_2020.tif") as src:
        ndbi_2020 = src.read(2)

    with rasterio.open("Delhi_2024.tif") as src:
        ndbi_2024 = src.read(2)

    # Heatmap
    plt.imshow(ndbi_2024)
    plt.colorbar(label="NDBI")
    plt.title("Urban Heatmap - Delhi 2024")
    plt.show()

    # Change detection
    change = ndbi_2024 - ndbi_2020
    plt.imshow(change)
    plt.colorbar(label="Change")
    plt.title("Urban Growth Map (2020 → 2024)")
    plt.show()

except:
    print("⚠ Spatial files not found, skipping maps")

# -----------------------------------
# STEP 14: SAVE FILES
# -----------------------------------
df.to_csv("urban_growth_final.csv", index=False)
scenario_df.to_csv("scenario_forecasts.csv", index=False)

print("\n✅ FINAL DATASET:\n", df)