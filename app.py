import streamlit as st
import pandas as pd
import rasterio
import plotly.express as px
import numpy as np

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Urban Growth Dashboard", layout="wide")

st.title("🌍 Smart Urban Growth Analytics")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/arus/Final-Project/Final-project-8sem/urban_growth_final.csv")
    scenario_df = pd.read_csv("scenario_forecasts.csv")
    return df, scenario_df

df, scenario_df = load_data()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Filters")

selected_city = st.sidebar.selectbox(
    "Select City",
    df['City'].unique()
)

# -----------------------------------
# FILTER DATA
# -----------------------------------
filtered = df[df['City'] == selected_city]

latest = filtered.sort_values("Year").iloc[-1]

# -----------------------------------
# KPI CARDS
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Urban Area %", f"{latest['Urban_Area_%']:.2f}%")
col2.metric("NDVI", f"{latest['NDVI']:.2f}")
col3.metric("Temperature", f"{latest['Temperature']}°C")
col4.metric("Growth Rate", f"{latest['Growth_Rate_%']:.2f}%")

# -----------------------------------
# MAIN CHARTS
# -----------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("📈 Urban Growth Trend")
    st.line_chart(filtered.set_index("Year")["Urban_Area_%"])

with colB:
    st.subheader("🌿 NDVI Trend")
    st.line_chart(filtered.set_index("Year")["NDVI"])

# -----------------------------------
# MULTI-CITY COMPARISON
# -----------------------------------
st.subheader("🏙 City Comparison")

comparison = df.pivot(index="Year", columns="City", values="Urban_Area_%")
st.line_chart(comparison)

# -----------------------------------
# SCENARIO FORECAST
# -----------------------------------
st.subheader("🔮 Scenario Forecasting")

scenario_filtered = scenario_df

for scenario in scenario_filtered['Scenario'].unique():
    subset = scenario_filtered[scenario_filtered['Scenario'] == scenario]
    st.line_chart(subset.set_index("Year")["Urban_Area_%"])

# -----------------------------------
# RECOMMENDATIONS
# -----------------------------------
st.subheader("🧭 Decision Support")

st.dataframe(filtered[['Year', 'Urban_Area_%', 'Recommendation', 'Risk_Level']])




st.markdown("### 🌐 Satellite-Based Spatial Insights")




st.subheader("🗺️ Urban Heatmap (Optimized)")

city_file_map = {
    "Delhi": "Delhi_2024.tif",
    "Mumbai": "Mumbai_2024.tif",
    "Bangalore": "Bangalore_2024.tif"
}

selected_file = city_file_map.get(selected_city)

try:
    with rasterio.open(selected_file) as src:
        ndbi = src.read(2)

        # 🔥 CRITICAL FIX: reduce size
        ndbi = ndbi[::15, ::15]

        # Clip values for better contrast
        ndbi = np.clip(ndbi, -0.1, 0.1)

        fig = px.imshow(
            ndbi,
            color_continuous_scale='RdYlGn',
            title=f"{selected_city} Urban Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠ Heatmap error: {e}")





st.subheader("🏙 Urban Classification Map")

try:
    with rasterio.open(selected_file) as src:
        ndbi = src.read(2)

        # Downsample
        ndbi = ndbi[::15, ::15]

        # Create binary map
        urban_map = (ndbi > -0.02).astype(int)

        fig2 = px.imshow(
            urban_map,
            color_continuous_scale='gray',
            title="Urban Areas (White = Urban)"
        )

        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"⚠ Classification error: {e}")





st.subheader("📍 Urban Change Detection")

file_2020 = selected_city + "_2020.tif"
file_2024 = selected_city + "_2024.tif"

try:
    with rasterio.open(file_2020) as src:
        ndbi_2020 = src.read(2)

    with rasterio.open(file_2024) as src:
        ndbi_2024 = src.read(2)

    # 🔥 Downsample BOTH
    ndbi_2020 = ndbi_2020[::15, ::15]
    ndbi_2024 = ndbi_2024[::15, ::15]

    change = ndbi_2024 - ndbi_2020

    fig3 = px.imshow(
        change,
        color_continuous_scale='RdBu',
        zmin=-0.05,
        zmax=0.05,
        title="Urban Growth (Red = Increase)"
    )

    st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.warning(f"⚠ Change detection error: {e}")



# -----------------------------------
# DOWNLOAD BUTTON
# -----------------------------------
st.download_button(
    "📥 Download Data",
    df.to_csv(index=False),
    file_name="urban_growth_final.csv"
)