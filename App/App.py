import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

from prediction import predict_lead_time
from recommendation import recommend_factory

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Nassau Candy Optimization",
    page_icon="🍬",
    layout="wide"
)

st.title("🍬 Factory Reallocation & Shipping Optimization System")

st.write(
"""
This dashboard predicts shipping lead time,
analyzes sales performance,
and recommends the best factory for production.
"""
)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Data" / "cleaned_nassau_candy.csv"

MODEL_PATH = BASE_DIR / "Models" / "random_forest_model.pkl"

SHIP_ENCODER = BASE_DIR / "Models" / "ship_encoder.pkl"

REGION_ENCODER = BASE_DIR / "Models" / "region_encoder.pkl"

FACTORY_ENCODER = BASE_DIR / "Models" / "factory_encoder.pkl"

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

ship_encoder = joblib.load(SHIP_ENCODER)

region_encoder = joblib.load(REGION_ENCODER)

factory_encoder = joblib.load(FACTORY_ENCODER)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

product = st.sidebar.selectbox(
    "Select Product",
    sorted(df["Product Name"].unique())
)

region = st.sidebar.selectbox(
    "Select Region",
    sorted(df["Region"].unique())
)

ship = st.sidebar.selectbox(
    "Select Ship Mode",
    sorted(df["Ship Mode"].unique())
)

filtered = df.copy()

filtered = filtered[
    filtered["Product Name"] == product
]

filtered = filtered[
    filtered["Region"] == region
]

filtered = filtered[
    filtered["Ship Mode"] == ship
]

if filtered.empty:
    st.warning("No matching records found.")
    st.stop()

    # ==========================================
# KPI Cards
# ==========================================

st.markdown("---")
st.subheader("📊 Business Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Orders",
    len(filtered)
)

col2.metric(
    "Total Sales",
    f"${filtered['Sales'].sum():,.2f}"
)

col3.metric(
    "Average Lead Time",
    f"{filtered['Lead Time'].mean():.2f} Days"
)

col4.metric(
    "Average Profit Margin",
    f"{filtered['Profit Margin (%)'].mean():.2f}%"
)

# ==========================================
# Filtered Dataset
# ==========================================

st.markdown("---")
st.subheader("📋 Filtered Orders")

st.dataframe(filtered, use_container_width=True)

# ==========================================
# Sales by Region
# ==========================================

st.markdown("---")
st.subheader("📈 Sales by Region")

sales_region = (
    df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    sales_region,
    x="Region",
    y="Sales",
    color="Sales",
    title="Total Sales by Region"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Profit Margin Distribution
# ==========================================

st.subheader("💰 Profit Margin Distribution")

fig = px.histogram(
    df,
    x="Profit Margin (%)",
    nbins=20,
    title="Profit Margin Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# AI Shipping Lead Time Prediction
# ==========================================

st.markdown("---")
st.subheader("🤖 AI Shipping Lead Time Prediction")

sample = filtered.iloc[0]

try:

    predicted_days = predict_lead_time(
        sales=sample["Sales"],
        units=sample["Units"],
        cost=sample["Cost"],
        gross_profit=sample["Gross Profit"],
        ship_mode=sample["Ship Mode"],
        region=sample["Region"],
        factory=sample["Factory"]
    )

    st.success(
        f"Estimated Shipping Lead Time: {predicted_days:.2f} Days"
    )

except Exception as e:

    st.error(f"Prediction Error: {e}")

    # ==========================================
# Factory Recommendation
# ==========================================

st.markdown("---")
st.subheader("🏭 Best Factory Recommendation")

recommendation = recommend_factory(df, product)

if recommendation.empty:

    st.warning("No recommendation available.")

else:

    st.dataframe(recommendation, use_container_width=True)

    best_factory = recommendation.iloc[0]["Factory"]

    st.success(
        f"Recommended Factory: {best_factory}"
    )

    # ==========================================
# Lead Time Comparison
# ==========================================

st.markdown("---")
st.subheader("📉 Factory Lead Time Comparison")

factory_data = (
    df.groupby("Factory")["Lead Time"]
    .mean()
    .reset_index()
)

fig = px.bar(
    factory_data,
    x="Factory",
    y="Lead Time",
    color="Lead Time",
    title="Average Lead Time by Factory"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🏭 Factory Recommendation")

recommendation = recommend_factory(df, product)

if recommendation.empty:

    st.warning("No recommendation available.")

else:

    st.dataframe(recommendation, use_container_width=True)

    best_factory = recommendation.iloc[0]["Factory"]

    st.success(
        f"✅ Recommended Factory : {best_factory}"
    )

    fig = px.bar(
        recommendation,
        x="Factory",
        y="Lead Time",
        color="Profit Margin (%)",
        title="Factory Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
st.subheader("📌 Prediction Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Predicted Lead Time",
    f"{predicted_days:.2f} Days"
)

c2.metric(
    "Recommended Factory",
    best_factory
)

c3.metric(
    "Average Profit Margin",
    f"{filtered['Profit Margin (%)'].mean():.2f}%"
)

st.markdown("---")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name="filtered_orders.csv",
    mime="text/csv"
)