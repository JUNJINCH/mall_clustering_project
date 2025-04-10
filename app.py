import streamlit as st
import pandas as pd
import numpy as np
from src.clustering import apply_kmeans
from src.utils import setup_logging
from src.visualization import plot_clusters_with_new

# Setup logging
setup_logging()

# App title
st.markdown("<h1 style='text-align: center;'>Customer Segmentation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app predicts the customer cluster using age, gender, income and spending score.</p>", unsafe_allow_html=True)

st.markdown("---")

# Input section
with st.form("input_form"):
    st.subheader("Enter Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=18, max_value=95, value=30)

    with col2:
        income = st.slider("Annual Income (in 1000 dollars)", min_value=0, max_value=200, value=60)
        score = st.slider("Spending Score (1–100)", min_value=0, max_value=100, value=50)

    submit = st.form_submit_button("Predict Customer Cluster")

# Run clustering and prediction
if submit:
    # Step 1: Load and prepare dataset
    df = pd.read_csv("data/mall_customers.csv")
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # Step 2: Train model
    features = df[["Gender", "Age", "Annual_Income", "Spending_Score"]]
    model = apply_kmeans(features, n_clusters=5)
    df["Cluster"] = model.labels_

    # Step 3: Predict new customer
    gender_value = 0 if gender == "Male" else 1
    new_customer = np.array([[gender_value, age, income, score]])
    prediction = model.predict(new_customer)[0]
    st.success(f"The predicted customer belongs to **Cluster {prediction}**.")

    # Step 4: Visualize
    fig = plot_clusters_with_new(df, income, score, prediction)
    st.pyplot(fig)
