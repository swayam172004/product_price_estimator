import streamlit as st
import pandas as pd
import os
from src.predict import train_model, predict_price

st.title("💰 Product Price Estimator")

category = st.selectbox("Category", ['Electronics', 'Furniture', 'Apparel', 'Kitchen', 'Toys'])
brand = st.selectbox("Brand", ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE'])
weight_kg = st.slider("Weight (kg)", 0.1, 50.0, 10.0)
rating = st.slider("Rating (1-5)", 1.0, 5.0, 4.0)
warranty_years = st.slider("Warranty (years)", 0, 5, 1)
power_usage_watts = st.slider("Power Usage (Watts)", 0, 2000, 500)
feature_score = st.slider("Feature Score (0-100)", 0.0, 100.0, 50.0)

user_input = {
    'category': category,
    'brand': brand,
    'weight_kg': weight_kg,
    'rating': rating,
    'warranty_years': warranty_years,
    'power_usage_watts': power_usage_watts,
    'feature_score': feature_score
}

if st.button("Predict Price"):
    score = train_model()
    st.success(f"📈 Model Score: {score}")
    price = predict_price(user_input)
    st.success(f"💵 Estimated Price: ₹{price}")
