import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained models
models = {}
model_names = ["Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "Support Vector Regressor"]
for name in model_names:
    with open(f'{name}_model.pkl', 'rb') as file:
        models[name] = pickle.load(file)

# Function to predict using selected model
def predict_price(model_name, features):
    model = models[model_name]
    prediction = model.predict(features)
    return prediction

# Streamlit app
def main():
    st.title("Gold Price Prediction")
    st.sidebar.header("Input Features")
    
    close = st.sidebar.number_input("Close", min_value=0.0)
    inflation = st.sidebar.number_input("Inflation", min_value=0.0)
    exchange_rate = st.sidebar.number_input("Exchange Rate", min_value=0.0)

    feature_vector = np.array([[close, inflation, exchange_rate]])

    st.write("\n### Predictions:")
    for name, model in models.items():
        prediction = predict_price(name, feature_vector)
        st.write(f"**{name}**: ${prediction[0]:.2f}")

    # Allow user to upload their own dataset
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("\n### Sample of Uploaded Data:")
        st.write(data.head())

if __name__ == "__main__":
    main()
