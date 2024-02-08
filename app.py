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
    st.title("Prediksi Harga Emas")
    st.sidebar.header("Fitur Input")
    
    harga_penutupan = st.sidebar.number_input("Harga Penutupan (IHSG)", min_value=0.0)
    inflasi = st.sidebar.number_input("Inflasi", min_value=0.0)
    kurs_jual = st.sidebar.number_input("Kurs Jual", min_value=0.0)
    kurs_beli = st.sidebar.number_input("Kurs Beli", min_value=0.0)

    kurs_dollar = (kurs_jual + kurs_beli) / 2  # Rata-rata kurs jual dan kurs beli sebagai kurs dollar

    fitur = np.array([[harga_penutupan, inflasi, kurs_dollar]])

    st.write("\n### Prediksi Harga Emas:")
    for name, model in models.items():
        prediksi = predict_price(name, fitur)
        st.write(f"**{name}**: Rp {prediksi[0]:,.2f}")

    # Memungkinkan pengguna mengunggah dataset mereka sendiri
    st.sidebar.header("Unggah Dataset Anda")
    file_yang_diunggah = st.sidebar.file_uploader("Unggah CSV", type=["csv"])

    if file_yang_diunggah is not None:
        data = pd.read_csv(file_yang_diunggah)
        st.write("\n### Sampel Data yang Diunggah:")
        st.write(data.head())

if __name__ == "__main__":
    main()
