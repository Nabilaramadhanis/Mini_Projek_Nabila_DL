# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

st.title("Mini Projek Nabila DL")
st.write("Aplikasi Streamlit untuk prediksi menggunakan model ML")

# ===========================
# Upload file CSV
# ===========================
st.header("Upload File CSV")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data berhasil diupload:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

# ===========================
# Load model ML (jika sudah ada)
# ===========================
st.header("Load Model Machine Learning")
model_file = st.file_uploader("Upload file model (.joblib)", type=["joblib"])

if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.success("Model berhasil di-load!")
    except Exception as e:
        st.error(f"Gagal load model: {e}")

# ===========================
# Input data manual dan prediksi
# ===========================
st.header("Prediksi Manual")
x_input = st.number_input("Masukkan nilai X untuk prediksi:", value=0)

if st.button("Prediksi"):
    try:
        if 'model' in locals():
            # jika ada model upload
            y_pred = model.predict(np.array([[x_input]]))
            st.write(f"Hasil prediksi: {y_pred[0]}")
        else:
            # fallback dummy model
            dummy_model = LinearRegression()
            dummy_model.coef_ = np.array([2])
            dummy_model.intercept_ = 5
            y_pred = dummy_model.coef_[0] * x_input + dummy_model.intercept_
            st.write(f"Hasil prediksi (dummy model): {y_pred}")
    except Exception as e:
        st.error(f"Gagal prediksi: {e}")




