import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Title
st.title("🎵 Spotify Prediction App")
st.write("Aplikasi Machine Learning untuk memprediksi popularitas lagu menggunakan Streamlit.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_data_clean.csv")
    return df

df = load_data()
st.subheader("📌 Data Sample")
st.write(df.head())

# Select Features & Target
features = st.multiselect("Pilih fitur model:", df.columns.tolist(), default=df.columns[1:7])
target = st.selectbox("Pilih kolom target:", df.columns.tolist(), index=0)

if st.button("Train Model"):
    X = df[features]
    y = df[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"🎉 Model berhasil dilatih dengan akurasi: **{acc:.2f}**")

    # Save model pickle
    with open("model_spotify.pkl", "wb") as file:
        pickle.dump((model, scaler, features, target), file)

    st.info("Model berhasil disimpan sebagai `model_spotify.pkl`")

# Predict New Data
st.subheader("🔮 Prediksi Data Baru")
try:
    with open("model_spotify.pkl", "rb") as file:
        model, scaler, selected_features, target_name = pickle.load(file)

    input_data = {}
    for col in selected_features:
        input_data[col] = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)

        st.success(f"📌 Prediksi untuk target **{target_name}** adalah: **{result[0]}**")

except:
    st.warning("⚠ Latih model dulu sebelum melakukan prediksi!")


