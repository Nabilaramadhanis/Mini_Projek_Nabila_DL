import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Spotify Popularity Predictor", layout="centered")

# --- Load data safely ---
FNAME = "spotify_data_clean.csv"
try:
    df = pd.read_csv(FNAME)
except Exception:
    # fallback: coba nama dengan spasi
    try:
        df = pd.read_csv("spotify_data clean.csv")
    except Exception as e:
        st.error(f"Error membaca file dataset: {e}")
        st.stop()

st.title("🎵 Spotify Popularity Predictor (CRISP-DM)")

st.write("### Dataset preview")
st.dataframe(df.head())

# --- Automatic feature detection ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("Tidak ada kolom numerik di dataset. Aplikasi membutuhkan fitur numerik (danceability, energy, valence, tempo, dll).")
    st.stop()

# Target detection
target = None
for t in ["popularity","Popularity","track_popularity"]:
    if t in df.columns:
        target = t
        break
if target is None:
    target = numeric_cols[-1]  # fallback
    st.info(f"Tidak menemukan kolom 'popularity' — menggunakan {target} sebagai target.")

# Feature selection (preferensi)
preferred = [c for c in ["danceability","energy","valence","tempo","loudness","speechiness","acousticness","instrumentalness"] if c in df.columns]
features = preferred if preferred else [c for c in numeric_cols if c != target][:6]

st.write("Fitur yang dipakai:", features)
st.write("Target:", target)

# Prepare data
df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Eval
r2 = model.score(X_test, y_test)
st.write(f"### Model evaluation — R²: {r2:.3f}")

st.write("### Predict song popularity")
# Input widgets: create slider ranges based on dataset min/max for each feature
input_vals = {}
for f in features:
    minv, maxv = float(df[f].min()), float(df[f].max())
    default = float(df[f].median())
    # if feature scale 0..1 use float slider else use an int/float slider with wider range
    if minv >= 0 and maxv <= 1:
        input_vals[f] = st.slider(f"{f}", min_value=0.0, max_value=1.0, value=default, step=0.01)
    else:
        input_vals[f] = st.slider(f"{f}", min_value=minv, max_value=maxv, value=default)

input_df = pd.DataFrame([input_vals])[features]
pred = model.predict(input_df)[0]
st.success(f"Predicted popularity score: {pred:.2f}")

# show feature importance (coefficients)
st.write("Feature coefficients (importance):")
coefs = pd.DataFrame({"feature": features, "coefficient": model.coef_})
st.table(coefs.sort_values(by="coefficient", ascending=False))
