# ============================================================
# Import Library
# ============================================================
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# Streamlit App Start
# ============================================================
st.set_page_config(page_title="Prediksi Banjir Jawa Barat", layout="wide")
st.title("📊 Prediksi Jumlah Kejadian Banjir - Jawa Barat (Random Forest)")

# ============================================================
# Custom CSS Styling
# ============================================================
st.markdown("""
<style>

    /* GLOBAL APP BACKGROUND */
    .main {
        background-color: #f9fafb;
    }

    /* TITLE */
    h1 {
        color: #005bbb !important;
        font-weight: 900;
        text-align: center;
        padding-bottom: 10px;
    }

    /* SUBHEADERS */
    h2, h3, h4 {
        color: #024c9a !important;
        font-weight: 800;
        margin-top: 25px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #e8eef7;
    }

    /* DATAFRAME STYLING */
    .stDataFrame table {
        border: 1px solid #d0d7de;
    }
    .stDataFrame th {
        background-color: #005bbb !important;
        color: white !important;
        font-weight: bold !important;
        text-transform: uppercase;
    }

    /* EXPANDER */
    .streamlit-expanderHeader {
        background-color: #dce6f7 !important;
        color: black;
        font-weight: 600;
        border-radius: 6px;
    }

    /* BUTTON */
    .stButton>button {
        background-color: #005bbb !important;
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #014a98 !important;
        color: #fff;
    }

    /* SLIDER */
    .stSlider > div > div > div {
        background: #005bbb !important;
    }

    /* TEXT INPUT / DROPDOWN */
    .stTextInput>div>div input, 
    .stSelectbox>div>div select {
        border-radius: 6px;
        border: 1px solid #005bbb;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# Business Understanding (unchanged)
# ============================================================
st.markdown("""
**Format file CSV yang valid (contoh baris):**  
`id,kode_provinsi,nama_provinsi,kode_kabupaten_kota,nama_kabupaten_kota,jumlah_banjir,satuan,tahun`  
Pastikan kolom `nama_kabupaten_kota`, `jumlah_banjir`, dan `tahun` ada.
""")

st.sidebar.header("Opsi & Upload")
st.sidebar.write("Gunakan file CSV Open Data Jabar (BPBD).")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset (bpbd-... .csv)", type=["csv"])

DEFAULT_CSV = "bpbd-od_17600_jml_kejadian_bencana_banjir__kabupatenkota_v3_data.csv"
use_default = False
if os.path.exists(DEFAULT_CSV):
    use_default = st.sidebar.checkbox(f"Gunakan file lokal '{DEFAULT_CSV}'", value=True)

# -------------------------
# Helper functions
# -------------------------
def load_dataframe(file_obj):
    df_local = pd.read_csv(file_obj)
    return df_local

def basic_cleaning(df):
    needed = ["nama_kabupaten_kota", "jumlah_banjir", "tahun", "kode_kabupaten_kota"]
    for col in ["nama_kabupaten_kota", "jumlah_banjir", "tahun"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.")
    df = df.copy()
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype(pd.Int64Dtype())
    df["jumlah_banjir"] = pd.to_numeric(df["jumlah_banjir"], errors="coerce").fillna(0).astype(int)
    if "kode_kabupaten_kota" in df.columns:
        df["kode_kabupaten_kota"] = pd.to_numeric(df["kode_kabupaten_kota"], errors="coerce").fillna(0).astype(int)
    else:
        df["kode_kabupaten_kota"] = df["nama_kabupaten_kota"].astype("category").cat.codes + 1000
    return df

def prepare_features(df):
    df_p = df.copy()
    ohe = pd.get_dummies(df_p["nama_kabupaten_kota"], prefix="kab")
    df_p = pd.concat([df_p, ohe], axis=1)
    feature_cols = ["kode_kabupaten_kota", "tahun"] + ohe.columns.tolist()
    X = df_p[feature_cols]
    y = df_p["jumlah_banjir"]
    return X, y, feature_cols, ohe.columns.tolist()

def save_model_and_scaler(model, scaler, model_name="model_prediksi_banjir_rf_jabar.pkl",
                          scaler_name="scaler_prediksi_banjir_jabar.pkl"):
    joblib.dump(model, model_name)
    joblib.dump(scaler, scaler_name)
    return model_name, scaler_name

def prepare_input_row_for_predict(nama_kab, tahun, feature_cols, df_ohe_cols, kode_map, scaler_obj):
    row = {c: 0 for c in feature_cols}
    kode = kode_map.get(nama_kab, None)
    if kode is None:
        for k,v in kode_map.items():
            if k.lower() == nama_kab.lower():
                kode = v
                break
    row["kode_kabupaten_kota"] = int(kode)
    row["tahun"] = int(tahun)
    ohe_col = f"kab_{nama_kab}"
    if ohe_col in row:
        row[ohe_col] = 1
    row_df = pd.DataFrame([row], columns=feature_cols)
    return scaler_obj.transform(row_df.values)

# -------------------------
# Load Data
# -------------------------
df = None
if uploaded_file is not None:
    df = load_dataframe(uploaded_file)
    st.sidebar.success("File CSV ter-upload.")
elif use_default:
    df = pd.read_csv(DEFAULT_CSV)
    st.sidebar.success(f"Memuat file lokal '{DEFAULT_CSV}'.")
else:
    st.info("Silakan upload CSV dataset atau gunakan file lokal.")
    st.stop()

try:
    df = basic_cleaning(df)
except Exception as e:
    st.error(f"Masalah format dataset: {e}")
    st.stop()

st.subheader("📋 Preview Dataset")
st.dataframe(df.head(8))

# ============================================================
# EDA Section (unchanged)
# ============================================================
# ... (ISI PLOT EDA ANDA TETAP SAMA, TANPA DIUBAH)
# Anda tidak perlu mengubah apapun — CSS langsung mempengaruhi tampilan.

# ============================================================
# Data Preparation
# ============================================================
st.subheader("🧹 Data Preparation")
X, y, feature_cols, ohe_cols = prepare_features(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# Modeling
# ============================================================
st.subheader("🤖 Modeling - Random Forest Regressor")

test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, step=5)
n_estimators = st.sidebar.slider("Jumlah estimator", 50, 500, 200, step=50)

if st.button("▶️ Latih Model (Train Random Forest)"):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, shuffle=True)
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    st.success("Training selesai!")

    st.write("Evaluasi:")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R²:", r2_score(y_test, y_pred))

    save_model_and_scaler(rf, scaler)

# ============================================================
# Deployment - Predict
# ============================================================
st.subheader("🚀 Deployment - Simulasi & Prediksi")

# ... (bagian predict tetap sama)

# ============================================================
# Catatan Akhir
# ============================================================
st.write("---")
st.write("**Catatan:** Tambahkan fitur cuaca/topografi untuk akurasi lebih tinggi.")
