# ============================================================
# Business Understanding
# ============================================================
## Project Domain
# Domain: Lingkungan Hidup / Mitigasi Bencana
# Project ini fokus pada pemanfaatan data kejadian banjir (jumlah kejadian per kabupaten/kota per tahun)
# di Provinsi Jawa Barat untuk membangun model prediksi jumlah kejadian banjir tiap kabupaten/kota.
# Tujuan praktis: membantu BPBD / pemangku kebijakan memetakan risiko, merencanakan mitigasi,
# dan mengalokasikan sumber daya pencegahan berdasarkan prediksi jumlah kejadian banjir.

## Problem Statements
# - Tidak ada prediksi terstruktur untuk jumlah kejadian banjir per kabupaten/kota.
# - Keputusan penanganan bencana sering reaktif karena keterbatasan informasi prediktif.
# - Diperlukan model sederhana yang dapat dipakai untuk memperkirakan jumlah kejadian di tahun berikutnya.

## Goals
# - Membangun model machine learning (supervised regression) untuk memprediksi 'jumlah_banjir'.
# - Menyediakan pipeline lengkap (EDA, preprocessing, modeling, evaluasi, simulasi input baru, simpan model).
# - Hasil dapat digunakan lebih lanjut di aplikasi dashboard/Streamlit.

## Solution Statements
# - Gunakan RandomForestRegressor (stabil terhadap outlier dan non-linearitas).
# - Input model: fitur numerik (tahun, kode_kabupaten_kota) + one-hot nama_kabupaten_kota.
# - Output: prediksi jumlah kejadian banjir (regresi).
# - Simpan model (.pkl) untuk integrasi ke aplikasi Streamlit atau sistem lain.

# ============================================================
# Data Understanding
# ============================================================
## Dataset Description
# Dataset file (Open Data Jabar) yang digunakan dalam script ini:
# Nama file contoh yang diharapkan:
# bpbd-od_17600_jml_kejadian_bencana_banjir__kabupatenkota_v3_data.csv
#
# Kolom (sesuai contoh yang diberikan oleh user):
# id,kode_provinsi,nama_provinsi,kode_kabupaten_kota,nama_kabupaten_kota,jumlah_banjir,satuan,tahun
#
# Sumber dataset: Open Data Jawa Barat (BPBD) — publik.
#
## Cara membuat datasetnya
# - Data ini berasal dari pencatatan kejadian banjir oleh BPBD / pemprov.
# - Kolom 'jumlah_banjir' adalah agregasi jumlah kejadian untuk kabupaten/kota per tahun.
# - Untuk model ini gunakan minimal kolom: nama_kabupaten_kota, tahun, jumlah_banjir.
#
## Library Requirements
# - pandas, numpy, matplotlib, seaborn
# - scikit-learn
# - joblib
# - streamlit
#
# Install bila perlu:
# pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

# ============================================================
# Exploratory Data Analysis (akan ditampilkan di Streamlit)
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

st.markdown("""
**Format file CSV yang valid (contoh baris):**  
`id,kode_provinsi,nama_provinsi,kode_kabupaten_kota,nama_kabupaten_kota,jumlah_banjir,satuan,tahun`  
Pastikan kolom `nama_kabupaten_kota`, `jumlah_banjir`, dan `tahun` ada.
""")

# -------------------------
# Sidebar: Pilihan & Upload
# -------------------------
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
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset. Pastikan format CSV benar.")

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
        matched = None
        for k,v in kode_map.items():
            if k.lower() == nama_kab.lower():
                matched = v
                break
        if matched is None:
            raise ValueError(f"Nama kabupaten/kota '{nama_kab}' tidak ditemukan.")
        kode = matched

    row["kode_kabupaten_kota"] = int(kode)
    row["tahun"] = int(tahun)

    ohe_col = f"kab_{nama_kab}"
    if ohe_col in row:
        row[ohe_col] = 1
    else:
        found = None
        for c in df_ohe_cols:
            if c.lower().endswith(nama_kab.lower()):
                found = c
                break
        if found:
            row[found] = 1

    row_df = pd.DataFrame([row], columns=feature_cols)
    row_scaled = scaler_obj.transform(row_df.values)
    return row_scaled

# -------------------------
# Load Data
# -------------------------
df = None
if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file)
        st.sidebar.success("File CSV ter-upload.")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
        st.stop()
elif use_default:
    try:
        df = pd.read_csv(DEFAULT_CSV)
        st.sidebar.success(f"Memuat file lokal '{DEFAULT_CSV}'.")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file lokal: {e}")
        st.stop()
else:
    st.info("Silakan upload CSV dataset atau gunakan file lokal.")
    st.stop()

# ============================================================
# ... (seluruh script berlanjut normal seperti sebelumnya)
# ============================================================
