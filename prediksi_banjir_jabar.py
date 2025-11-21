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
# Exploratory Data Abalasis (akan ditampilkan di Streamlit)
# - 2 line plot
# - 2 box plot
# - 2 pie chart
# - scatter plot
# - correlation matrix
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

# Provide option to use local CSV if exists with expected filename
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
    # keep necessary columns; if absent, raise
    for col in ["nama_kabupaten_kota", "jumlah_banjir", "tahun"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset. Pastikan format CSV benar.")
    # ensure types
    df = df.copy()
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype(pd.Int64Dtype())
    df["jumlah_banjir"] = pd.to_numeric(df["jumlah_banjir"], errors="coerce").fillna(0).astype(int)
    if "kode_kabupaten_kota" in df.columns:
        df["kode_kabupaten_kota"] = pd.to_numeric(df["kode_kabupaten_kota"], errors="coerce").fillna(0).astype(int)
    else:
        # create fallback kode (based on label encoding)
        df["kode_kabupaten_kota"] = df["nama_kabupaten_kota"].astype("category").cat.codes + 1000
    # drop rows with NaN year
    df = df[df["tahun"].notna()]
    return df

def prepare_features(df):
    df_p = df.copy()
    # one-hot encode nama_kabupaten_kota
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
    # create dict zeros
    row = {c: 0 for c in feature_cols}
    # kode_kabupaten_kota
    kode = kode_map.get(nama_kab, None)
    if kode is None:
        # fallback: try match ignoring case
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
    # set one-hot
    ohe_col = f"kab_{nama_kab}"
    if ohe_col in row:
        row[ohe_col] = 1
    else:
        # try to find matching ohe col ignoring case/spaces
        found = None
        for c in df_ohe_cols:
            # match end of col name with kabupaten name (case-insensitive)
            if c.lower().endswith(nama_kab.lower()):
                found = c
                break
        if found:
            row[found] = 1
        else:
            # no matching one-hot -> leave zeros (model will use kode_kabupaten_kota and tahun)
            pass
    row_df = pd.DataFrame([row], columns=feature_cols)
    row_scaled = scaler_obj.transform(row_df.values)
    return row_scaled

# -------------------------
# Load Data (uploaded or default)
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
    st.info("Silakan upload CSV dataset di sidebar atau tempatkan file lokal yang bernama:\n"
            f"'{DEFAULT_CSV}' dan centang opsi 'Gunakan file lokal'.")
    st.stop()

# -------------------------
# Clean & EDA (display)
# -------------------------
try:
    df = basic_cleaning(df)
except Exception as e:
    st.error(f"Masalah pada format dataset: {e}")
    st.stop()

# compute max_year for later validation
max_year = int(df["tahun"].max())

st.subheader("📋 Preview Dataset")
st.dataframe(df.head(8))

with st.expander("ℹ️ Informasi Dataset (jumlah baris, kolom, tipe)"):
    st.write("Shape:", df.shape)
    st.write(df.dtypes)
    st.write("Jumlah unique kabupaten/kota:", df["nama_kabupaten_kota"].nunique())
    st.write("Rentang tahun:", int(df["tahun"].min()), "-", int(df["tahun"].max()))
    st.write(df.describe(include="all"))

# ============= EDA PLOTS (2 line, 2 box, 2 pie, scatter, correlation) =============
st.subheader("📈 Exploratory Data Analysis (EDA)")

# Line plot 1: total kejadian provinsi per tahun
agg_prov = df.groupby("tahun")["jumlah_banjir"].sum().reset_index()
fig_lp1, ax = plt.subplots(figsize=(8,3))
ax.plot(agg_prov["tahun"], agg_prov["jumlah_banjir"], marker="o")
ax.set_title("Tren Total Jumlah Kejadian Banjir - Jawa Barat (per Tahun)")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Kejadian (Total)")
ax.grid(True)
st.pyplot(fig_lp1)

# Line plot 2: per-kabupaten contoh (top 2 by total)
top2 = df.groupby("nama_kabupaten_kota")["jumlah_banjir"].sum().sort_values(ascending=False).head(2).index.tolist()
fig_lp2, ax = plt.subplots(figsize=(8,3))
for k in top2:
    d = df[df["nama_kabupaten_kota"]==k].groupby("tahun")["jumlah_banjir"].sum().reset_index()
    ax.plot(d["tahun"], d["jumlah_banjir"], marker="o", label=k)
ax.set_title("Tren Jumlah Kejadian Banjir - Top 2 Kabupaten/Kota")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Kejadian")
ax.legend()
ax.grid(True)
st.pyplot(fig_lp2)

# Box plot 1: distribusi jumlah_banjir per tahun (pakai 8 tahun terakhir jika tersedia)
years_sorted = sorted(df["tahun"].unique())
years_choice = years_sorted[-8:] if len(years_sorted) >= 8 else years_sorted
data_for_box = [df[df["tahun"]==y]["jumlah_banjir"].values for y in years_choice]
fig_box1, ax = plt.subplots(figsize=(9,3))
ax.boxplot(data_for_box, labels=years_choice)
ax.set_title("Boxplot Distribusi Jumlah Kejadian Banjir per Tahun (pilihan tahun)")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Kejadian")
st.pyplot(fig_box1)

# Box plot 2: distribusi per kabupaten (6 teratas)
top6 = df.groupby("nama_kabupaten_kota")["jumlah_banjir"].sum().sort_values(ascending=False).head(6).index.tolist()
data_box_kab = [df[df["nama_kabupaten_kota"]==k]["jumlah_banjir"].values for k in top6]
fig_box2, ax = plt.subplots(figsize=(10,4))
ax.boxplot(data_box_kab, labels=top6)
ax.set_title("Boxplot Distribusi Jumlah Kejadian Banjir - 6 Kabupaten/Kota Teratas (Total)")
ax.set_xlabel("Kabupaten/Kota")
ax.set_ylabel("Jumlah Kejadian")
st.pyplot(fig_box2)

# Pie chart 1: proporsi total kejadian per kabupaten (top 8 + others)
total_by_kab = df.groupby("nama_kabupaten_kota")["jumlah_banjir"].sum().sort_values(ascending=False)
top8 = total_by_kab.head(8)
others = total_by_kab.iloc[8:].sum()
pie_vals = list(top8.values) + ([others] if others>0 else [])
pie_labels = list(top8.index) + (["Lainnya"] if others>0 else [])
fig_p1, ax = plt.subplots(figsize=(6,6))
ax.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", startangle=90)
ax.set_title("Proporsi Total Kejadian Banjir per Kabupaten/Kota (Top 8 + Lainnya)")
st.pyplot(fig_p1)

# Pie chart 2: proporsi kejadian per tahun (6 tahun terakhir)
total_by_year = df.groupby("tahun")["jumlah_banjir"].sum().sort_index()
year_slice = total_by_year[-6:] if len(total_by_year) >= 6 else total_by_year
fig_p2, ax = plt.subplots(figsize=(6,6))
ax.pie(year_slice.values, labels=year_slice.index.astype(str).tolist(), autopct="%1.1f%%", startangle=90)
ax.set_title("Proporsi Total Kejadian Banjir (6 Tahun Terakhir)")
st.pyplot(fig_p2)

# Scatter plot: tahun vs jumlah_banjir
fig_sc, ax = plt.subplots(figsize=(8,3))
ax.scatter(df["tahun"], df["jumlah_banjir"], alpha=0.5)
ax.set_title("Scatter: Tahun vs Jumlah Kejadian Banjir")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Kejadian")
st.pyplot(fig_sc)

# Correlation matrix (numeric)
corr_df = df[["kode_kabupaten_kota", "tahun", "jumlah_banjir"]] if "kode_kabupaten_kota" in df.columns else df[["tahun", "jumlah_banjir"]]
fig_corr, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix (numeric columns)")
st.pyplot(fig_corr)

# Short EDA text summary
with st.expander("Ringkasan EDA (interpretasi singkat)"):
    st.write("- Data berisi kejadian banjir per kabupaten/kota per tahun.")
    st.write("- Terdapat fluktuasi antar-tahun; beberapa kabupaten menunjukkan jumlah kejadian yang lebih tinggi.")
    st.write("- Korelasi sederhana menunjukkan hubungan 'tahun' vs 'jumlah_banjir' bersifat lemah/tergantung lokasi; perlu fitur eksternal (curah hujan, topografi) untuk perbaikan.")

# ============================================================
# Data Preparation
# ============================================================
st.subheader("🧹 Data Preparation")

st.write("Mempersiapkan fitur untuk modeling: one-hot nama_kabupaten_kota + kode_kabupaten_kota + tahun")

X, y, feature_cols, ohe_cols = prepare_features(df)
st.write("Jumlah fitur setelah one-hot:", len(feature_cols))
st.write("Contoh fitur (pertama):", feature_cols[:8])

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# informasi dataset
with st.expander("Detail Dataset Information"):
    st.write("Jumlah baris:", X.shape[0])
    st.write("Jumlah kolom fitur:", X.shape[1])
    st.write("Beberapa nilai unik kabupaten/kota:", df["nama_kabupaten_kota"].unique()[:10].tolist())
    st.write("Rentang tahun:", int(df["tahun"].min()), "-", int(df["tahun"].max()))

# ============================================================
# Modeling
# ============================================================
st.subheader("🤖 Modeling - Random Forest Regressor")

test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
n_estimators = st.sidebar.slider("Jumlah estimator (n_estimators)", 50, 500, value=200, step=50)
random_state = 42

if st.button("▶️ Latih Model (Train Random Forest)"):
    with st.spinner("Melatih model Random Forest..."):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=(test_size/100), random_state=random_state, shuffle=True)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)

        # prediksi & evaluasi
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("Training selesai ✅")
        st.write("**Metode evaluasi (regresi):** MSE, RMSE, MAE, R²")
        st.write(f"- MSE : {mse:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        st.write(f"- MAE : {mae:.3f}")
        st.write(f"- R²  : {r2:.3f}")

        # tampilkan sample prediksi
        df_compare = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": np.round(y_pred, 2)
        }).reset_index(drop=True).head(10)
        st.write("Contoh hasil prediksi (sample 10):")
        st.dataframe(df_compare)

        # Save model & scaler
        model_filename, scaler_filename = save_model_and_scaler(rf, scaler)
        st.write(f"Model disimpan sebagai: `{model_filename}`")
        st.write(f"Scaler disimpan sebagai: `{scaler_filename}`")

        # Simpan juga mapping nama_kab -> kode (untuk simulasi)
        kode_map = dict(zip(df["nama_kabupaten_kota"], df["kode_kabupaten_kota"]))
        joblib.dump(kode_map, "kode_map_kabupaten.pkl")
        st.write("Mapping kode kabupaten disimpan sebagai: `kode_map_kabupaten.pkl`")

        # expose trained model to session state for prediction without reload
        st.session_state["trained_model"] = rf
        st.session_state["scaler_obj"] = scaler
        st.session_state["feature_cols"] = feature_cols
        st.session_state["ohe_cols"] = ohe_cols
        st.session_state["kode_map"] = kode_map

# If a model file already exists, load it as a convenience
MODEL_LOCAL = "model_prediksi_banjir_rf_jabar.pkl"
SCALER_LOCAL = "scaler_prediksi_banjir_jabar.pkl"
KODEMAP_LOCAL = "kode_map_kabupaten.pkl"

if os.path.exists(MODEL_LOCAL) and "trained_model" not in st.session_state:
    try:
        loaded_rf = joblib.load(MODEL_LOCAL)
        loaded_scaler = joblib.load(SCALER_LOCAL) if os.path.exists(SCALER_LOCAL) else scaler
        loaded_kodemap = joblib.load(KODEMAP_LOCAL) if os.path.exists(KODEMAP_LOCAL) else dict(zip(df["nama_kabupaten_kota"], df["kode_kabupaten_kota"]))
        st.session_state["trained_model"] = loaded_rf
        st.session_state["scaler_obj"] = loaded_scaler
        st.session_state["feature_cols"] = feature_cols
        st.session_state["ohe_cols"] = ohe_cols
        st.session_state["kode_map"] = loaded_kodemap
        st.info("Model terdeteksi di folder lokal dan dimuat otomatis.")
    except Exception as e:
        st.warning(f"Gagal memuat model lokal: {e}")

# ============================================================
# Evaluation (if model trained or loaded)
# ============================================================
if "trained_model" in st.session_state:
    st.subheader("📊 Evaluasi Model (terakhir dilatih/termuat)")
    rf_model = st.session_state["trained_model"]

    # Evaluate on a holdout split for reporting (recompute)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=(test_size/100), random_state=random_state, shuffle=True)
    y_pred_te = rf_model.predict(X_te)
    mse_te = mean_squared_error(y_te, y_pred_te)
    rmse_te = np.sqrt(mse_te)
    mae_te = mean_absolute_error(y_te, y_pred_te)
    r2_te = r2_score(y_te, y_pred_te)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("MAE", f"{mae_te:.2f}")
    col_b.metric("RMSE", f"{rmse_te:.2f}")
    col_c.metric("MSE", f"{mse_te:.2f}")
    col_d.metric("R²", f"{r2_te:.3f}")

    st.write("Catatan: evaluasi di atas dihasilkan dari model yang saat ini dimuat/dilatih pada data ini. Untuk metrik yang stabil, gunakan cross-validation atau holdout dataset terpisah.")

# ============================================================
# Deployment
# ============================================================
st.subheader("🚀 Deployment - Simulasi & Prediksi (UI)")
st.write("Prediksi sekarang hanya menerima input tahun **masa depan** (lebih besar dari tahun maksimum dataset).")

# Informasi tahun maksimum (prevent backward prediction)
st.info(f"Tahun maksimum dalam dataset: **{max_year}** — Prediksi dibatasi untuk tahun > {max_year}.")

if "trained_model" in st.session_state:
    kab_list = sorted(df["nama_kabupaten_kota"].unique().tolist())
    selected_kab = st.selectbox("Pilih Kabupaten/Kota untuk Prediksi", kab_list)

    # Ganti dropdown tahun dengan number_input yang memaksa tahun masa depan
    selected_year = st.number_input(
        f"Masukkan Tahun Prediksi (harus > {max_year})",
        min_value=max_year + 1,
        max_value=2100,
        value=max_year + 1,
        step=1
    )

    if st.button("🔮 Lakukan Prediksi (gunakan model terlatih/termuat)"):
        try:
            # double-check model present
            rf_model = st.session_state["trained_model"]
            scaler_obj = st.session_state["scaler_obj"]
            feature_cols = st.session_state["feature_cols"]
            ohe_cols = st.session_state["ohe_cols"]
            kode_map = st.session_state["kode_map"]

            # prepare input (will be scaled inside)
            X_input_scaled = prepare_input_row_for_predict(selected_kab, selected_year, feature_cols, ohe_cols, kode_map, scaler_obj)
            pred_val = rf_model.predict(X_input_scaled)[0]
            st.success(f"Hasil prediksi jumlah kejadian banjir di **{selected_kab}** pada tahun **{selected_year}**: **{pred_val:.0f} kejadian**")

            # Tampilkan grafik historis kabupaten + titik prediksi (prediksi berada di masa depan)
            df_kab = df[df["nama_kabupaten_kota"] == selected_kab].groupby("tahun")["jumlah_banjir"].sum().reset_index()
            fig_hist, ax = plt.subplots(figsize=(8,3))
            ax.plot(df_kab["tahun"], df_kab["jumlah_banjir"], marker="o", label="historis")
            ax.scatter([selected_year], [pred_val], label="prediksi (future)", zorder=5)
            ax.set_title(f"Tren Historis & Prediksi - {selected_kab}")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Jumlah Kejadian")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig_hist)

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
else:
    st.warning("Model belum dilatih atau dimuat. Tekan 'Latih Model' atau muat model lokal.")

# Save model button (manual)
if st.button("💾 Simpan Model Saat Ini ke .pkl"):
    if "trained_model" in st.session_state:
        joblib.dump(st.session_state["trained_model"], MODEL_LOCAL)
        joblib.dump(st.session_state["scaler_obj"], SCALER_LOCAL)
        joblib.dump(st.session_state["kode_map"], KODEMAP_LOCAL)
        st.success(f"Model disimpan ke `{MODEL_LOCAL}` dan scaler ke `{SCALER_LOCAL}` serta kode_map `{KODEMAP_LOCAL}`")
    else:
        st.error("Model belum tersedia untuk disimpan.")

# ============================================================
# Model Simulation (Batch)
# ============================================================
st.subheader("🧪 Contoh Simulasi (Batch) - Upload CSV kecil berisi kabupaten & tahun (opsional)")
st.write("Format contoh CSV: `nama_kabupaten_kota,tahun` per baris. Tahun harus lebih besar dari tahun maksimum dataset.")

sim_file = st.file_uploader("Upload CSV simulasi (opsional)", type=["csv"], key="simfile")
if sim_file is not None and "trained_model" in st.session_state:
    try:
        df_sim = pd.read_csv(sim_file)
        if not {"nama_kabupaten_kota", "tahun"}.issubset(set(df_sim.columns)):
            st.error("File simulasi harus memiliki kolom: nama_kabupaten_kota,tahun")
        else:
            rows = []
            for _, r in df_sim.iterrows():
                try:
                    tahun_input = int(r["tahun"])
                    nama_input = r["nama_kabupaten_kota"]
                    if tahun_input <= max_year:
                        # Reject backward-year entries in batch; report error per row
                        rows.append({
                            "nama_kabupaten_kota": nama_input,
                            "tahun": tahun_input,
                            "prediksi_jumlah_banjir": None,
                            "error": f"Tahun harus > {max_year}. (ditolak)"
                        })
                        continue

                    Xs = prepare_input_row_for_predict(nama_input, tahun_input, st.session_state["feature_cols"], st.session_state["ohe_cols"], st.session_state["kode_map"], st.session_state["scaler_obj"])
                    p = st.session_state["trained_model"].predict(Xs)[0]
                    rows.append({"nama_kabupaten_kota": nama_input, "tahun": tahun_input, "prediksi_jumlah_banjir": int(round(p))})
                except Exception as ie:
                    rows.append({"nama_kabupaten_kota": r.get("nama_kabupaten_kota"), "tahun": r.get("tahun"), "prediksi_jumlah_banjir": None, "error": str(ie)})
            df_out = pd.DataFrame(rows)
            st.write("Hasil simulasi batch:")
            st.dataframe(df_out)
            csv_out = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download hasil simulasi (CSV)", data=csv_out, file_name="simulasi_prediksi_banjir.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Gagal proses simulasi: {e}")
elif sim_file is not None and "trained_model" not in st.session_state:
    st.error("Model belum tersedia. Latih model atau muat model lokal terlebih dahulu.")

# ============================================================
# Save Model (already implemented via button above)
# ============================================================

# ============================================================
# Catatan akhir / rekomendasi (ditampilkan di app)
# ============================================================
st.write("---")
st.write("**Catatan & rekomendasi:**")
st.write("- Untuk meningkatkan performa, tambahkan fitur eksternal seperti curah hujan, elevasi/topografi, penggunaan lahan, atau data drainase.")
st.write("- Untuk produksi, pisahkan pipeline training & inference, dan gunakan validasi silang (cross-validation).")
st.write("- Simpan versi model setiap kali retraining dan catat metrik evaluasi untuk monitoring.")
