# ===============================================================
# IMPORT
# ===============================================================
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

# ===============================================================
# LOAD DATASET
# ===============================================================
st.title("🌧️ Prediksi Jumlah Kejadian Banjir - Provinsi Jawa Barat")

data_path = "bpbd-od_17600_jml_kejadian_bencana_banjir__kabupatenkota_v3_data.csv"

df = pd.read_csv(data_path)

st.write("### Dataset Awal")
st.dataframe(df)

# ===============================================================
# ENCODING
# ===============================================================
st.subheader("🔧 Proses Encoding")

# buat kode unik kabupaten/kota
unique_kab = sorted(df["nama_kabupaten_kota"].unique())
kode_map = {name: i for i, name in enumerate(unique_kab)}
df["kode_kabupaten_kota"] = df["nama_kabupaten_kota"].map(kode_map)

# one-hot encoding
df_ohe = pd.get_dummies(df["nama_kabupaten_kota"], prefix="kab")
df = pd.concat([df, df_ohe], axis=1)

st.write("### Data Setelah Encoding")
st.dataframe(df)

# ===============================================================
# SPLIT FITUR & TARGET
# ===============================================================
target_col = "jumlah_kejadian"
feature_cols = ["kode_kabupaten_kota", "tahun"] + list(df_ohe.columns)

X = df[feature_cols]
y = df[target_col]

# ===============================================================
# SCALING
# ===============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================================
# SPLIT TRAIN-TEST
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================================================
# TRAIN MODEL
# ===============================================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
)

model.fit(X_train, y_train)

# ===============================================================
# EVALUASI
# ===============================================================
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📊 Evaluasi Model")
st.write(f"**MSE:** {mse:.4f}")

# simpan ke session_state
st.session_state["trained_model"] = model
st.session_state["scaler_obj"] = scaler
st.session_state["feature_cols"] = feature_cols
st.session_state["ohe_cols"] = list(df_ohe.columns)
st.session_state["kode_map"] = kode_map

max_year = df["tahun"].max()

# ===============================================================
# FITUR PREDIKSI SATUAN (manual)
# ===============================================================
st.subheader("🎯 Prediksi Jumlah Banjir (1 Tahun)")

nama_kab_single = st.selectbox("Pilih Kabupaten/Kota", unique_kab, key="single")
tahun_pred_single = st.number_input("Tahun prediksi:", min_value=2000, max_value=2100, value=max_year+1)

if st.button("Prediksi Satu Tahun"):
    x_row = {c: 0 for c in feature_cols}
    x_row["kode_kabupaten_kota"] = kode_map[nama_kab_single]
    x_row["tahun"] = tahun_pred_single

    # pasang one-hot
    for c in df_ohe.columns:
        x_row[c] = 1 if c.endswith(nama_kab_single) else 0

    df_input = pd.DataFrame([x_row], columns=feature_cols)
    df_scaled = scaler.transform(df_input)

    hasil = model.predict(df_scaled)[0]

    st.success(f"Prediksi Jumlah Kejadian Banjir Tahun {tahun_pred_single}: **{hasil:.2f}**")


# ===============================================================
# PREDIKSI 1–3 TAHUN KE DEPAN (MULTI-STEP FORECASTING)
# ===============================================================
st.subheader("🔮 Prediksi Multi-Tahun (1 - 3 Tahun ke Depan)")

nama_kab_multi = st.selectbox("Pilih Kabupaten/Kota:", unique_kab, key="multi")
start_year = max_year

if st.button("Prediksi 1-3 Tahun ke Depan"):
    model = st.session_state["trained_model"]
    scaler_obj = st.session_state["scaler_obj"]
    feature_cols = st.session_state["feature_cols"]
    ohe_cols = st.session_state["ohe_cols"]
    kode_map = st.session_state["kode_map"]

    results = {}

    current_year = start_year

    for step in [1, 2, 3]:
        tahun_prediksi = current_year + step

        x_row = {c: 0 for c in feature_cols}
        x_row["kode_kabupaten_kota"] = kode_map[nama_kab_multi]
        x_row["tahun"] = tahun_prediksi

        # isi one-hot
        for c in ohe_cols:
            x_row[c] = 1 if c.endswith(nama_kab_multi) else 0

        df_input = pd.DataFrame([x_row], columns=feature_cols)
        df_scaled = scaler_obj.transform(df_input)

        pred = model.predict(df_scaled)[0]
        results[tahun_prediksi] = round(pred, 2)

    st.success(f"Prediksi 1-3 Tahun ke Depan untuk {nama_kab_multi}:")
    st.write("### 📘 Hasil Prediksi:")
    st.write(results)

# ===============================================================
# SELESAI
# ===============================================================
st.info("Selesai — aplikasi siap digunakan.")
