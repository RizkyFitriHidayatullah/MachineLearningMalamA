import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Prediksi Banjir Jawa Barat", layout="wide")
st.title("🌊 Prediksi Jumlah Kejadian Banjir di Jawa Barat")
st.write("Sumber data: [Open Data Jabar](https://opendata.jabarprov.go.id/id)")

# === Upload Dataset ===
uploaded_file = st.file_uploader("📤 Upload file CSV dari Open Data Jabar", type=["csv"])

if uploaded_file:
    # === Baca Dataset ===
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "nama_kabupaten_kota": "KabupatenKota",
        "jumlah_banjir": "JumlahBanjir",
        "tahun": "Tahun"
    })

    df = df.dropna(subset=["KabupatenKota", "JumlahBanjir", "Tahun"])
    df["Tahun"] = df["Tahun"].astype(int)
    df["JumlahBanjir"] = df["JumlahBanjir"].astype(float)

    st.subheader("📋 Data Banjir Jawa Barat")
    st.dataframe(df.head(10))

    kabupaten_list = sorted(df["KabupatenKota"].unique())
    kabupaten_pilih = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)

    data_kab = df[df["KabupatenKota"] == kabupaten_pilih]

    X = data_kab[["Tahun"]]
    y = data_kab["JumlahBanjir"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    tahun_pred = st.number_input("Masukkan Tahun Prediksi", min_value=int(df["Tahun"].min()), max_value=2030, value=2025)
    prediksi = model.predict([[tahun_pred]])[0]

    st.subheader(f"📈 Prediksi Jumlah Banjir di {kabupaten_pilih}")
    st.write(f"Tahun {tahun_pred}: **{prediksi:.2f} kejadian**")
    st.write(f"Mean Absolute Error: `{mae:.2f}` | R² Score: `{r2:.2f}`")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(X, y, color='blue', label='Data Aktual')
    ax.plot(X, model.predict(X), color='red', label='Regresi Linear')
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Kejadian Banjir")
    ax.set_title(f"Tren Banjir di {kabupaten_pilih}")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu untuk memulai analisis.")
