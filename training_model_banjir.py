# ===============================================================
# TRAINING MODEL - RANDOM FOREST (FIX: dataset BPBD JABAR)
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# ---------------------------------------------------------
# LOAD DATASET (Nama EXACT sesuai dataset kamu)
# ---------------------------------------------------------
data = pd.read_csv("bpbd-od_17600_jml_kejadian_bencana_banjir__kabupatenkota_v3_data.csv")

# ---------------------------------------------------------
# CEK 5 BARIS PERTAMA
# ---------------------------------------------------------
print("Contoh data:")
print(data.head())

# ---------------------------------------------------------
# PILIH FITUR & LABEL
# ---------------------------------------------------------
# Fitur minimal: tahun + kode kabupaten
X = data[["tahun", "kode_kabupaten_kota"]]

# Label: jumlah kejadian banjir
y = data["jumlah_banjir"]

# ---------------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# TRAIN RANDOM FOREST
# ---------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# EVALUASI
# ---------------------------------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("\nTraining selesai!")
print("MSE :", mse)
print("RMSE:", mse ** 0.5)

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
with open("model_banjir.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel berhasil disimpan → model_banjir.pkl")
