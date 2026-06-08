import os
import pandas as pd
from sklearn.datasets import fetch_openml

print("Memulai proses unduh dataset German Credit...")
# Mengambil dataset dari repository publik OpenML
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
df = credit_data.frame

# Memastikan folder dataset_raw tersedia
os.makedirs('dataset_raw', exist_ok=True)

# Menyimpan dataframe menjadi file CSV murni
file_path = os.path.join('dataset_raw', 'credit_scoring_raw.csv')
df.to_csv(file_path, index=False)

print(f"Berhasil! Dataset (1000 baris, 21 kolom) telah disimpan di: {file_path}")
