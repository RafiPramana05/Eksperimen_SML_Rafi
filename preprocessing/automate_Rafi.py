import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Memuat data mentah."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Melakukan imputasi dan pembersihan dasar."""
    # Mengisi nilai kosong dengan median untuk numerik
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Mengisi nilai kosong kategorik dengan modus
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def preprocess_data(df):
    """Melakukan encoding dan scaling."""
    # Label Encoding untuk kolom kategorik
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def main():
    # Menentukan path yang solid menggunakan os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'dataset_raw', 'credit_scoring_raw.csv')
    processed_dir = os.path.join(base_dir, 'preprocessing', 'dataset_preprocessing')
    processed_path = os.path.join(processed_dir, 'credit_scoring_clean.csv')

    # Eksekusi fungsi
    print("Memulai proses automasi preprocessing...")
    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_final = preprocess_data(df_clean)

    # Memastikan direktori output tersedia dan menyimpan hasil
    os.makedirs(processed_dir, exist_ok=True)
    df_final.to_csv(processed_path, index=False)
    print(f"Data bersih berhasil disimpan di: {processed_path}")

if __name__ == "__main__":
    main()