from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import os

app = FastAPI(title="Credit Scoring API")

# Setup otomatisasi metrik untuk Prometheus
Instrumentator().instrument(app).expose(app)

# Mencari model dari folder MLProject (karena tadi kita simpan di sana/menggunakan file bersih)
# Catatan: Di dunia nyata ini menggunakan MLflow URI, tapi untuk contoh ini kita buat mock simpel agar API bisa menyala
@app.get("/")
def home():
    return {"message": "Credit Scoring API is Running!"}

@app.post("/predict")
def predict():
    # Ini adalah endpoint simulasi prediksi
    return {"prediction": "Approved", "confidence": 0.85}

if __name__ == "__main__":
    print("Menjalankan server di http://127.0.0.1:8000")
    print("Metrik Prometheus tersedia di http://127.0.0.1:8000/metrics")
    uvicorn.run(app, host="127.0.0.1", port=8000)