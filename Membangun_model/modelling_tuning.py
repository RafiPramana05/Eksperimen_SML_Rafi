import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os

# Hubungkan ke DagsHub
dagshub.init(repo_owner='RafiPramana05', repo_name='mlsystem-studi-kasus-cs', mlflow=True)

def main():
    # RADAR PENCARI FOLDER CERDAS
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Posisi file modelling_tuning.py
    base_dir = os.path.dirname(current_dir) # Mundur 1 langkah ke Eksperimen_SML_Rafi
    
    # Arahkan langsung ke markas data bersih
    csv_path = os.path.join(base_dir, 'preprocessing', 'dataset_preprocessing', 'credit_scoring_clean.csv')
    print(f"Mengambil data dari: {csv_path}")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # 2. Hyperparameter Tuning (GridSearchCV)
        print("Memulai proses training dan tuning (Harap tunggu sebentar)...")
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # 3. Hitung Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 4. Manual Logging Parameter & Metric (Syarat Advanced)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 5. Artefak 1: Gambar Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Tuned Model")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(current_dir, "training_confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # 6. Artefak 2: File JSON Metric Info
        metrics_dict = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
        json_path = os.path.join(current_dir, "metric_info.json")
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f)
        mlflow.log_artifact(json_path)

        # 7. Log Model Utamanya
        mlflow.sklearn.log_model(best_model, "model")

        print("\n🎉 Luar Biasa! Model Tuning dan 2 Artefak tambahan berhasil dikirim ke DagsHub!")

if __name__ == "__main__":
    main()