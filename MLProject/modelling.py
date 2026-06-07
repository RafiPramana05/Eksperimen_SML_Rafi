import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import dagshub

# Inisialisasi otomatis ke DagsHub
dagshub.init(repo_owner='RafiPramana05', repo_name='mlsystem-studi-kasus-cs', mlflow=True)

# Autologging aktif
mlflow.sklearn.autolog()

df = pd.read_csv('credit_scoring_clean.csv')
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Basic berhasil di-training dan dicatat ke DagsHub.")