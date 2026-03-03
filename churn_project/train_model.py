"""
train_model.py - Entrena y guarda el modelo de predicción de Churn
Ejecutar una sola vez: python train_model.py
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from database import get_engine, get_session, MetricaModelo

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "telco_churn.csv")

def cargar_datos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df

# ── 2. Limpiar datos ─────────────────────────────────────────────────────────
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges viene como string en algunos casos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Eliminar columna ID (no aporta información)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Target: 1 = Churn, 0 = No Churn
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"✅ Datos limpios. Churn rate: {df['Churn'].mean():.1%}")
    return df

# ── 3. Preparar features ─────────────────────────────────────────────────────
COLUMNAS_NUMERICAS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

COLUMNAS_CATEGORICAS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

def construir_pipeline(modelo_tipo: str = "random_forest") -> Pipeline:
    preprocesador = ColumnTransformer(transformers=[
        ("num", StandardScaler(), COLUMNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUMNAS_CATEGORICAS),
    ])

    if modelo_tipo == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", clf)
    ])
    return pipeline

# ── 4. Entrenar ──────────────────────────────────────────────────────────────
def entrenar(df: pd.DataFrame, modelo_tipo: str = "random_forest"):
    X = df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = construir_pipeline(modelo_tipo)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
    }

    print("\n📊 MÉTRICAS DEL MODELO")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall   : {metrics['recall']:.3f}")
    print(f"  F1 Score : {metrics['f1_score']:.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return pipeline, metrics

# ── 5. Guardar modelo ────────────────────────────────────────────────────────
def guardar_modelo(pipeline, path: str = "model/churn_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Modelo guardado en: {path}")

# ── 6. Guardar métricas en BD ────────────────────────────────────────────────
def guardar_metricas_bd(metrics: dict, modelo_tipo: str):
    engine = get_engine()
    session = get_session(engine)
    registro = MetricaModelo(
        modelo    = modelo_tipo,
        accuracy  = metrics["accuracy"],
        precision = metrics["precision"],
        recall    = metrics["recall"],
        f1_score  = metrics["f1_score"],
    )
    session.add(registro)
    session.commit()
    session.close()
    print("✅ Métricas guardadas en la base de datos.")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODELO_TIPO = "random_forest"   # Cambia a "logistic" si prefieres

    df = cargar_datos(DATA_PATH)
    df = limpiar_datos(df)
    pipeline, metrics = entrenar(df, MODELO_TIPO)
    guardar_modelo(pipeline)
    guardar_metricas_bd(metrics, MODELO_TIPO)

    print("\n🎉 Entrenamiento completado. Ya puedes lanzar: streamlit run app.py")
