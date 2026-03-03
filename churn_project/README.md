# 🔮 Customer Churn Predictor — Guía de instalación paso a paso

## 📁 Estructura del proyecto

```
churn_project/
│
├── data/
│   └── telco_churn.csv        # Dataset de Telco Churn
│
├── model/                     # Se crea automáticamente al entrenar
│   └── churn_model.pkl        # Modelo entrenado (joblib)
│
├── database.py                # Modelos SQLAlchemy + setup SQLite
├── train_model.py             # Script de entrenamiento
├── app.py                     # Interfaz Streamlit
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

---

## ⚙️ PASO 1 — Verificar que tienes Python instalado

Abre la **Terminal de VS Code** (`Ctrl + ñ` o `Terminal > New Terminal`) y escribe:

```bash
python --version
```

Necesitas **Python 3.10 o superior**. Si no tienes Python:
- Descárgalo de https://www.python.org/downloads/
- Durante la instalación, marca ✅ **"Add Python to PATH"**

---

## ⚙️ PASO 2 — Crear un entorno virtual (recomendado)

En la terminal de VS Code, desde la carpeta del proyecto:

```bash
# Crear entorno virtual
python -m venv venv

# Activarlo en Windows:
venv\Scripts\activate

# Activarlo en Mac/Linux:
source venv/bin/activate
```

Verás `(venv)` al inicio de la línea → el entorno está activo ✅

---

## ⚙️ PASO 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instala: pandas, scikit-learn, streamlit, plotly, sqlalchemy, joblib...

Tardará unos 2-3 minutos. Es normal.

---

## ⚙️ PASO 4 — Entrenar el modelo

```bash
python train_model.py
```

Verás algo así:

```
✅ Dataset cargado: 50 filas × 21 columnas
✅ Datos limpios. Churn rate: 34.0%
📊 MÉTRICAS DEL MODELO
  Accuracy : 0.800
  F1 Score : 0.750
✅ Modelo guardado en: model/churn_model.pkl
✅ Métricas guardadas en la base de datos.
🎉 Entrenamiento completado.
```

Se crea automáticamente la carpeta `model/` con el modelo entrenado.

---

## ⚙️ PASO 5 — Lanzar la aplicación

```bash
streamlit run app.py
```

El navegador se abrirá automáticamente en:
```
http://localhost:8501
```

---

## 🔥 USO DE LA APP

### Pestaña 1 — Predicción Individual
1. Rellena los datos del cliente (contrato, cargos, servicios...)
2. Haz clic en **"Predecir probabilidad de abandono"**
3. Verás el porcentaje de riesgo con un gauge visual
4. La predicción se guarda automáticamente en SQLite

### Pestaña 2 — Análisis & Estadísticas
- Gráficas de churn por tipo de contrato, internet, cargos
- Perfil del cliente típico que abandona
- Métricas del modelo (Accuracy, F1, etc.)

### Pestaña 3 — Historial
- Todas las predicciones guardadas en la base de datos

---

## ❓ Problemas comunes

### Error: "streamlit no se reconoce como comando"
```bash
# Solución: instalar con pip y usar python -m
python -m streamlit run app.py
```

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "Modelo no encontrado"
```bash
# Asegúrate de haber ejecutado primero:
python train_model.py
```

### Error: Puerto 8501 ocupado
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Dataset completo (Kaggle)

Para más datos reales (7.000 filas), descarga el dataset oficial:
👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Descarga `WA_Fn-UseC_-Telco-Customer-Churn.csv` y reemplaza el archivo en `data/telco_churn.csv`.

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.10+ |
| Modelo ML | Scikit-learn (RandomForest) |
| Interfaz | Streamlit |
| Base de datos | SQLite + SQLAlchemy |
| Gráficas | Plotly |
| Serialización | Joblib |

---

*Proyecto Portfolio — Customer Churn Prediction*
