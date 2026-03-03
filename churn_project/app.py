"""
app.py - Interfaz Streamlit para predicción de Churn de clientes
Ejecutar: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from database import get_engine, get_session, Prediccion, MetricaModelo

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 Predictor de Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "model/churn_model.pkl"

# ── Estilos CSS personalizados ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .churn-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .churn-low {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modelo no encontrado. Ejecuta primero: `python train_model.py`")
        st.stop()
    return joblib.load(MODEL_PATH)

pipeline = cargar_modelo()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔮 Predictor de Abandono de Clientes</h1>
    <p>Sistema de Machine Learning para detectar clientes en riesgo de Churn</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs principales ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predicción Individual", "📊 Análisis & Estadísticas", "📋 Historial de Predicciones"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 - Predicción Individual
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Introduce los datos del cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Información Personal**")
        customer_id     = st.text_input("ID del Cliente", value="NUEVO-0001")
        gender          = st.selectbox("Género", ["Male", "Female"])
        senior_citizen  = st.selectbox("¿Es mayor de 65 años?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        partner         = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents      = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        tenure          = st.slider("Meses como cliente", 0, 72, 12)

    with col2:
        st.markdown("**📞 Servicios Contratados**")
        phone_service    = st.selectbox("Servicio telefónico", ["Yes", "No"])
        multiple_lines   = st.selectbox("Múltiples líneas", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
        device_protection= st.selectbox("Protección dispositivo", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**💳 Contrato & Facturación**")
        tech_support      = st.selectbox("Soporte técnico", ["Yes", "No", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies  = st.selectbox("Streaming películas", ["Yes", "No", "No internet service"])
        contract          = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Factura electrónica", ["Yes", "No"])
        payment_method    = st.selectbox("Método de pago", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges   = st.number_input("Cargo mensual ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges     = st.number_input("Cargo total ($)", 0.0, 10000.0, monthly_charges * tenure, step=1.0)

    st.divider()

    # ── Botón de predicción ──────────────────────────────────────────────────
    if st.button("🔮 Predecir probabilidad de abandono", use_container_width=True, type="primary"):

        input_data = pd.DataFrame([{
            "SeniorCitizen":    senior_citizen,
            "tenure":           tenure,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            "gender":           gender,
            "Partner":          partner,
            "Dependents":       dependents,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protection,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod":    payment_method,
        }])

        prob = pipeline.predict_proba(input_data)[0][1]
        pred = "Yes" if prob >= 0.5 else "No"

        # ── Resultado visual ─────────────────────────────────────────────────
        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            if prob >= 0.5:
                st.markdown(f"""
                <div class="churn-high">
                    ⚠️ RIESGO DE ABANDONO<br>
                    <span style="font-size:2.5rem">{prob:.0%}</span><br>
                    <small>Este cliente probablemente se irá</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="churn-low">
                    ✅ CLIENTE FIDELIZADO<br>
                    <span style="font-size:2.5rem">{prob:.0%}</span><br>
                    <small>Este cliente probablemente se quedará</small>
                </div>""", unsafe_allow_html=True)

        with col_r2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": "Probabilidad de Churn", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 30],  "color": "#55efc4"},
                        {"range": [30, 60], "color": "#ffeaa7"},
                        {"range": [60, 100],"color": "#ff7675"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        # ── Recomendaciones ──────────────────────────────────────────────────
        st.subheader("💡 Recomendaciones")
        recs = []
        if contract == "Month-to-month":
            recs.append("📋 **Ofrecer descuento por contrato anual** — Los contratos mensuales tienen mayor churn.")
        if prob >= 0.5 and tenure < 12:
            recs.append("🎁 **Cliente nuevo en riesgo** — Considera un programa de onboarding o regalo de bienvenida.")
        if monthly_charges > 80:
            recs.append("💰 **Cargo mensual elevado** — Revisar si el cliente usa todos los servicios que paga.")
        if internet_service == "Fiber optic" and online_security == "No":
            recs.append("🔒 **Activar seguridad online** — Los clientes de fibra sin seguridad tienen más churn.")
        if not recs:
            recs.append("✨ **Cliente en buen estado** — Mantener comunicación regular y programa de fidelización.")
        for r in recs:
            st.markdown(r)

        # ── Guardar en BD ────────────────────────────────────────────────────
        try:
            engine  = get_engine()
            session = get_session(engine)
            pred_obj = Prediccion(
                customer_id        = customer_id,
                probabilidad_churn = round(float(prob), 4),
                prediccion         = pred,
                modelo_usado       = "random_forest",
                notas              = f"Contrato: {contract} | Cargo: ${monthly_charges}"
            )
            session.add(pred_obj)
            session.commit()
            session.close()
            st.success(f"✅ Predicción guardada en la base de datos para cliente {customer_id}")
        except Exception as e:
            st.warning(f"No se pudo guardar en BD: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 - Análisis & Estadísticas
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Análisis del Dataset de Entrenamiento")

    data_path = os.path.join("data", "telco_churn.csv")
    if not os.path.exists(data_path):
        st.error("No se encuentra el archivo de datos.")
    else:
        df = pd.read_csv(data_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # KPIs
        total    = len(df)
        churned  = (df["Churn"] == "Yes").sum()
        rate     = churned / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Clientes", f"{total:,}")
        c2.metric("Han Abandonado", f"{churned:,}")
        c3.metric("Tasa de Churn", f"{rate:.1%}")
        c4.metric("Cargo Medio/Mes", f"${df['MonthlyCharges'].mean():.2f}")

        st.divider()
        col_g1, col_g2 = st.columns(2)

        # Churn por tipo de contrato
        with col_g1:
            df_contract = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
            fig1 = px.bar(df_contract, x="Contract", y="count", color="Churn",
                          barmode="group", title="Churn por Tipo de Contrato",
                          color_discrete_map={"Yes": "#ff7675", "No": "#55efc4"})
            st.plotly_chart(fig1, use_container_width=True)

        # Distribución de cargos mensuales
        with col_g2:
            fig2 = px.histogram(df, x="MonthlyCharges", color="Churn", nbins=30,
                                title="Distribución de Cargos Mensuales por Churn",
                                color_discrete_map={"Yes": "#ff7675", "No": "#55efc4"},
                                barmode="overlay", opacity=0.7)
            st.plotly_chart(fig2, use_container_width=True)

        col_g3, col_g4 = st.columns(2)

        # Churn por internet service
        with col_g3:
            df_internet = df.groupby(["InternetService", "Churn"]).size().reset_index(name="count")
            fig3 = px.bar(df_internet, x="InternetService", y="count", color="Churn",
                          barmode="group", title="Churn por Tipo de Internet",
                          color_discrete_map={"Yes": "#ff7675", "No": "#55efc4"})
            st.plotly_chart(fig3, use_container_width=True)

        # Tenure vs Monthly Charges scatter
        with col_g4:
            fig4 = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                              title="Antigüedad vs Cargo Mensual",
                              color_discrete_map={"Yes": "#ff7675", "No": "#55efc4"},
                              opacity=0.6)
            st.plotly_chart(fig4, use_container_width=True)

        # Perfil del cliente que abandona
        st.subheader("🔎 Perfil típico del cliente que abandona")
        churners    = df[df["Churn"] == "Yes"]
        no_churners = df[df["Churn"] == "No"]

        comparison = pd.DataFrame({
            "Métrica":           ["Antigüedad media (meses)", "Cargo mensual medio ($)", "Cargo total medio ($)"],
            "Clientes en Riesgo":  [
                f"{churners['tenure'].mean():.1f}",
                f"${churners['MonthlyCharges'].mean():.2f}",
                f"${churners['TotalCharges'].mean():.2f}",
            ],
            "Clientes Fieles": [
                f"{no_churners['tenure'].mean():.1f}",
                f"${no_churners['MonthlyCharges'].mean():.2f}",
                f"${no_churners['TotalCharges'].mean():.2f}",
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Métricas del modelo
        st.subheader("🤖 Métricas del Modelo Entrenado")
        try:
            engine   = get_engine()
            session  = get_session(engine)
            metricas = session.query(MetricaModelo).order_by(MetricaModelo.fecha.desc()).first()
            session.close()
            if metricas:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",  f"{metricas.accuracy:.3f}")
                mc2.metric("Precision", f"{metricas.precision:.3f}")
                mc3.metric("Recall",    f"{metricas.recall:.3f}")
                mc4.metric("F1 Score",  f"{metricas.f1_score:.3f}")
            else:
                st.info("Entrena el modelo primero: `python train_model.py`")
        except Exception as e:
            st.info(f"Ejecuta `python train_model.py` para ver métricas. ({e})")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 - Historial de Predicciones
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Historial de predicciones realizadas")
    try:
        engine  = get_engine()
        session = get_session(engine)
        preds   = session.query(Prediccion).order_by(Prediccion.fecha_prediccion.desc()).all()
        session.close()

        if preds:
            data_preds = [{
                "ID Cliente":          p.customer_id,
                "Prob. Churn":         f"{p.probabilidad_churn:.1%}",
                "Predicción":          "⚠️ Abandona" if p.prediccion == "Yes" else "✅ Se queda",
                "Modelo":              p.modelo_usado,
                "Fecha":               p.fecha_prediccion.strftime("%Y-%m-%d %H:%M"),
                "Notas":               p.notas or "",
            } for p in preds]
            st.dataframe(pd.DataFrame(data_preds), use_container_width=True, hide_index=True)

            # Mini resumen
            total_pred = len(preds)
            total_churn = sum(1 for p in preds if p.prediccion == "Yes")
            st.info(f"Total predicciones: **{total_pred}** | En riesgo de churn: **{total_churn}** ({total_churn/total_pred:.0%})")
        else:
            st.info("Aún no hay predicciones guardadas. Ve a la pestaña 'Predicción Individual' y predice un cliente.")
    except Exception as e:
        st.error(f"Error consultando BD: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#888'>Proyecto Portfolio · Customer Churn Prediction · ML con Scikit-learn + Streamlit</p>",
    unsafe_allow_html=True
)
