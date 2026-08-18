import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sys
import sklearn.ensemble
import sklearn.svm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Salud Mental",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 1. FUNCIÓN PARA CARGAR EL MODELO
# ---------------------------------------------------------
@st.cache_resource
def load_trained_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_gbr = os.path.join(base_dir, 'modelos', 'best_gbr_models.pkl')
    
    if not os.path.exists(path_gbr):
        st.error(f"❌ No se encontró el archivo del modelo en: {path_gbr}")
        return None

    try:
        with open(path_gbr, 'rb') as f:
            pack = pickle.load(f)
        return pack
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo .pkl: {e}")
        return None

# Cargar el paquete
pack = load_trained_models()

# ---------------------------------------------------------
# 2. INTERFAZ DE LA APLICACIÓN
# ---------------------------------------------------------
st.title("🧠 Evaluación Predictiva de Salud Mental")
st.write("Ingrese los datos del usuario en el formulario para obtener las estimaciones de Niveles de Ansiedad, Estrés y Depresión.")

if pack is not None:
    # Extraer componentes del paquete
    model_ansiedad = pack['ansiedad']
    model_estres = pack['estres']
    model_depresion = pack['depresion']
    scaler = pack['scaler']

    st.sidebar.header("📋 Formulario de Entrada")

    # --- CAMPOS DE ENTRADA EN LA BARRA LATERAL ---
    # NOTA: Ajusta estos valores y nombres según las variables reales con las que entrenaste el modelo.
    edad = st.sidebar.number_input("Edad", min_value=12, max_value=100, value=25)
    horas_sueño = st.sidebar.slider("Horas de sueño diarias", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
    actividad_fisica = st.sidebar.slider("Horas de ejercicio por semana", min_value=0, max_value=20, value=3)
    nivel_estudio_trabajo = st.sidebar.selectbox("Carga de Trabajo/Estudio (1: Baja, 5: Muy Alta)", [1, 2, 3, 4, 5], index=2)
    consumo_cafeina = st.sidebar.slider("Tazas de café/energizantes al día", min_value=0, max_value=10, value=1)

    # Botón para ejecutar la predicción
    if st.sidebar.button("📊 Realizar Predicción", type="primary"):
        
        # ---------------------------------------------------------
        # 3. PREPARACIÓN Y ESCALADO DE DATOS
        # ---------------------------------------------------------
        # ⚠️ IMPORTANTE: Mantén exactamente el mismo orden de variables que usaste en Colab.
        input_data = pd.DataFrame([{
            'edad': edad,
            'horas_sueño': horas_sueño,
            'actividad_fisica': actividad_fisica,
            'carga_trabajo': nivel_estudio_trabajo,
            'consumo_cafeina': consumo_cafeina
        }])

        # Escalar los datos con el StandardScaler cargado
        input_scaled = scaler.transform(input_data)

        # ---------------------------------------------------------
        # 4. GENERAR PREDICCIONES
        # ---------------------------------------------------------
        pred_ansiedad = model_ansiedad.predict(input_scaled)[0]
        pred_estres = model_estres.predict(input_scaled)[0]
        pred_depresion = model_depresion.predict(input_scaled)[0]

        # ---------------------------------------------------------
        # 5. MOSTRAR RESULTADOS
        # ---------------------------------------------------------
        st.subheader("🎯 Resultados Estimados")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Nivel de Ansiedad", value=f"{pred_ansiedad:.2f}")
            st.progress(min(max(float(pred_ansiedad) / 100.0, 0.0), 1.0))

        with col2:
            st.metric(label="Nivel de Estrés", value=f"{pred_estres:.2f}")
            st.progress(min(max(float(pred_estres) / 100.0, 0.0), 1.0))

        with col3:
            st.metric(label="Nivel de Depresión", value=f"{pred_depresion:.2f}")
            st.progress(min(max(float(pred_depresion) / 100.0, 0.0), 1.0))

        # Cuadro informativo de resumen
        st.success("✅ Predicción completada utilizando modelos Gradient Boosting Regressor.")

else:
    st.warning("⚠️ No se pudo inicializar la aplicación porque los modelos no se cargaron correctamente.")
