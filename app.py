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

@st.cache_resource
def load_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    path_gbr = os.path.join(base_dir, 'modelos', 'best_gbr_models.joblib')
    path_svr = os.path.join(base_dir, 'modelos', 'best_svr_models.joblib')

    pack_gbr = joblib.load(path_gbr) if os.path.exists(path_gbr) else None
    pack_svr = joblib.load(path_svr) if os.path.exists(path_svr) else None

    return pack_gbr, pack_svr 


# Cargar ambos paquetes
pack_gbr, pack_svr = load_all_models()

# ---------------------------------------------------------
# INTERFAZ Y SELECCIÓN DE MODELO
# ---------------------------------------------------------
st.title("🧠 Evaluación Predictiva de Salud Mental")

if pack_gbr is not None and pack_svr is not None:
    
    # Permite al usuario elegir qué algoritmo desea utilizar
    modelo_seleccionado = st.radio(
        "Seleccione el algoritmo de predicción:",
        ["Gradient Boosting (GBR)", "Support Vector Regressor (SVR)"],
        horizontal=True
    )

    # Asignar el paquete según la elección
    if modelo_seleccionado == "Gradient Boosting (GBR)":
        current_pack = pack_gbr
    else:
        current_pack = pack_svr

    # Extraer componentes del paquete seleccionado
    model_ansiedad = current_pack['ansiedad']
    model_estres = current_pack['estres']
    model_depresion = current_pack['depresion']
    scaler = current_pack['scaler']

    # --- BARRA LATERAL CON FORMULARIO ---
    st.sidebar.header("📋 Formulario de Entrada")
    edad = st.sidebar.number_input("Edad", min_value=12, max_value=100, value=25)
    horas_sueño = st.sidebar.slider("Horas de sueño diarias", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
    actividad_fisica = st.sidebar.slider("Horas de ejercicio por semana", min_value=0, max_value=20, value=3)
    nivel_estudio_trabajo = st.sidebar.selectbox("Carga de Trabajo/Estudio (1: Baja, 5: Muy Alta)", [1, 2, 3, 4, 5], index=2)
    consumo_cafeina = st.sidebar.slider("Tazas de café/energizantes al día", min_value=0, max_value=10, value=1)

    if st.sidebar.button("📊 Realizar Predicción", type="primary"):
        # Crear DataFrame de entrada
        input_data = pd.DataFrame([{
            'edad': edad,
            'horas_sueño': horas_sueño,
            'actividad_fisica': actividad_fisica,
            'carga_trabajo': nivel_estudio_trabajo,
            'consumo_cafeina': consumo_cafeina
        }])

        # Escalar los datos con el scaler correspondiente al modelo seleccionado
        input_scaled = scaler.transform(input_data)

        # Generar predicciones
        pred_ansiedad = model_ansiedad.predict(input_scaled)[0]
        pred_estres = model_estres.predict(input_scaled)[0]
        pred_depresion = model_depresion.predict(input_scaled)[0]

        # Mostrar Resultados
        st.subheader(f"🎯 Resultados Estimados ({modelo_seleccionado})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Nivel de Ansiedad", value=f"{pred_ansiedad:.2f}")

        with col2:
            st.metric(label="Nivel de Estrés", value=f"{pred_estres:.2f}")

        with col3:
            st.metric(label="Nivel de Depresión", value=f"{pred_depresion:.2f}")

        st.success(f"✅ Predicción ejecutada con éxito utilizando {modelo_seleccionado}.")

else:
    st.warning("⚠️ Asegúrate de que ambos archivos (`best_gbr_models.pkl` y `best_svr_models.pkl`) estén presentes en la carpeta `/modelos/`.")
