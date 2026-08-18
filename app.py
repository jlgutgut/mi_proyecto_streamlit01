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
# UNPICKLER PERSONALIZADO (Evita errores de módulo al deserializar)
# ---------------------------------------------------------
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirección para objetos GradientBoostingRegressor
        if module == 'GradientBoostingRegressor' or name == 'GradientBoostingRegressor':
            return GradientBoostingRegressor
        # Redirección para objetos SVR
        if module == 'SVR' or name == 'SVR':
            return SVR
        return super().find_class(module, name)

# ---------------------------------------------------------
# FUNCIÓN DE CARGA DE AMBOS MODELOS
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    # Obtiene la ruta absoluta de la carpeta actual donde reside app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(base_dir, 'modelos')
    
    path_gbr = os.path.join(modelos_dir, 'best_gbr_models.pkl')
    path_svr = os.path.join(modelos_dir, 'best_svr_models.pkl')
    
    # Verificar existencia del directorio 'modelos'
    if not os.path.exists(modelos_dir):
        st.error(f"📁 La carpeta de modelos no se encuentra en la ruta: {modelos_dir}")
        return None, None

    pack_gbr = None
    pack_svr = None

    # 1. Cargar modelo GBR
    if os.path.exists(path_gbr):
        try:
            with open(path_gbr, 'rb') as f:
                pack_gbr = CustomUnpickler(f).load()
        except Exception as e:
            st.error(f"❌ Error al deserializar best_gbr_models.pkl: {e}")
    else:
        st.error(f"❌ Archivo no encontrado: {path_gbr}")

    # 2. Cargar modelo SVR
    if os.path.exists(path_svr):
        try:
            with open(path_svr, 'rb') as f:
                pack_svr = CustomUnpickler(f).load()
        except Exception as e:
            st.error(f"❌ Error al deserializar best_svr_models.pkl: {e}")
    else:
        st.error(f"❌ Archivo no encontrado: {path_svr}")

    return pack_gbr, pack_svr

# Cargar ambos paquetes al iniciar la app
pack_gbr, pack_svr = load_all_models()

# ---------------------------------------------------------
# INTERFAZ Y LÓGICA DE PREDICCIÓN
# ---------------------------------------------------------
st.title("🧠 Evaluación Predictiva de Salud Mental")
st.write("Selecciona el algoritmo de Machine Learning que deseas utilizar e ingresa las variables del usuario.")

# Comprobar que ambos paquetes de modelos se hayan cargado con éxito
if pack_gbr is not None and pack_svr is not None:
    
    # Selector de modelo en la pantalla principal
    modelo_seleccionado = st.radio(
        "Algoritmo a utilizar:",
        ["Gradient Boosting (GBR)", "Support Vector Regressor (SVR)"],
        horizontal=True
    )

    # Asignar el paquete de modelos según la selección
    if modelo_seleccionado == "Gradient Boosting (GBR)":
        current_pack = pack_gbr
    else:
        current_pack = pack_svr

    # Extraer los estimadores y el escalador del paquete activo
    model_ansiedad = current_pack['ansiedad']
    model_estres = current_pack['estres']
    model_depresion = current_pack['depresion']
    scaler = current_pack['scaler']

    # --- BARRA LATERAL (ENTRADA DE DATOS) ---
    st.sidebar.header("📋 Formulario de Entrada")
    
    # ⚠️ AJUSTA ESTAS VARIABLES AL ORDEN Y NOMBRES REALES DE TU ENTRENAMIENTO
    edad = st.sidebar.number_input("Edad", min_value=12, max_value=100, value=25)
    horas_sueño = st.sidebar.slider("Horas de sueño diarias", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
    actividad_fisica = st.sidebar.slider("Horas de ejercicio por semana", min_value=0, max_value=20, value=3)
    nivel_estudio_trabajo = st.sidebar.selectbox("Carga de Trabajo/Estudio (1: Baja, 5: Muy Alta)", [1, 2, 3, 4, 5], index=2)
    consumo_cafeina = st.sidebar.slider("Tazas de café/energizantes al día", min_value=0, max_value=10, value=1)

    # Botón para ejecutar la predicción
    if st.sidebar.button("📊 Realizar Predicción", type="primary"):
        
        # 1. Crear DataFrame con las características de entrada
        input_data = pd.DataFrame([{
            'edad': edad,
            'horas_sueño': horas_sueño,
            'actividad_fisica': actividad_fisica,
            'carga_trabajo': nivel_estudio_trabajo,
            'consumo_cafeina': consumo_cafeina
        }])

        # 2. Escalar los datos de entrada
        input_scaled = scaler.transform(input_data)

        # 3. Realizar predicciones con cada modelo
        pred_ansiedad = model_ansiedad.predict(input_scaled)[0]
        pred_estres = model_estres.predict(input_scaled)[0]
        pred_depresion = model_depresion.predict(input_scaled)[0]

        # 4. Mostrar métricas de salida
        st.subheader(f"🎯 Resultados Estimados ({modelo_seleccionado})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Nivel de Ansiedad", value=f"{pred_ansiedad:.2f}")

        with col2:
            st.metric(label="Nivel de Estrés", value=f"{pred_estres:.2f}")

        with col3:
            st.metric(label="Nivel de Depresión", value=f"{pred_depresion:.2f}")

        st.success(f"✅ Predicción ejecutada con éxito utilizando **{modelo_seleccionado}**.")

else:
    st.warning("⚠️ Asegúrate de que los archivos `best_gbr_models.pkl` y `best_svr_models.pkl` estén dentro de la carpeta `modelos/` y que la versión de `scikit-learn` en `requirements.txt` sea la misma con la que entrenaste en Google Colab.")
