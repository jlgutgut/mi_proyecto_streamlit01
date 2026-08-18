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

# Inyectar las clases en sys.modules para que pickle las encuentre
sys.modules['GradientBoostingRegressor'] = GradientBoostingRegressor
sys.modules['StandardScaler'] = StandardScaler
sys.modules['SVR'] = SVR

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Salud Mental | SVR & GBR",
    page_icon="🧠",
    layout="wide"
)

# Estilos CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-alert { padding: 12px; border-radius: 8px; font-weight: 600; text-align: center; margin-top: 10px; }
    .status-high { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .status-low { background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UNPICKLER PERSONALIZADO COMPATIBLE
# ---------------------------------------------------------
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Mapeos directos para evitar el fallo de atributos/dtype sobre clases
        if name == 'StandardScaler' or module == 'StandardScaler':
            return StandardScaler
        if name == 'GradientBoostingRegressor' or module == 'GradientBoostingRegressor':
            return GradientBoostingRegressor
        if name == 'SVR' or module == 'SVR':
            return SVR
        
        # Mapeos de submódulos internos de scikit-learn
        if module == 'sklearn.preprocessing._data' and name == 'StandardScaler':
            return StandardScaler
        if module == 'sklearn.ensemble._gb' and name == 'GradientBoostingRegressor':
            return GradientBoostingRegressor
            
        return super().find_class(module, name)

# ---------------------------------------------------------
# FUNCIÓN DE CARGA
# ---------------------------------------------------------
@st.cache_resource
def load_trained_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(base_dir, 'modelos')
    
    path_svr = os.path.join(modelos_dir, 'best_svr_models.pkl')
    path_gbr = os.path.join(modelos_dir, 'best_gbr_models.pkl')
    
    pack_svr, pack_gbr = None, None

    if os.path.exists(path_svr):
        try:
            with open(path_svr, 'rb') as f:
                pack_svr = CustomUnpickler(f).load()
        except Exception as e:
            st.error(f"❌ Error al cargar best_svr_models.pkl: {e}")

    if os.path.exists(path_gbr):
        try:
            with open(path_gbr, 'rb') as f:
                pack_gbr = CustomUnpickler(f).load()
        except Exception as e:
            st.error(f"❌ Error al cargar best_gbr_models.pkl: {e}")

    return pack_svr, pack_gbr


pack_svr, pack_gbr = load_trained_models()

# ---------------------------------------------------------
# 3. INTERFAZ PRINCIPAL
# ---------------------------------------------------------
st.title("🧠 Evaluación de Salud Mental (SVR & GBR)")
st.write("Ingrese las variables descriptivas. El sistema usará los modelos preentrenados para calcular la predicción.")

# ---------------------------------------------------------
# 4. FORMULARIO DE ENTRADA
# ---------------------------------------------------------
with st.form(key="mental_health_form"):
    st.subheader("📋 Ingreso de Predictores")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("1. Edad", min_value=10, max_value=100, value=25)
        sexo = st.selectbox("2. Sexo", options=["Masculino", "Femenino", "Otro"])
        horas_suenho = st.number_input("3. Horas de Sueño", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        actividad_fisica = st.number_input("4. Actividad Física (Días)", min_value=0, max_value=7, value=3)
        apoyo_social = st.number_input("5. Apoyo Social", min_value=0, max_value=10, value=7)
        
    with col2:
        eventos_estresantes = st.number_input("6. Eventos Estresantes", min_value=0, max_value=20, value=2)
        rumiacion = st.number_input("7. Rumiación", min_value=0, max_value=10, value=4)
        autoestima = st.number_input("8. Autoestima", min_value=0, max_value=10, value=6)
        perfeccionismo = st.number_input("9. Perfeccionismo", min_value=0, max_value=10, value=5)
        
    with col3:
        incertidumbre = st.number_input("10. Incertidumbre", min_value=0, max_value=10, value=4)
        cafeina = st.number_input("11. Cafeína", min_value=0, max_value=20, value=2)
        carga_laboral = st.number_input("12. Carga Laboral", min_value=0, max_value=10, value=6)
        responsabilidades_familiares = st.number_input("13. Responsabilidades Fam.", min_value=0, max_value=10, value=4)

    submit_button = st.form_submit_button(label="🚀 Ejecutar Diagnóstico Predictivo", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. EJECUCIÓN E INFERENCIA DE LOS MODELOS
# ---------------------------------------------------------
if submit_button:
    if pack_svr is not None and pack_gbr is not None:
        
        # A) LÓGICA DE ONE-HOT ENCODING
        if sexo == "Masculino":
            sexo_masculino = 1
            sexo_otro = 0
        elif sexo == "Femenino":
            sexo_masculino = 0
            sexo_otro = 0
        else: # "Otro"
            sexo_masculino = 0
            sexo_otro = 1
            
        # B) CONSTRUCCIÓN DEL DATAFRAME DE ENTRADA
        input_data = pd.DataFrame([{
            'edad': edad,
            'horas_suenho': horas_suenho,
            'actividad_fisica': actividad_fisica,
            'apoyo_social': apoyo_social,
            'eventos_estresantes': eventos_estresantes,
            'rumiacion': rumiacion,
            'autoestima': autoestima,
            'perfeccionismo': perfeccionismo,
            'incertidumbre': incertidumbre,
            'cafeina': cafeina,
            'carga_laboral': carga_laboral,
            'responsabilidades_familiares': responsabilidades_familiares,
            'sexo_masculino': sexo_masculino,
            'sexo_otro': sexo_otro
        }])
        
        with st.spinner("Procesando datos con los paquetes SVR y GBR..."):
            
            # C) EXTRAER MODELOS Y SCALER DESDE LOS DICCIONARIOS
            scaler_svr = pack_svr['scaler']
            model_ansiedad = pack_svr['ansiedad']
            model_estres = pack_svr['estres']
            
            scaler_gbr = pack_gbr['scaler']
            model_depresion = pack_gbr['depresion']
            
            # D) ESCALADO DE DATOS E INFERENCIA
            input_scaled_svr = scaler_svr.transform(input_data)
            nivel_ansiedad = model_ansiedad.predict(input_scaled_svr)[0]
            nivel_estres = model_estres.predict(input_scaled_svr)[0]
            
            input_scaled_gbr = scaler_gbr.transform(input_data)
            nivel_depresion = model_depresion.predict(input_scaled_gbr)[0]
            
            # E) REGLAS Y UMBRALES
            ansiedad_threshold, estres_threshold, depresion_threshold = 18, 8, 5
            
            ansiedad_emoji = "😞" if nivel_ansiedad >= ansiedad_threshold else "😊"
            estres_emoji = "😞" if nivel_estres >= estres_threshold else "😊"
            depresion_emoji = "😞" if nivel_depresion >= depresion_threshold else "😊"
        
        # F) MOSTRAR RESULTADOS
        st.markdown("---")
        st.subheader("📊 Resultados de la Evaluación")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Ansiedad (SVR)", value=f"{nivel_ansiedad:.2f}")
            status_class = "status-high" if nivel_ansiedad >= ansiedad_threshold else "status-low"
            status_text = "Elevado" if nivel_ansiedad >= ansiedad_threshold else "Normal"
            st.markdown(f'<div class="status-alert {status_class}">{ansiedad_emoji} {status_text} (Umbral: {ansiedad_threshold})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Estrés (SVR)", value=f"{nivel_estres:.2f}")
            status_class = "status-high" if nivel_estres >= estres_threshold else "status-low"
            status_text = "Elevado" if nivel_estres >= estres_threshold else "Normal"
            st.markdown(f'<div class="status-alert {status_class}">{estres_emoji} {status_text} (Umbral: {estres_threshold})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Depresión (GBR)", value=f"{nivel_depresion:.2f}")
            status_class = "status-high" if nivel_depresion >= depresion_threshold else "status-low"
            status_text = "Elevado" if nivel_depresion >= depresion_threshold else "Normal"
            st.markdown(f'<div class="status-alert {status_class}">{depresion_emoji} {status_text} (Umbral: {depresion_threshold})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ No se pudieron procesar las predicciones debido a un error al cargar los paquetes `.pkl`.")
