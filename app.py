import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuración inicial de la interfaz de Streamlit
st.set_page_config(
    page_title="Evaluación Predictiva de Salud Mental",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. CARGA DE MODELOS Y ESCALADOR DESDE DICCIONARIO NATIVO
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_models():
    """
    Carga los diccionarios .pkl que contienen los estimadores y el StandardScaler.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_svr = os.path.join(base_dir, 'modelos', 'best_svr_models.pkl')
    path_gbr = os.path.join(base_dir, 'modelos', 'best_gbr_models.pkl')
    
    try:
        pack_svr = joblib.load(path_svr)
        pack_gbr = joblib.load(path_gbr)
        return pack_svr, pack_gbr
    except FileNotFoundError as e:
        st.error(f"⚠️ No se encontró el archivo del modelo en la ruta: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error al cargar los archivos de modelo ({type(e).__name__}): {e}")
        return None, None

# Cargar paquetes al iniciar la aplicación
pack_svr, pack_gbr = load_trained_models()

# -----------------------------------------------------------------------------
# 2. INTERFAZ DE USUARIO (FORMULARIO DE ENTRADA)
# -----------------------------------------------------------------------------
st.title("📊 Evaluación Predictiva de Salud Mental")
st.write("Complete la información solicitada a continuación para calcular los niveles estimados de Ansiedad, Estrés y Depresión.")

with st.form("mental_health_form"):
    st.subheader("📋 Datos Demográficos y Hábitos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad", min_value=12, max_value=100, value=25)
        horas_suenho = st.number_input("Horas de Sueño diarias", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        actividad_fisica = st.number_input("Actividad Física (horas/semana)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
        
    with col2:
        apoyo_social = st.slider("Apoyo Social (1-10)", 1, 10, 5)
        eventos_estresantes = st.slider("Eventos Estresantes Recientes (1-10)", 1, 10, 5)
        rumiacion = st.slider("Pensamientos Rumiantes (1-10)", 1, 10, 5)
        autoestima = st.slider("Nivel de Autoestima (1-10)", 1, 10, 5)

    with col3:
        perfeccionismo = st.slider("Nivel de Perfeccionismo (1-10)", 1, 10, 5)
        incertidumbre = st.slider("Tolerancia a la Incertidumbre (1-10)", 1, 10, 5)
        cafeina = st.number_input("Tazas de Cafeína/Café diarias", min_value=0, max_value=20, value=1)
        carga_laboral = st.slider("Carga Laboral / Académica (1-10)", 1, 10, 5)
        responsabilidades_familiares = st.slider("Responsabilidades Familiares (1-10)", 1, 10, 5)

    submit_button = st.form_submit_button("🔍 Realizar Predicción", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. PROCESAMIENTO, ESCALADO Y PREDICCIÓN
# -----------------------------------------------------------------------------
if submit_button:
    if pack_svr is not None and pack_gbr is not None:
        
        # A) Codificación One-Hot para la variable 'Sexo' (drop_first=True)
        sexo_masculino = 1 if sexo == "Masculino" else 0
        sexo_otro = 1 if sexo == "Otro" else 0
            
        # B) Construcción del DataFrame con los datos recibidos
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
        
        with st.spinner("Procesando datos y calculando métricas..."):
            try:
                # C) Escalar los datos de entrada usando el StandardScaler del paquete SVR
                scaler = pack_svr['scaler']
                input_data_scaled = scaler.transform(input_data)
                
                # D) Predicción de Ansiedad y Estrés mediante SVR
                nivel_ansiedad = float(pack_svr['ansiedad'].predict(input_data_scaled)[0])
                nivel_estres = float(pack_svr['estres'].predict(input_data_scaled)[0])

                # E) Predicción de Depresión mediante GBR
                if isinstance(pack_gbr, dict) and 'depresion' in pack_gbr:
                    nivel_depresion = float(pack_gbr['depresion'].predict(input_data_scaled)[0])
                else:
                    # En caso de que el GBR sea un objeto directo no envuelto en diccionario
                    pred_gbr = pack_gbr.predict(input_data_scaled).flatten()
                    nivel_depresion = float(pred_gbr[2]) if len(pred_gbr) >= 3 else float(pred_gbr[0])

                # F) Definición de Umbrales
                ansiedad_threshold, estres_threshold, depresion_threshold = 18, 8, 5
                
                ansiedad_emoji = "😞" if nivel_ansiedad >= ansiedad_threshold else "😊"
                estres_emoji = "😞" if nivel_estres >= estres_threshold else "😊"
                depresion_emoji = "😞" if nivel_depresion >= depresion_threshold else "😊"

                # G) Mostrar Resultados
                st.markdown("---")
                st.subheader("📊 Resultados de la Evaluación")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(label="Ansiedad (Modelo: SVR)", value=f"{nivel_ansiedad:.2f}")
                    status_text = "Elevado" if nivel_ansiedad >= ansiedad_threshold else "Normal"
                    if nivel_ansiedad >= ansiedad_threshold:
                        st.error(f"{ansiedad_emoji} {status_text} (Umbral: {ansiedad_threshold})")
                    else:
                        st.success(f"{ansiedad_emoji} {status_text} (Umbral: {ansiedad_threshold})")
                    
                with res_col2:
                    st.metric(label="Estrés (Modelo: SVR)", value=f"{nivel_estres:.2f}")
                    status_text = "Elevado" if nivel_estres >= estres_threshold else "Normal"
                    if nivel_estres >= estres_threshold:
                        st.error(f"{estres_emoji} {status_text} (Umbral: {estres_threshold})")
                    else:
                        st.success(f"{estres_emoji} {status_text} (Umbral: {estres_threshold})")

                with res_col3:
                    st.metric(label="Depresión (Modelo: GBR)", value=f"{nivel_depresion:.2f}")
                    status_text = "Elevado" if nivel_depresion >= depresion_threshold else "Normal"
                    if nivel_depresion >= depresion_threshold:
                        st.error(f"{depresion_emoji} {status_text} (Umbral: {depresion_threshold})")
                    else:
                        st.success(f"{depresion_emoji} {status_text} (Umbral: {depresion_threshold})")

            except Exception as e:
                st.error(f"⚠️ Error durante el cálculo de la predicción: {e}")
    else:
        st.warning("⚠️ No se pudieron cargar los modelos. Verifique que los archivos `.pkl` estén guardados en la carpeta `modelos/` del repositorio.")
