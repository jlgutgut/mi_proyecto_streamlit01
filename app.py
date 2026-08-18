import os
import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Evaluación de Salud Mental",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. CARGA DE MODELOS (Con gestión de caché)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_models():
    """
    Carga los archivos .pkl de la carpeta 'modelos'.
    Maneja tanto modelos simples como diccionarios de estimadores.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_svr = os.path.join(base_dir, 'modelos', 'best_svr_models.pkl')
    path_gbr = os.path.join(base_dir, 'modelos', 'best_gbr_models.pkl')
    
    try:
        loaded_svr = joblib.load(path_svr)
        loaded_gbr = joblib.load(path_gbr)
        return loaded_svr, loaded_gbr
    except FileNotFoundError as e:
        st.error(f"⚠️ No se encontró el archivo del modelo: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error al cargar los modelos ({type(e).__name__}): {e}")
        return None, None

# Cargar modelos al iniciar la app
svr_model, gbr_model = load_trained_models()

# -----------------------------------------------------------------------------
# 2. INTERFAZ DE USUARIO (FORMULARIO)
# -----------------------------------------------------------------------------
st.title("📊 Evaluación Predictiva de Salud Mental")
st.write("Complete el siguiente formulario para estimar los niveles de Ansiedad, Estrés y Depresión.")

with st.form("mental_health_form"):
    st.subheader("📋 Datos del Usuario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad", min_value=12, max_value=100, value=25)
        horas_suenho = st.number_input("Horas de Sueño al día", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        actividad_fisica = st.number_input("Actividad Física (horas/semana)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
        
    with col2:
        apoyo_social = st.slider("Apoyo Social (1-10)", 1, 10, 5)
        eventos_estresantes = st.slider("Eventos Estresantes (1-10)", 1, 10, 5)
        rumiacion = st.slider("Rumiación Pensamientos (1-10)", 1, 10, 5)
        autoestima = st.slider("Nivel de Autoestima (1-10)", 1, 10, 5)

    with col3:
        perfeccionismo = st.slider("Nivel de Perfeccionismo (1-10)", 1, 10, 5)
        incertidumbre = st.slider("Tolerancia a la Incertidumbre (1-10)", 1, 10, 5)
        cafeina = st.number_input("Tazas de Café/Cafeína al día", min_value=0, max_value=20, value=1)
        carga_laboral = st.slider("Carga Laboral / Académica (1-10)", 1, 10, 5)
        responsabilidades_familiares = st.slider("Carga Familiar (1-10)", 1, 10, 5)

    submit_button = st.form_submit_button("🔍 Calcular Predicciones", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. PROCESAMIENTO Y PREDICCIÓN
# -----------------------------------------------------------------------------
if submit_button:
    if svr_model is not None and gbr_model is not None:
        
        # A) Codificación One-Hot Encoding para Categoría 'Sexo' (drop_first=True)
        sexo_masculino = 1 if sexo == "Masculino" else 0
        sexo_otro = 1 if sexo == "Otro" else 0
            
        # B) Construcción del DataFrame de Entrada
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
        
        with st.spinner("Ejecutando inferencia con los modelos entrenados..."):
            try:
                # --- PREDICCIÓN CON SVR (Ansiedad y Estrés) ---
                if isinstance(svr_model, dict):
                    nivel_ansiedad = float(svr_model['ansiedad'].predict(input_data)[0])
                    nivel_estres = float(svr_model['estres'].predict(input_data)[0])
                elif hasattr(svr_model, "estimators_"):
                    # Si es un MultiOutputRegressor
                    pred_svr = svr_model.predict(input_data).flatten()
                    nivel_ansiedad = float(pred_svr[1])
                    nivel_estres = float(pred_svr[2])
                else:
                    pred_svr = svr_model.predict(input_data).flatten()
                    nivel_ansiedad = float(pred_svr[0])
                    nivel_estres = float(pred_svr[1]) if len(pred_svr) > 1 else 0.0

                # --- PREDICCIÓN CON GBR (Depresión) ---
                if isinstance(gbr_model, dict):
                    nivel_depresion = float(gbr_model['depresion'].predict(input_data)[0])
                elif hasattr(gbr_model, "estimators_"):
                    pred_gbr = gbr_model.predict(input_data).flatten()
                    nivel_depresion = float(pred_gbr[0])
                else:
                    pred_gbr = gbr_model.predict(input_data).flatten()
                    nivel_depresion = float(pred_gbr[0])

                # C) Umbrales y Alertas
                ansiedad_threshold, estres_threshold, depresion_threshold = 18, 8, 5
                
                ansiedad_emoji = "😞" if nivel_ansiedad >= ansiedad_threshold else "😊"
                estres_emoji = "😞" if nivel_estres >= estres_threshold else "😊"
                depresion_emoji = "😞" if nivel_depresion >= depresion_threshold else "😊"

                # D) Mostrar Resultados en Paneles
                st.markdown("---")
                st.subheader("📊 Resultados de la Evaluación")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(label="Ansiedad (SVR)", value=f"{nivel_ansiedad:.2f}")
                    status_text = "Elevado" if nivel_ansiedad >= ansiedad_threshold else "Normal"
                    if nivel_ansiedad >= ansiedad_threshold:
                        st.error(f"{ansiedad_emoji} {status_text} (Umbral: {ansiedad_threshold})")
                    else:
                        st.success(f"{ansiedad_emoji} {status_text} (Umbral: {ansiedad_threshold})")
                    
                with res_col2:
                    st.metric(label="Estrés (SVR)", value=f"{nivel_estres:.2f}")
                    status_text = "Elevado" if nivel_estres >= estres_threshold else "Normal"
                    if nivel_estres >= estres_threshold:
                        st.error(f"{estres_emoji} {status_text} (Umbral: {estres_threshold})")
                    else:
                        st.success(f"{estres_emoji} {status_text} (Umbral: {estres_threshold})")

                with res_col3:
                    st.metric(label="Depresión (GBR)", value=f"{nivel_depresion:.2f}")
                    status_text = "Elevado" if nivel_depresion >= depresion_threshold else "Normal"
                    if nivel_depresion >= depresion_threshold:
                        st.error(f"{depresion_emoji} {status_text} (Umbral: {depresion_threshold})")
                    else:
                        st.success(f"{depresion_emoji} {status_text} (Umbral: {depresion_threshold})")

            except Exception as e:
                st.error(f"⚠️ Error durante la inferencia: {e}")
    else:
        st.warning("⚠️ No se pudieron cargar los modelos de predicción. Verifique los archivos en la carpeta 'modelos'.")
