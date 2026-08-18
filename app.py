import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Predicción de Salud Mental",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# 2. ESTILOS
# =========================================================

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

.status-alert {
    padding: 12px;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
    margin-top: 10px;
}

.status-high {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

.status-low {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #6ee7b7;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CARGAR MODELOS
# =========================================================

@st.cache_resource
def load_trained_models():

    # Directorio donde está app.py
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Directorio modelos
    modelos_dir = os.path.join(
        base_dir,
        "modelos"
    )

    # Archivos
    path_svr = os.path.join(
        modelos_dir,
        "best_svr_models.pkl"
    )

    path_gbr = os.path.join(
        modelos_dir,
        "best_gbr_models.pkl"
    )

    svr_model = None
    gbr_model = None


    # =====================================================
    # CARGAR SVR
    # =====================================================

    if not os.path.exists(path_svr):

        st.error(
            "❌ No se encontró el archivo:\n\n"
            f"{path_svr}"
        )

    else:

        try:

            # IMPORTANTE:
            # usamos joblib.load()
            svr_model = joblib.load(path_svr)

            st.success(
                "✅ Modelo SVR cargado correctamente"
            )

        except Exception as e:

            st.error(
                "❌ Error cargando SVR"
            )

            st.exception(e)


    # =====================================================
    # CARGAR GBR
    # =====================================================

    if not os.path.exists(path_gbr):

        st.error(
            "❌ No se encontró el archivo:\n\n"
            f"{path_gbr}"
        )

    else:

        try:

            # IMPORTANTE:
            # usamos joblib.load()
            gbr_model = joblib.load(path_gbr)

            st.success(
                "✅ Modelo GBR cargado correctamente"
            )

        except Exception as e:

            st.error(
                "❌ Error cargando GBR"
            )

            st.exception(e)


    return svr_model, gbr_model


# Cargar modelos
svr_model, gbr_model = load_trained_models()


# =========================================================
# 4. VERIFICAR ESTRUCTURA DE LOS MODELOS
# =========================================================

if svr_model is not None:

    required_keys_svr = [
        "scaler",
        "ansiedad",
        "estres",
        "depresion"
    ]

    missing_svr = [
        key
        for key in required_keys_svr
        if key not in svr_model
    ]

    if missing_svr:

        st.error(
            "❌ El archivo SVR no contiene "
            f"las claves esperadas: {missing_svr}"
        )


if gbr_model is not None:

    required_keys_gbr = [
        "scaler",
        "ansiedad",
        "estres",
        "depresion"
    ]

    missing_gbr = [
        key
        for key in required_keys_gbr
        if key not in gbr_model
    ]

    if missing_gbr:

        st.error(
            "❌ El archivo GBR no contiene "
            f"las claves esperadas: {missing_gbr}"
        )


# =========================================================
# 5. TÍTULO
# =========================================================

st.title(
    "🧠 Evaluación de Salud Mental"
)

st.write(
    "Ingrese las variables descriptivas. "
    "El sistema utilizará modelos preentrenados "
    "para calcular los niveles de ansiedad, estrés "
    "y depresión."
)


# =========================================================
# 6. FORMULARIO
# =========================================================

with st.form(
    key="mental_health_form"
):

    st.subheader(
        "📋 Ingreso de Predictores"
    )

    col1, col2, col3 = st.columns(3)


    # =====================================================
    # COLUMNA 1
    # =====================================================

    with col1:

        edad = st.number_input(
            "1. Edad",
            min_value=10,
            max_value=100,
            value=25
        )

        sexo = st.selectbox(
            "2. Sexo",
            options=[
                "Masculino",
                "Femenino",
                "Otro"
            ]
        )

        horas_suenho = st.number_input(
            "3. Horas de Sueño",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        actividad_fisica = st.number_input(
            "4. Actividad Física (Días)",
            min_value=0,
            max_value=7,
            value=3
        )

        apoyo_social = st.number_input(
            "5. Apoyo Social",
            min_value=0,
            max_value=10,
            value=7
        )


    # =====================================================
    # COLUMNA 2
    # =====================================================

    with col2:

        eventos_estresantes = st.number_input(
            "6. Eventos Estresantes",
            min_value=0,
            max_value=20,
            value=2
        )

        rumiacion = st.number_input(
            "7. Rumiación",
            min_value=0,
            max_value=10,
            value=4
        )

        autoestima = st.number_input(
            "8. Autoestima",
            min_value=0,
            max_value=10,
            value=6
        )

        perfeccionismo = st.number_input(
            "9. Perfeccionismo",
            min_value=0,
            max_value=10,
            value=5
        )


    # =====================================================
    # COLUMNA 3
    # =====================================================

    with col3:

        incertidumbre = st.number_input(
            "10. Incertidumbre",
            min_value=0,
            max_value=10,
            value=4
        )

        cafeina = st.number_input(
            "11. Cafeína",
            min_value=0,
            max_value=20,
            value=2
        )

        carga_laboral = st.number_input(
            "12. Carga Laboral",
            min_value=0,
            max_value=10,
            value=6
        )

        responsabilidades_familiares = st.number_input(
            "13. Responsabilidades Fam.",
            min_value=0,
            max_value=10,
            value=4
        )


    # =====================================================
    # BOTÓN
    # =====================================================

    submit_button = st.form_submit_button(
        label="🚀 Ejecutar Diagnóstico Predictivo",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# 7. PREDICCIÓN
# =========================================================

if submit_button:

    # Verificar que ambos modelos existan

    if svr_model is None:

        st.error(
            "❌ El modelo SVR no está disponible."
        )

        st.stop()


    if gbr_model is None:

        st.error(
            "❌ El modelo GBR no está disponible."
        )

        st.stop()


    # =====================================================
    # 8. CODIFICACIÓN DEL SEXO
    # =====================================================

    if sexo == "Masculino":

        sexo_masculino = 1
        sexo_otro = 0

    elif sexo == "Femenino":

        sexo_masculino = 0
        sexo_otro = 0

    else:

        sexo_masculino = 0
        sexo_otro = 1


    # =====================================================
    # 9. CREAR DATAFRAME
    # =====================================================

    input_data = pd.DataFrame([{

        "edad": edad,

        "horas_suenho": horas_suenho,

        "actividad_fisica": actividad_fisica,

        "apoyo_social": apoyo_social,

        "eventos_estresantes": eventos_estresantes,

        "rumiacion": rumiacion,

        "autoestima": autoestima,

        "perfeccionismo": perfeccionismo,

        "incertidumbre": incertidumbre,

        "cafeina": cafeina,

        "carga_laboral": carga_laboral,

        "responsabilidades_familiares":
            responsabilidades_familiares,

        "sexo_masculino":
            sexo_masculino,

        "sexo_otro":
            sexo_otro

    }])


    # =====================================================
    # 10. REALIZAR PREDICCIONES
    # =====================================================

    with st.spinner(
        "Procesando predicción..."
    ):

        try:

            # =================================================
            # SVR
            # =================================================

            scaler_svr = svr_model["scaler"]

            datos_svr = scaler_svr.transform(
                input_data
            )


            nivel_ansiedad_svr = (
                svr_model["ansiedad"]
                .predict(datos_svr)[0]
            )

            nivel_estres_svr = (
                svr_model["estres"]
                .predict(datos_svr)[0]
            )

            nivel_depresion_svr = (
                svr_model["depresion"]
                .predict(datos_svr)[0]
            )


            # =================================================
            # GBR
            # =================================================

            scaler_gbr = gbr_model["scaler"]

            datos_gbr = scaler_gbr.transform(
                input_data
            )


            nivel_ansiedad_gbr = (
                gbr_model["ansiedad"]
                .predict(datos_gbr)[0]
            )

            nivel_estres_gbr = (
                gbr_model["estres"]
                .predict(datos_gbr)[0]
            )

            nivel_depresion_gbr = (
                gbr_model["depresion"]
                .predict(datos_gbr)[0]
            )


            # =================================================
            # MODELOS PRINCIPALES
            # =================================================
            #
            # Según tu aplicación:
            #
            # Ansiedad -> SVR
            # Estrés   -> SVR
            # Depresión -> GBR
            #

            nivel_ansiedad = float(
                nivel_ansiedad_svr
            )

            nivel_estres = float(
                nivel_estres_svr
            )

            nivel_depresion = float(
                nivel_depresion_gbr
            )


        except Exception as e:

            st.error(
                "❌ Error durante la predicción"
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # 11. UMBRALES
    # =====================================================

    ansiedad_threshold = 18

    estres_threshold = 8

    depresion_threshold = 5


    # =====================================================
    # 12. ESTADOS
    # =====================================================

    ansiedad_alta = (
        nivel_ansiedad >= ansiedad_threshold
    )

    estres_alto = (
        nivel_estres >= estres_threshold
    )

    depresion_alta = (
        nivel_depresion >= depresion_threshold
    )


    # =====================================================
    # 13. EMOJIS
    # =====================================================

    ansiedad_emoji = (
        "😞"
        if ansiedad_alta
        else "😊"
    )

    estres_emoji = (
        "😞"
        if estres_alto
        else "😊"
    )

    depresion_emoji = (
        "😞"
        if depresion_alta
        else "😊"
    )


    # =====================================================
    # 14. RESULTADOS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📊 Resultados de la Evaluación"
    )


    res_col1, res_col2, res_col3 = st.columns(3)


    # =====================================================
    # ANSIEDAD
    # =====================================================

    with res_col1:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            label="Ansiedad (SVR)",
            value=f"{nivel_ansiedad:.2f}"
        )

        status_class = (
            "status-high"
            if ansiedad_alta
            else "status-low"
        )

        status_text = (
            "Elevado"
            if ansiedad_alta
            else "Normal"
        )

        st.markdown(
            f"""
            <div class="status-alert {status_class}">
                {ansiedad_emoji}
                {status_text}
                (Umbral: {ansiedad_threshold})
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # ESTRÉS
    # =====================================================

    with res_col2:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            label="Estrés (SVR)",
            value=f"{nivel_estres:.2f}"
        )

        status_class = (
            "status-high"
            if estres_alto
            else "status-low"
        )

        status_text = (
            "Elevado"
            if estres_alto
            else "Normal"
        )

        st.markdown(
            f"""
            <div class="status-alert {status_class}">
                {estres_emoji}
                {status_text}
                (Umbral: {estres_threshold})
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # DEPRESIÓN
    # =====================================================

    with res_col3:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            label="Depresión (GBR)",
            value=f"{nivel_depresion:.2f}"
        )

        status_class = (
            "status-high"
            if depresion_alta
            else "status-low"
        )

        status_text = (
            "Elevado"
            if depresion_alta
            else "Normal"
        )

        st.markdown(
            f"""
            <div class="status-alert {status_class}">
                {depresion_emoji}
                {status_text}
                (Umbral: {depresion_threshold})
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # 15. COMPARACIÓN SVR vs GBR
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🔎 Comparación de modelos"
    )

    comparacion = pd.DataFrame({

        "Variable": [
            "Ansiedad",
            "Estrés",
            "Depresión"
        ],

        "SVR": [
            nivel_ansiedad_svr,
            nivel_estres_svr,
            nivel_depresion_svr
        ],

        "GBR": [
            nivel_ansiedad_gbr,
            nivel_estres_gbr,
            nivel_depresion_gbr
        ]

    })

    comparacion["SVR"] = comparacion["SVR"].round(2)

    comparacion["GBR"] = comparacion["GBR"].round(2)

    st.dataframe(
        comparacion,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 16. INFORMACIÓN DEL SISTEMA
# =========================================================

with st.sidebar:

    st.header(
        "ℹ️ Información"
    )

    st.write(
        "Sistema de predicción basado "
        "en modelos de Machine Learning."
    )

    st.write(
        "**Modelos utilizados:**"
    )

    st.write(
        "- SVR para ansiedad"
    )

    st.write(
        "- SVR para estrés"
    )

    st.write(
        "- GBR para depresión"
    )

    st.caption(
        "Los resultados son predictivos y "
        "no constituyen un diagnóstico clínico."
    )
