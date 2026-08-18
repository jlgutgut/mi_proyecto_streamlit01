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
# 3. CARGAR LOS MODELOS
# =========================================================

@st.cache_resource
def load_trained_models():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    modelos_dir = os.path.join(
        base_dir,
        "modelos"
    )

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

    # -----------------------------------------------------
    # CARGAR SVR
    # -----------------------------------------------------

    if not os.path.exists(path_svr):

        st.error(
            f"❌ No se encontró el archivo SVR:\n\n"
            f"{path_svr}"
        )

    else:

        try:

            svr_model = joblib.load(
                path_svr
            )

            st.success(
                "✅ Modelo SVR cargado correctamente"
            )

        except Exception as e:

            st.error(
                "❌ Error cargando SVR"
            )

            st.exception(e)


    # -----------------------------------------------------
    # CARGAR GBR
    # -----------------------------------------------------

    if not os.path.exists(path_gbr):

        st.error(
            f"❌ No se encontró el archivo GBR:\n\n"
            f"{path_gbr}"
        )

    else:

        try:

            gbr_model = joblib.load(
                path_gbr
            )

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
# 4. VALIDAR ESTRUCTURA DE LOS MODELOS
# =========================================================

required_keys = [
    "ansiedad",
    "estres",
    "depresion",
    "scaler",
    "feature_columns",
    "numeric_columns"
]


if svr_model is not None:

    missing_svr = [
        key
        for key in required_keys
        if key not in svr_model
    ]

    if missing_svr:

        st.error(
            "❌ Al modelo SVR le faltan las siguientes claves:"
        )

        st.write(
            missing_svr
        )


if gbr_model is not None:

    missing_gbr = [
        key
        for key in required_keys
        if key not in gbr_model
    ]

    if missing_gbr:

        st.error(
            "❌ Al modelo GBR le faltan las siguientes claves:"
        )

        st.write(
            missing_gbr
        )


# =========================================================
# 5. TÍTULO
# =========================================================

st.title(
    "🧠 Evaluación de Salud Mental"
)

st.write(
    "Ingrese las variables descriptivas. "
    "El sistema utilizará modelos SVR y GBR "
    "preentrenados para realizar las predicciones."
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
# 7. EJECUTAR PREDICCIÓN
# =========================================================

if submit_button:

    # -----------------------------------------------------
    # Verificar modelos
    # -----------------------------------------------------

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
    # 9. CONSTRUIR DATOS DE ENTRADA
    # =====================================================

    raw_data = {

        "edad":
            edad,

        "horas_suenho":
            horas_suenho,

        "actividad_fisica":
            actividad_fisica,

        "apoyo_social":
            apoyo_social,

        "eventos_estresantes":
            eventos_estresantes,

        "rumiacion":
            rumiacion,

        "autoestima":
            autoestima,

        "perfeccionismo":
            perfeccionismo,

        "incertidumbre":
            incertidumbre,

        "cafeina":
            cafeina,

        "carga_laboral":
            carga_laboral,

        "responsabilidades_familiares":
            responsabilidades_familiares,

        "sexo_masculino":
            sexo_masculino,

        "sexo_otro":
            sexo_otro
    }


    input_data = pd.DataFrame(
        [raw_data]
    )


    # =====================================================
    # 10. OBTENER COLUMNAS EXACTAS DEL ENTRENAMIENTO
    # =====================================================

    feature_columns_svr = svr_model[
        "feature_columns"
    ]

    feature_columns_gbr = gbr_model[
        "feature_columns"
    ]


    # =====================================================
    # 11. COMPLETAR COLUMNAS SVR
    # =====================================================

    for column in feature_columns_svr:

        if column not in input_data.columns:

            input_data[column] = 0


    # =====================================================
    # 12. COMPLETAR COLUMNAS GBR
    # =====================================================

    for column in feature_columns_gbr:

        if column not in input_data.columns:

            input_data[column] = 0


    # =====================================================
    # 13. PREDICCIÓN
    # =====================================================

    with st.spinner(
        "Procesando predicción..."
    ):

        try:

            # =================================================
            # SVR
            # =================================================

            scaler_svr = svr_model[
                "scaler"
            ]


            # -----------------------------------------------
            # Obtener las columnas que REALMENTE recibió
            # el StandardScaler durante el fit
            # -----------------------------------------------

            if hasattr(
                scaler_svr,
                "feature_names_in_"
            ):

                scaler_columns_svr = list(
                    scaler_svr.feature_names_in_
                )

            else:

                scaler_columns_svr = list(
                    svr_model[
                        "numeric_columns"
                    ]
                )


            # Crear dataframe SVR
            X_svr = input_data[
                feature_columns_svr
            ].copy()


            # Verificar columnas necesarias

            for column in scaler_columns_svr:

                if column not in X_svr.columns:

                    X_svr[column] = 0


            # Mantener el orden exacto del scaler

            X_to_scale_svr = X_svr[
                scaler_columns_svr
            ].copy()


            # Escalar

            X_scaled_svr = (
                scaler_svr.transform(
                    X_to_scale_svr
                )
            )


            # Reemplazar las columnas escaladas

            X_svr[
                scaler_columns_svr
            ] = X_scaled_svr


            # -----------------------------------------------
            # Predicciones SVR
            # -----------------------------------------------

            nivel_ansiedad_svr = float(
                svr_model[
                    "ansiedad"
                ].predict(X_svr)[0]
            )


            nivel_estres_svr = float(
                svr_model[
                    "estres"
                ].predict(X_svr)[0]
            )


            nivel_depresion_svr = float(
                svr_model[
                    "depresion"
                ].predict(X_svr)[0]
            )


            # =================================================
            # GBR
            # =================================================

            scaler_gbr = gbr_model[
                "scaler"
            ]


            # -----------------------------------------------
            # Obtener columnas reales del scaler GBR
            # -----------------------------------------------

            if hasattr(
                scaler_gbr,
                "feature_names_in_"
            ):

                scaler_columns_gbr = list(
                    scaler_gbr.feature_names_in_
                )

            else:

                scaler_columns_gbr = list(
                    gbr_model[
                        "numeric_columns"
                    ]
                )


            # Crear dataframe GBR

            X_gbr = input_data[
                feature_columns_gbr
            ].copy()


            # Verificar columnas necesarias

            for column in scaler_columns_gbr:

                if column not in X_gbr.columns:

                    X_gbr[column] = 0


            # Mantener orden exacto

            X_to_scale_gbr = X_gbr[
                scaler_columns_gbr
            ].copy()


            # Escalar

            X_scaled_gbr = (
                scaler_gbr.transform(
                    X_to_scale_gbr
                )
            )


            # Reemplazar columnas escaladas

            X_gbr[
                scaler_columns_gbr
            ] = X_scaled_gbr


            # -----------------------------------------------
            # Predicciones GBR
            # -----------------------------------------------

            nivel_ansiedad_gbr = float(
                gbr_model[
                    "ansiedad"
                ].predict(X_gbr)[0]
            )


            nivel_estres_gbr = float(
                gbr_model[
                    "estres"
                ].predict(X_gbr)[0]
            )


            nivel_depresion_gbr = float(
                gbr_model[
                    "depresion"
                ].predict(X_gbr)[0]
            )


        except Exception as e:

            st.error(
                "❌ Error durante la predicción"
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # 14. RESULTADOS PRINCIPALES
    # =====================================================

    # Modelo elegido para cada indicador:
    #
    # Ansiedad  -> SVR
    # Estrés    -> SVR
    # Depresión -> GBR

    nivel_ansiedad = (
        nivel_ansiedad_svr
    )

    nivel_estres = (
        nivel_estres_svr
    )

    nivel_depresion = (
        nivel_depresion_gbr
    )


    # =====================================================
    # 15. UMBRALES
    # =====================================================

    ansiedad_threshold = 18

    estres_threshold = 8

    depresion_threshold = 5


    # =====================================================
    # 16. DETERMINAR ESTADOS
    # =====================================================

    ansiedad_alta = (
        nivel_ansiedad >=
        ansiedad_threshold
    )

    estres_alto = (
        nivel_estres >=
        estres_threshold
    )

    depresion_alta = (
        nivel_depresion >=
        depresion_threshold
    )


    # =====================================================
    # 17. EMOJIS
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
    # 18. MOSTRAR RESULTADOS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📊 Resultados de la Evaluación"
    )


    res_col1, res_col2, res_col3 = (
        st.columns(3)
    )


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
    # 19. COMPARACIÓN SVR VS GBR
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🔎 Comparación SVR vs GBR"
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


    comparacion[
        "SVR"
    ] = comparacion[
        "SVR"
    ].round(2)


    comparacion[
        "GBR"
    ] = comparacion[
        "GBR"
    ].round(2)


    st.dataframe(
        comparacion,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # 20. INFORMACIÓN TÉCNICA
    # =====================================================

    with st.expander(
        "🔧 Información técnica de la predicción"
    ):

        st.write(
            "Columnas utilizadas por SVR:"
        )

        st.write(
            feature_columns_svr
        )


        st.write(
            "Columnas utilizadas por GBR:"
        )

        st.write(
            feature_columns_gbr
        )


        st.write(
            "Columnas escaladas por SVR:"
        )

        st.write(
            scaler_columns_svr
        )


        st.write(
            "Columnas escaladas por GBR:"
        )

        st.write(
            scaler_columns_gbr
        )


# =========================================================
# 21. BARRA LATERAL
# =========================================================

with st.sidebar:

    st.header(
        "ℹ️ Información"
    )

    st.write(
        "**Modelos utilizados:**"
    )

    st.write(
        "• SVR para ansiedad"
    )

    st.write(
        "• SVR para estrés"
    )

    st.write(
        "• GBR para depresión"
    )

    st.write(
        "Los modelos fueron entrenados "
        "previamente en Google Colab."
    )

    st.write(
        "La aplicación utiliza los mismos "
        "modelos y escaladores almacenados "
        "en los archivos `.pkl`."
    )

    st.caption(
        "Los resultados son predictivos y "
        "no constituyen un diagnóstico clínico."
    )
