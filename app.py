import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. CONFIGURACIÓN DE PÁGINA
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
# 3. CARGA DE MODELOS
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
    # SVR
    # -----------------------------------------------------

    if not os.path.exists(path_svr):

        st.error(
            "❌ No se encontró el archivo:\n\n"
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
    # GBR
    # -----------------------------------------------------

    if not os.path.exists(path_gbr):

        st.error(
            "❌ No se encontró el archivo:\n\n"
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
# 4. VALIDAR MODELOS
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
            "❌ Al modelo SVR le faltan estas claves:"
        )

        st.write(
            missing_svr
        )

        st.stop()


if gbr_model is not None:

    missing_gbr = [
        key
        for key in required_keys
        if key not in gbr_model
    ]

    if missing_gbr:

        st.error(
            "❌ Al modelo GBR le faltan estas claves:"
        )

        st.write(
            missing_gbr
        )

        st.stop()


# =========================================================
# 5. TÍTULO
# =========================================================

st.title(
    "🧠 Evaluación de Salud Mental"
)

st.write(
    "Ingrese las variables descriptivas. "
    "El sistema utilizará los modelos preentrenados "
    "SVR y GBR para realizar las predicciones."
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
    # 8. CODIFICAR SEXO
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
    # 9. CONSTRUIR DATAFRAME
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
    # 10. OBTENER COLUMNAS DEL MODELO
    # =====================================================

    feature_columns_svr = list(
        svr_model["feature_columns"]
    )

    feature_columns_gbr = list(
        gbr_model["feature_columns"]
    )


    # =====================================================
    # 11. OBTENER COLUMNAS DEL SCALER
    # =====================================================

    scaler_svr = svr_model[
        "scaler"
    ]

    scaler_gbr = gbr_model[
        "scaler"
    ]


    # -----------------------------------------------------
    # Columnas reales utilizadas durante fit()
    # -----------------------------------------------------

    if hasattr(
        scaler_svr,
        "feature_names_in_"
    ):

        scaler_columns_svr = list(
            scaler_svr.feature_names_in_
        )

    else:

        scaler_columns_svr = list(
            svr_model["numeric_columns"]
        )


    if hasattr(
        scaler_gbr,
        "feature_names_in_"
    ):

        scaler_columns_gbr = list(
            scaler_gbr.feature_names_in_
        )

    else:

        scaler_columns_gbr = list(
            gbr_model["numeric_columns"]
        )


    # =====================================================
    # 12. VALIDACIÓN DE COLUMNAS
    # =====================================================

    missing_svr = [
        col
        for col in feature_columns_svr
        if col not in input_data.columns
    ]

    missing_gbr = [
        col
        for col in feature_columns_gbr
        if col not in input_data.columns
    ]


    if missing_svr:

        st.error(
            "❌ Faltan columnas para SVR:"
        )

        st.write(
            missing_svr
        )

        st.stop()


    if missing_gbr:

        st.error(
            "❌ Faltan columnas para GBR:"
        )

        st.write(
            missing_gbr
        )

        st.stop()


    # =====================================================
    # 13. EJECUTAR PREDICCIONES
    # =====================================================
# =====================================================
# 13. EJECUTAR PREDICCIONES
# =====================================================

with st.spinner("Procesando predicción..."):

    try:

        # =================================================
        # ==================== SVR ========================
        # =================================================

        # -------------------------------------------------
        # Crear las 14 variables en el orden exacto
        # utilizado durante el entrenamiento
        # -------------------------------------------------

        X_svr = input_data[
            feature_columns_svr
        ].copy()

        # Asegurar que todas sean numéricas
        X_svr = X_svr.astype(float)

        # -------------------------------------------------
        # ESCALAR SOLO LAS VARIABLES QUE EL SCALER
        # UTILIZÓ DURANTE EL ENTRENAMIENTO
        #
        # IMPORTANTE:
        # Se convierte a NumPy antes de transform().
        # Esto evita el error de feature_names.
        # -------------------------------------------------

        X_svr_scaled = scaler_svr.transform(
            X_svr[
                scaler_columns_svr
            ].to_numpy()
        )

        # Convertir nuevamente las 12 variables escaladas
        # a DataFrame

        X_svr_scaled_df = pd.DataFrame(
            X_svr_scaled,
            columns=scaler_columns_svr,
            index=X_svr.index
        )

        # -------------------------------------------------
        # Recuperar las variables sexo SIN ESCALAR
        # -------------------------------------------------

        for col in feature_columns_svr:

            if col not in scaler_columns_svr:

                X_svr_scaled_df[col] = (
                    X_svr[col].values
                )

        # -------------------------------------------------
        # Volver a colocar las 14 variables en el orden
        # exacto utilizado durante el entrenamiento
        # -------------------------------------------------

        X_svr_final = X_svr_scaled_df[
            feature_columns_svr
        ].copy()

        # Asegurar tipo numérico

        X_svr_final = X_svr_final.astype(float)

        # -------------------------------------------------
        # PREDICCIÓN SVR - ANSIEDAD
        # -------------------------------------------------

        nivel_ansiedad_svr = float(
            svr_model["ansiedad"].predict(
                X_svr_final
            )[0]
        )

        # -------------------------------------------------
        # PREDICCIÓN SVR - ESTRÉS
        # -------------------------------------------------

        nivel_estres_svr = float(
            svr_model["estres"].predict(
                X_svr_final
            )[0]
        )

        # -------------------------------------------------
        # PREDICCIÓN SVR - DEPRESIÓN
        # -------------------------------------------------

        nivel_depresion_svr = float(
            svr_model["depresion"].predict(
                X_svr_final
            )[0]
        )


        # =================================================
        # ==================== GBR ========================
        # =================================================

        # -------------------------------------------------
        # Crear las 14 variables
        # -------------------------------------------------

        X_gbr = input_data[
            feature_columns_gbr
        ].copy()

        # Asegurar que sean numéricas

        X_gbr = X_gbr.astype(float)

        # -------------------------------------------------
        # ESCALAR SOLO LAS VARIABLES NUMÉRICAS
        #
        # También usamos NumPy para evitar cualquier
        # comprobación de nombres por parte de sklearn.
        # -------------------------------------------------

        X_gbr_scaled = scaler_gbr.transform(
            X_gbr[
                scaler_columns_gbr
            ].to_numpy()
        )

        # Convertir las variables escaladas a DataFrame

        X_gbr_scaled_df = pd.DataFrame(
            X_gbr_scaled,
            columns=scaler_columns_gbr,
            index=X_gbr.index
        )

        # -------------------------------------------------
        # Recuperar sexo SIN ESCALAR
        # -------------------------------------------------

        for col in feature_columns_gbr:

            if col not in scaler_columns_gbr:

                X_gbr_scaled_df[col] = (
                    X_gbr[col].values
                )

        # -------------------------------------------------
        # Orden final de las 14 variables
        # -------------------------------------------------

        X_gbr_final = X_gbr_scaled_df[
            feature_columns_gbr
        ].copy()

        X_gbr_final = X_gbr_final.astype(float)

        # -------------------------------------------------
        # PREDICCIÓN GBR - ANSIEDAD
        # -------------------------------------------------

        nivel_ansiedad_gbr = float(
            gbr_model["ansiedad"].predict(
                X_gbr_final
            )[0]
        )

        # -------------------------------------------------
        # PREDICCIÓN GBR - ESTRÉS
        # -------------------------------------------------

        nivel_estres_gbr = float(
            gbr_model["estres"].predict(
                X_gbr_final
            )[0]
        )

        # -------------------------------------------------
        # PREDICCIÓN GBR - DEPRESIÓN
        # -------------------------------------------------

        nivel_depresion_gbr = float(
            gbr_model["depresion"].predict(
                X_gbr_final
            )[0]
        )


    except Exception as e:

        st.error(
            "❌ Error durante la predicción"
        )

        st.exception(e)

        st.stop()
