import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. CONFIGURACIÓN
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
def load_models():

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

    svr = None
    gbr = None

    # -----------------------------------------------------
    # SVR
    # -----------------------------------------------------

    try:

        if not os.path.exists(path_svr):

            st.error(
                f"❌ No existe el archivo SVR:\n{path_svr}"
            )

        else:

            svr = joblib.load(path_svr)

            st.success(
                "✅ SVR cargado correctamente"
            )

    except Exception as e:

        st.error("❌ Error cargando SVR")
        st.exception(e)


    # -----------------------------------------------------
    # GBR
    # -----------------------------------------------------

    try:

        if not os.path.exists(path_gbr):

            st.error(
                f"❌ No existe el archivo GBR:\n{path_gbr}"
            )

        else:

            gbr = joblib.load(path_gbr)

            st.success(
                "✅ GBR cargado correctamente"
            )

    except Exception as e:

        st.error("❌ Error cargando GBR")
        st.exception(e)


    return svr, gbr


svr_model, gbr_model = load_models()


# =========================================================
# 4. VALIDACIÓN
# =========================================================

if svr_model is None or gbr_model is None:

    st.error(
        "❌ No se pudieron cargar ambos modelos."
    )

    st.stop()


# =========================================================
# 5. OBTENER LAS COLUMNAS REALES DE CADA MODELO
# =========================================================

svr_ansiedad = svr_model["ansiedad"]
svr_estres = svr_model["estres"]
svr_depresion = svr_model["depresion"]

gbr_ansiedad = gbr_model["ansiedad"]
gbr_estres = gbr_model["estres"]
gbr_depresion = gbr_model["depresion"]


# =========================================================
# COLUMNAS QUE REALMENTE RECIBIÓ CADA MODELO DURANTE FIT()
# =========================================================

svr_columns = list(
    svr_ansiedad.feature_names_in_
)

gbr_columns = list(
    gbr_ansiedad.feature_names_in_
)


# =========================================================
# COLUMNAS DEL SCALER
# =========================================================

svr_scaler = svr_model["scaler"]
gbr_scaler = gbr_model["scaler"]


if hasattr(
    svr_scaler,
    "feature_names_in_"
):

    svr_scaler_columns = list(
        svr_scaler.feature_names_in_
    )

else:

    svr_scaler_columns = list(
        svr_model["numeric_columns"]
    )


if hasattr(
    gbr_scaler,
    "feature_names_in_"
):

    gbr_scaler_columns = list(
        gbr_scaler.feature_names_in_
    )

else:

    gbr_scaler_columns = list(
        gbr_model["numeric_columns"]
    )


# =========================================================
# INFORMACIÓN TÉCNICA
# =========================================================

with st.sidebar:

    st.header("🔧 Información técnica")

    st.write(
        "**SVR recibió durante entrenamiento:**"
    )

    st.write(svr_columns)

    st.write(
        "**GBR recibió durante entrenamiento:**"
    )

    st.write(gbr_columns)

    st.write(
        "**Scaler SVR recibió:**"
    )

    st.write(svr_scaler_columns)

    st.write(
        "**Scaler GBR recibió:**"
    )

    st.write(gbr_scaler_columns)

    st.markdown("---")

    st.write(
        f"SVR: {len(svr_columns)} variables"
    )

    st.write(
        f"GBR: {len(gbr_columns)} variables"
    )

    st.write(
        f"Scaler SVR: {len(svr_scaler_columns)} variables"
    )

    st.write(
        f"Scaler GBR: {len(gbr_scaler_columns)} variables"
    )


# =========================================================
# 6. TÍTULO
# =========================================================

st.title(
    "🧠 Evaluación de Salud Mental"
)

st.write(
    "Ingrese las variables descriptivas para "
    "realizar la predicción."
)


# =========================================================
# 7. FORMULARIO
# =========================================================

with st.form(
    "mental_health_form"
):

    st.subheader(
        "📋 Ingreso de Predictores"
    )

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # COLUMNA 1
    # -----------------------------------------------------

    with col1:

        edad = st.number_input(
            "1. Edad",
            min_value=10,
            max_value=100,
            value=25
        )

        sexo = st.selectbox(
            "2. Sexo",
            [
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


    # -----------------------------------------------------
    # COLUMNA 2
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # COLUMNA 3
    # -----------------------------------------------------

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


    submit_button = st.form_submit_button(
        "🚀 Ejecutar Diagnóstico Predictivo",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# 8. PREDICCIÓN
# =========================================================

if submit_button:

    # =====================================================
    # CODIFICACIÓN DE SEXO
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
    # DATAFRAME COMPLETO DE ENTRADA
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
    # MOSTRAR ENTRADA
    # =====================================================

    with st.expander(
        "🔍 Ver datos ingresados"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # =====================================================
    # EJECUTAR PREDICCIÓN
    # =====================================================

    with st.spinner(
        "Procesando predicciones..."
    ):

        try:

            # =================================================
            # =================================================
            #                       SVR
            # =================================================
            # =================================================

            # -------------------------------------------------
            # SVR fue entrenado SOLO con 12 columnas
            # -------------------------------------------------

            X_svr = input_data[
                svr_columns
            ].copy()


            # -------------------------------------------------
            # IMPORTANTE:
            #
            # El scaler SVR también fue entrenado con esas
            # mismas 12 columnas.
            # -------------------------------------------------

            X_svr_scaled = svr_scaler.transform(
                X_svr
            )


            # -------------------------------------------------
            # Convertir nuevamente a DataFrame
            # -------------------------------------------------

            X_svr_scaled = pd.DataFrame(
                X_svr_scaled,
                columns=svr_columns,
                index=X_svr.index
            )


            # -------------------------------------------------
            # Predicciones SVR
            # -------------------------------------------------

            nivel_ansiedad_svr = float(
                svr_ansiedad.predict(
                    X_svr_scaled
                )[0]
            )

            nivel_estres_svr = float(
                svr_estres.predict(
                    X_svr_scaled
                )[0]
            )

            nivel_depresion_svr = float(
                svr_depresion.predict(
                    X_svr_scaled
                )[0]
            )


            # =================================================
            # =================================================
            #                       GBR
            # =================================================
            # =================================================

            # -------------------------------------------------
            # GBR fue entrenado con 14 columnas
            # -------------------------------------------------

            X_gbr = input_data[
                gbr_columns
            ].copy()


            # -------------------------------------------------
            # El scaler GBR solo escaló 12 columnas
            # -------------------------------------------------

            X_gbr[
                gbr_scaler_columns
            ] = gbr_scaler.transform(
                X_gbr[
                    gbr_scaler_columns
                ]
            )


            # -------------------------------------------------
            # Predicciones GBR
            # -------------------------------------------------

            nivel_ansiedad_gbr = float(
                gbr_ansiedad.predict(
                    X_gbr
                )[0]
            )

            nivel_estres_gbr = float(
                gbr_estres.predict(
                    X_gbr
                )[0]
            )

            nivel_depresion_gbr = float(
                gbr_depresion.predict(
                    X_gbr
                )[0]
            )


        except Exception as e:

            st.error(
                "❌ Error durante la predicción"
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # 9. RESULTADOS PRINCIPALES
    # =====================================================

    # Tu aplicación utiliza:
    #
    # Ansiedad  → SVR
    # Estrés    → SVR
    # Depresión → GBR

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
    # UMBRALES
    # =====================================================

    ansiedad_threshold = 18
    estres_threshold = 8
    depresion_threshold = 5


    # =====================================================
    # ESTADOS
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
    # RESULTADOS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📊 Resultados de la Evaluación"
    )

    res1, res2, res3 = st.columns(3)


    # =====================================================
    # ANSIEDAD
    # =====================================================

    with res1:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Ansiedad (SVR)",
            f"{nivel_ansiedad:.2f}"
        )

        if ansiedad_alta:

            st.markdown(
                f"""
                <div class="status-alert status-high">
                    😞 Elevado
                    (Umbral: {ansiedad_threshold})
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="status-alert status-low">
                    😊 Normal
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

    with res2:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Estrés (SVR)",
            f"{nivel_estres:.2f}"
        )

        if estres_alto:

            st.markdown(
                f"""
                <div class="status-alert status-high">
                    😞 Elevado
                    (Umbral: {estres_threshold})
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="status-alert status-low">
                    😊 Normal
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

    with res3:

        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.metric(
            "Depresión (GBR)",
            f"{nivel_depresion:.2f}"
        )

        if depresion_alta:

            st.markdown(
                f"""
                <div class="status-alert status-high">
                    😞 Elevado
                    (Umbral: {depresion_threshold})
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="status-alert status-low">
                    😊 Normal
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
    # 10. COMPARACIÓN DE MODELOS
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

    comparacion["SVR"] = (
        comparacion["SVR"].round(2)
    )

    comparacion["GBR"] = (
        comparacion["GBR"].round(2)
    )

    st.dataframe(
        comparacion,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # 11. RESULTADO TÉCNICO
    # =====================================================

    with st.expander(
        "🔧 Ver configuración de los modelos"
    ):

        st.write(
            "### SVR"
        )

        st.write(
            "Columnas utilizadas durante el entrenamiento:"
        )

        st.code(
            str(svr_columns)
        )

        st.write(
            f"Cantidad: {len(svr_columns)}"
        )


        st.write(
            "### GBR"
        )

        st.write(
            "Columnas utilizadas durante el entrenamiento:"
        )

        st.code(
            str(gbr_columns)
        )

        st.write(
            f"Cantidad: {len(gbr_columns)}"
        )


        st.write(
            "### Scaler SVR"
        )

        st.code(
            str(svr_scaler_columns)
        )


        st.write(
            "### Scaler GBR"
        )

        st.code(
            str(gbr_scaler_columns)
        )


# =========================================================
# 12. PIE DE PÁGINA
# =========================================================

st.markdown("---")

st.caption(
    "Los resultados son predicciones generadas por "
    "modelos de aprendizaje automático y no constituyen "
    "un diagnóstico clínico."
)
