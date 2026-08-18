import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predicción de Salud Mental",
    page_icon="🧠",
    layout="wide"
)

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


# ---------------------------------------------------------
# 2. CARGA DE MODELOS
# ---------------------------------------------------------

def load_model(filepath):
    """
    Carga un modelo serializado con pickle.
    """

    with open(filepath, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_trained_models():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(base_dir, "modelos")

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
    # Cargar modelo SVR
    # -----------------------------------------------------

    if not os.path.exists(path_svr):

        st.error(
            f"❌ No se encontró el archivo SVR:\n{path_svr}"
        )

    else:

        try:

            svr_model = load_model(path_svr)

            st.success(
                "✅ Modelo SVR cargado correctamente"
            )

        except Exception as e:

            st.error(
                f"❌ Error cargando SVR: "
                f"{type(e).__name__}: {e}"
            )

    # -----------------------------------------------------
    # Cargar modelo GBR
    # -----------------------------------------------------

    if not os.path.exists(path_gbr):

        st.error(
            f"❌ No se encontró el archivo GBR:\n{path_gbr}"
        )

    else:

        try:

            gbr_model = load_model(path_gbr)

            st.success(
                "✅ Modelo GBR cargado correctamente"
            )

        except Exception as e:

            st.error(
                f"❌ Error cargando GBR: "
                f"{type(e).__name__}: {e}"
            )

    return svr_model, gbr_model


svr_model, gbr_model = load_trained_models()


# ---------------------------------------------------------
# 3. FUNCIONES DE PREDICCIÓN
# ---------------------------------------------------------

def predict_svr(modelos, data):

    """
    Realiza las predicciones utilizando los tres modelos SVR.

    Estructura esperada del PKL:

        scaler
        ansiedad
        estres
        depresion
    """

    scaler = modelos["scaler"]

    data_scaled = scaler.transform(data)

    pred_ansiedad = modelos["ansiedad"].predict(data_scaled)

    pred_estres = modelos["estres"].predict(data_scaled)

    pred_depresion = modelos["depresion"].predict(data_scaled)

    return (
        float(pred_ansiedad[0]),
        float(pred_estres[0]),
        float(pred_depresion[0])
    )


def predict_gbr(modelos, data):

    """
    Realiza las predicciones utilizando los tres modelos GBR.

    Estructura esperada del PKL:

        ansiedad
        estres
        depresion
        scaler
    """

    scaler = modelos["scaler"]

    data_scaled = scaler.transform(data)

    pred_ansiedad = modelos["ansiedad"].predict(data_scaled)

    pred_estres = modelos["estres"].predict(data_scaled)

    pred_depresion = modelos["depresion"].predict(data_scaled)

    return (
        float(pred_ansiedad[0]),
        float(pred_estres[0]),
        float(pred_depresion[0])
    )


# ---------------------------------------------------------
# 4. INTERFAZ PRINCIPAL
# ---------------------------------------------------------

st.title("🧠 Evaluación de Salud Mental")

st.write(
    "Ingrese las variables descriptivas. "
    "El sistema utilizará los modelos preentrenados "
    "para calcular las predicciones."
)


# ---------------------------------------------------------
# 5. FORMULARIO
# ---------------------------------------------------------

with st.form(key="mental_health_form"):

    st.subheader("📋 Ingreso de Predictores")

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
        label="🚀 Ejecutar Diagnóstico Predictivo",
        use_container_width=True,
        type="primary"
    )


# ---------------------------------------------------------
# 6. PREDICCIÓN
# ---------------------------------------------------------

if submit_button:

    if svr_model is None or gbr_model is None:

        st.error(
            "❌ No se pueden realizar las predicciones "
            "porque uno o ambos modelos no pudieron cargarse."
        )

    else:

        # -------------------------------------------------
        # Encoding de sexo
        # -------------------------------------------------

        if sexo == "Masculino":

            sexo_masculino = 1
            sexo_otro = 0

        elif sexo == "Femenino":

            sexo_masculino = 0
            sexo_otro = 0

        else:

            sexo_masculino = 0
            sexo_otro = 1


        # -------------------------------------------------
        # Construcción del DataFrame
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Realizar predicciones
        # -------------------------------------------------

        with st.spinner(
            "Procesando predicción..."
        ):

            try:

                # =========================================
                # SVR
                # =========================================

                (
                    nivel_ansiedad_svr,
                    nivel_estres_svr,
                    nivel_depresion_svr
                ) = predict_svr(
                    svr_model,
                    input_data
                )


                # =========================================
                # GBR
                # =========================================

                (
                    nivel_ansiedad_gbr,
                    nivel_estres_gbr,
                    nivel_depresion_gbr
                ) = predict_gbr(
                    gbr_model,
                    input_data
                )


                # =========================================
                # MODELOS SELECCIONADOS PARA MOSTRAR
                # =========================================

                nivel_ansiedad = nivel_ansiedad_svr

                nivel_estres = nivel_estres_svr

                nivel_depresion = nivel_depresion_gbr


                # =========================================
                # Umbrales
                # =========================================

                ansiedad_threshold = 18

                estres_threshold = 8

                depresion_threshold = 5


                # =========================================
                # Estados
                # =========================================

                ansiedad_alta = (
                    nivel_ansiedad >= ansiedad_threshold
                )

                estres_alto = (
                    nivel_estres >= estres_threshold
                )

                depresion_alta = (
                    nivel_depresion >= depresion_threshold
                )


                # =========================================
                # Emojis
                # =========================================

                ansiedad_emoji = (
                    "😞" if ansiedad_alta else "😊"
                )

                estres_emoji = (
                    "😞" if estres_alto else "😊"
                )

                depresion_emoji = (
                    "😞" if depresion_alta else "😊"
                )


            except Exception as e:

                st.error(
                    "❌ Error durante la predicción"
                )

                st.exception(e)

                st.stop()


        # -------------------------------------------------
        # 7. PRESENTACIÓN DE RESULTADOS
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Resultados de la Evaluación"
        )


        res_col1, res_col2, res_col3 = st.columns(3)


        # =================================================
        # ANSIEDAD
        # =================================================

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
                f'''
                <div class="status-alert {status_class}">
                    {ansiedad_emoji}
                    {status_text}
                    (Umbral: {ansiedad_threshold})
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # =================================================
        # ESTRÉS
        # =================================================

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
                f'''
                <div class="status-alert {status_class}">
                    {estres_emoji}
                    {status_text}
                    (Umbral: {estres_threshold})
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # =================================================
        # DEPRESIÓN
        # =================================================

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
                f'''
                <div class="status-alert {status_class}">
                    {depresion_emoji}
                    {status_text}
                    (Umbral: {depresion_threshold})
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )
