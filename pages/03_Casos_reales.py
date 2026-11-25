import streamlit as st
import pandas as pd
import plotly.express as px

# Importamos tus funciones desde utils.datasets
# Asegúrate de tener un archivo vacio __init__.py en la carpeta utils/
from utils.datasets import (
    available_contexts,
    context_info,
    load_context_df,
    numeric_columns,
    default_numeric_columns
)

st.set_page_config(page_title="Exploración de Datos", layout="wide")

st.title("Exploración de Casos Reales (EDA)")

st.markdown("""
Antes de aplicar cualquier algoritmo, debemos "sentir" los datos.
En el aprendizaje no supervisado, buscamos **estructuras naturales** (grupos, densidades)
y debemos estar atentos a problemas de **escala** (magnitudes muy diferentes).
""")

# -----------------------------------------------------------------------------
# 1. BARRA LATERAL: SELECCIÓN Y CONFIGURACIÓN
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Selección del Caso")
    
    # Usamos tu función para llenar el selectbox
    # key="contexto": el caso elegido aquí sigue elegido en el laboratorio.
    context = st.selectbox("Elige un dataset:", available_contexts(), key="contexto")
    
    # Obtenemos la info del contexto seleccionado
    info = context_info(context)
    
    st.info(f"**Dataset actual:** {context}")
    st.markdown("---")
    st.write("Modifica las visualizaciones en el panel principal.")

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS
# -----------------------------------------------------------------------------
try:
    df = load_context_df(context)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# Identificamos columnas
id_cols = info.get("id_cols", [])
cols_num = numeric_columns(df, exclude=id_cols)
cols_default = default_numeric_columns(context, df)

# -----------------------------------------------------------------------------
# 3. DESCRIPCIÓN DEL PROBLEMA (CONTEXTO)
# -----------------------------------------------------------------------------
st.info(f"📖 **Contexto:** {info['description']}")

# -----------------------------------------------------------------------------
# 4. TABS DE EXPLORACIÓN
# -----------------------------------------------------------------------------
tab_datos, tab_stats, tab_viz = st.tabs(["🗃️ Datos Crudos", "📊 Estadísticas", "🎨 Visualización"])

# --- TAB 1: DATOS ---
with tab_datos:
    st.subheader("Muestra de datos")
    st.markdown(f"El dataset tiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")
    st.dataframe(df.head(50), width="stretch")

# --- TAB 2: ESTADÍSTICAS ---
with tab_stats:
    st.subheader("Resumen Numérico")
    st.markdown("""
    Revisa la **media (mean)** y la **desviación estándar (std)**.
    ¿Ves variables con valores muy grandes (miles) y otras con valores pequeños (decimales)?
    *Eso será importante para el escalamiento más adelante.*
    """)
    
    # Transponemos (.T) para facilitar la lectura
    desc = df[cols_num].describe().T
    
    # Estilizamos la tabla para resaltar magnitudes (Opcional, requiere pandas reciente)
    st.dataframe(desc.style.background_gradient(cmap="Blues"), width="stretch")

# --- TAB 3: VISUALIZACIÓN ---
with tab_viz:
    st.subheader("Buscando patrones visuales")
    
    if len(cols_num) >= 2:
        c1, c2, c3 = st.columns(3)
        
        # Lógica para pre-seleccionar columnas interesantes usando tus defaults
        idx_x = cols_num.index(cols_default[0]) if cols_default and cols_default[0] in cols_num else 0
        idx_y = cols_num.index(cols_default[1]) if cols_default and len(cols_default) > 1 and cols_default[1] in cols_num else 1
        
        with c1:
            x_axis = st.selectbox("Eje X", cols_num, index=idx_x)
        with c2:
            y_axis = st.selectbox("Eje Y", cols_num, index=idx_y)
        with c3:
            # Opción para añadir una tercera dimensión (color o tamaño) si quisieras
            # Por ahora lo dejamos simple para enfatizar la falta de etiquetas
            st.write("") 
            st.markdown("*(Recuerda: No hay etiquetas de color reales)*")

        # Gráfico interactivo con Plotly
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            hover_data=id_cols or None, # Al pasar el mouse, vemos el ID (ej. Municipio o Estudiante)
            title=f"Dispersión: {x_axis} vs {y_axis}",
            opacity=0.7,
            template="plotly_white"
        )
        st.plotly_chart(fig, width="stretch")
        
        # Reflexión guiada
        st.divider()
        st.markdown("### 🧠 Preguntas para el análisis")
        st.write(f"""
        1. Al cruzar **{x_axis}** con **{y_axis}**, ¿se forman grupos separados o es una mancha uniforme?
        2. Si ves puntos muy alejados del resto, ¿podrían ser **anomalías** (outliers)?
        3. ¿Crees que estas dos variables son suficientes para segmentar, o necesitaríamos usar todas las columnas a la vez?
        """)
        
    else:
        st.warning("Este dataset no tiene suficientes columnas numéricas para graficar.")

if st.button("Siguiente página: Solución intuitiva"):
    st.switch_page("pages/04_Agrupamiento_intuitivo.py") # Asumiendo que usarás multipage
    # Si no usas multipage, aquí podrías cambiar una variable de estado session_state