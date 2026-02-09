import streamlit as st
import pandas as pd

st.set_page_config(page_title="Repaso Supervisado", layout="wide")

st.title("Recordando el Aprendizaje Supervisado")

st.markdown("""
Para entender la *novedad* del aprendizaje no supervisado, primero debemos tener muy claro
cómo funciona el paradigma estándar con el que solemos empezar: el **Aprendizaje Supervisado**.
""")

st.divider()

# ---------------------------------------------------------
st.header("1. La Estructura de los Datos")

st.markdown("""
En este paradigma, la característica fundamental es que **contamos con la respuesta correcta**
durante el entrenamiento. Nuestros datos tienen una estructura clara de "Pregunta" y "Respuesta".
""")

# Creamos columnas para poner el DataFrame a la izquierda y la explicación a la derecha
col_data, col_text = st.columns([1, 1.5])

with col_data:
    st.markdown("**Ejemplo: Datos de Inmuebles**")
    # Creamos un dataframe pequeño ilustrativo
    df_ejemplo = pd.DataFrame({
        'Area_m2 (X1)': [80, 120, 60, 200],
        'Habitaciones (X2)': [2, 3, 1, 4],
        'Antiguedad (X3)': [5, 10, 2, 1],
        'PRECIO (Y)': [150, 220, 110, 400]
    })
    st.dataframe(df_ejemplo, hide_index=True)

with col_text:
    st.info("🔎 **Observa la tabla:**")
    st.markdown(r"""
    * **Variables Predictoras ($X$):** Son las columnas que describen el objeto.
        * *Ej:* Área, Habitaciones, Antigüedad.
        * Matemáticamente: $X_1, X_2, \dots, X_p$.
    
    * **Variable Objetivo o Etiqueta ($Y$):** Es la columna que queremos predecir.
        * *Ej:* **PRECIO**.
        * Esta columna es la "supervisión". El modelo ve estos ejemplos y ajusta sus parámetros para acertar a este valor.
    """)

# ---------------------------------------------------------
st.divider()

st.header("2. El Objetivo del Modelo")

st.markdown(r"""
El corazón del aprendizaje supervisado es encontrar una función matemática $f$ que, al recibir
las características de una casa nueva, nos devuelva un precio muy cercano al real.

$$
\hat{Y} = f(X_1, X_2, \dots, X_p)
$$

Buscamos minimizar el error entre nuestra predicción ($\hat{Y}$) y el valor real ($Y$).
""")
st.write("El flujo típico es el siguiente:")

# Diagrama de flujo usando Graphviz
st.graphviz_chart("""
    digraph G {
        # Configuración general: de izquierda a derecha
        rankdir=LR;
        node [fontname="Helvetica", shape=box, style=filled, fillcolor="#f0f2f6", color="white"];
        edge [color="#555555", arrowsize=0.8];

        # Definición de Nodos (Cajas)
        datos [label="Datos de Entrada\n(X)", shape=ellipse, fillcolor="#e3f2fd"];
        modelo [label="Modelo / Función\n f(x)", shape=component, fillcolor="#fff9c4"];
        pred [label="Predicción\n(Y estimada)", fillcolor="#e8f5e9"];
        real [label="Valor Real\n(Y etiqueta)", shape=ellipse, fillcolor="#ffebee"];
        error [label="Cálculo del Error\n(Y - Y_est)", shape=diamond, fillcolor="#ffccbc"];

        # Conexiones (Flechas)
        datos -> modelo [label="Entran"];
        modelo -> pred [label="Genera"];
        pred -> error [label="Se compara"];
        real -> error [label="Supervisa"];
        
        # Nota visual: una línea punteada que indica aprendizaje
        error -> modelo [label="Ajuste", style="dashed", color="red"];
    }
""")

# ---------------------------------------------------------
st.divider()

st.header("3. Los dos grandes problemas")

st.markdown("Dependiendo del **tipo de dato** que sea nuestra etiqueta ($Y$), dividimos los problemas en dos categorías:")

col_reg, col_clas = st.columns(2)

with col_reg:
    st.subheader("📉 Regresión")
    st.success("La etiqueta $Y$ es un número continuo.")
    st.markdown("""
    **La pregunta es "¿Cuánto?".**
    
    *Ejemplos:*
    * Predecir el precio de una casa (euros).
    * Estimar la temperatura de mañana (grados).
    * Calcular ventas mensuales (unidades).
    """)

with col_clas:
    st.subheader("🗂️ Clasificación")
    st.warning("La etiqueta $Y$ es una categoría.")
    st.markdown("""
    **La pregunta es "¿Qué tipo es?".**
    
    *Ejemplos:*
    * ¿Correo Spam o No Spam? (Binaria)
    * ¿Imagen de Perro, Gato o Ave? (Multiclase)
    * ¿Cliente cancelará o renovará?
    """)

# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Reflexión final")
st.write("""
Hasta aquí, todo depende de que **alguien haya recolectado esas etiquetas ($Y$)**. 
Necesitamos históricos de precios, correos ya marcados como spam, etc.
""")

st.subheader("Pero... ¿Qué pasa si borramos la columna $Y$?")

if st.button("Siguiente página: Eliminando las etiquetas"):
    st.switch_page("pages/02_Aprendizaje_no_supervisado.py") # Asumiendo que usarás multipage
    # Si no usas multipage, aquí podrías cambiar una variable de estado session_state