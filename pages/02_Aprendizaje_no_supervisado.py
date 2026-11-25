import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Introducción al Clustering", layout="wide")

st.title("¿Qué buscamos cuando no hay etiquetas?")

# ---------------------------------------------------------
# 1. ESCENARIO DE NEGOCIO (CONCRETO)
# ---------------------------------------------------------
st.header("1. Un caso práctico: Segmentación de Clientes")

st.markdown("""
Imagina que trabajas en el departamento de marketing de una entidad financiera.
Tienes acceso a una base de datos con información financiera de **150 clientes**.

Las variables que conoces de cada cliente son:
1.  **Ingreso Mensual** (en miles de $).
2.  **Gasto Mensual** (en miles de $).

Tu jefe te pide: *"Necesitamos lanzar ofertas personalizadas. Agrupa a estos clientes por su perfil financiero."*
""")

# --- Generación de Datos Sintéticos ---
np.random.seed(42)

# Grupo 1: Ahorradores (Ingresos altos, Gastos bajos)
ingresos_1 = np.random.normal(8, 1.0, 50)
gastos_1   = np.random.normal(2, 0.5, 50)

# Grupo 2: Gastadores (Ingresos altos, Gastos altos)
ingresos_2 = np.random.normal(8, 1.0, 50)
gastos_2   = np.random.normal(8, 1.0, 50)

# Grupo 3: Limitados (Ingresos bajos, Gastos bajos)
ingresos_3 = np.random.normal(3, 0.8, 50)
gastos_3   = np.random.normal(3, 0.8, 50)

# Concatenamos todo en un DataFrame sin etiquetas (simulando la realidad)
datos = pd.DataFrame({
    'Ingreso_Mensual': np.concatenate([ingresos_1, ingresos_2, ingresos_3]),
    'Gasto_Mensual': np.concatenate([gastos_1, gastos_2, gastos_3])
})

# Creamos etiquetas internas solo para la solución (el usuario no las ve al principio)
etiquetas_reales = [0]*50 + [1]*50 + [2]*50

# ---------------------------------------------------------
# 2. LOS DATOS CRUDOS (TABLA)
# ---------------------------------------------------------
st.subheader("La materia prima: La Tabla de Datos")

col_tabla, col_info = st.columns([1, 2])

with col_tabla:
    # Mostramos solo el encabezado para no saturar
    st.dataframe(datos.head(10), width="stretch")

with col_info:
    st.warning("⚠️ **¡Atención!**")
    st.markdown("""
    Observa detenidamente la tabla de la izquierda.
    
    * Tenemos características (**Ingresos, Gastos**).
    * **NO tenemos una columna que diga "Tipo de Cliente"**.
    
    Nadie ha etiquetado a estos clientes como "Ahorradores" o "Derrochadores".
    Esa información **no existe** en la base de datos. Solo tenemos números crudos.
    """)

st.divider()

# ---------------------------------------------------------
# 3. VISUALIZACIÓN Y NECESIDAD (GRÁFICA)
# ---------------------------------------------------------
st.header("2. Visualizando el problema")

st.markdown("""
Si no tenemos etiquetas, ¿cómo entendemos estos datos?
Lo primero que hacemos es llevarlos a un plano visual. Grafiquemos **Ingresos vs. Gastos**.
""")

col_graf, col_analisis = st.columns([1.5, 1])

with col_graf:
    # Toggle para revelar
    revelar = st.toggle("🕵️ Revelar patrones ocultos (Solución ideal)", value=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if revelar:
        # Gráfica coloreada (lo que queremos lograr)
        scatter = ax.scatter(datos['Ingreso_Mensual'], datos['Gasto_Mensual'], 
                             c=etiquetas_reales, cmap='viridis', s=60, alpha=0.8, edgecolors='w')
        ax.set_title("¡Patrones Detectados! (3 Grupos Naturales)")
    else:
        # Gráfica en gris (la realidad inicial)
        ax.scatter(datos['Ingreso_Mensual'], datos['Gasto_Mensual'], 
                   c='gray', s=60, alpha=0.6, edgecolors='w')
        ax.set_title("Datos Crudos (Lo que ve el algoritmo)")

    ax.set_xlabel("Ingreso Mensual (miles)")
    ax.set_ylabel("Gasto Mensual (miles)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with col_analisis:
    st.markdown("### ¿Qué vemos aquí?")
    
    if not revelar:
        st.write("""
        Vemos una **nube de puntos**. A simple vista, parece un caos, pero si miras bien, 
        puedes notar que los puntos no están distribuidos uniformemente.
        
        Hay zonas donde los puntos se "amontonan". Esas acumulaciones sugieren que
        hay **tipos de clientes** que se comportan igual.
        """)
        st.info("👆 **Activa el interruptor** para ver lo que el Clustering intenta descubrir.")
    
    else:
        st.success("""
        **¡Exacto!**
        
        Al colorear los grupos, identificamos claramente 3 perfiles:
        1.  🟡 **Altos Ingresos / Altos Gastos** (Arriba-Derecha)
        2.  🟢 **Bajos Ingresos / Bajos Gastos** (Abajo-Izquierda)
        3.  🟣 **Altos Ingresos / Bajos Gastos** (Abajo-Derecha)
        
        El **Clustering** es el proceso matemático para pasar de los puntos grises a los puntos de colores automáticamente.
        """)

# ---------------------------------------------------------
# 4. FORMALIZACIÓN (EL PUENTE AL ÁLGEBRA)
# ---------------------------------------------------------
st.divider()
st.header("3. Definición Formal")

st.markdown(r"""
Ahora que tenemos la intuición visual, podemos definirlo matemáticamente.
Lo que acabamos de ver es una matriz de datos $\mathbf{X}$ con $n=150$ observaciones (clientes) y $p=2$ variables.

$$
\mathbf{X}=
\begin{bmatrix}
Ingreso_1 & Gasto_1 \\
Ingreso_2 & Gasto_2 \\
\vdots & \vdots \\
Ingreso_{150} & Gasto_{150}
\end{bmatrix}
$$

El algoritmo de agrupamiento recibe esta matriz $\mathbf{X}$ y nos devuelve un vector de etiquetas $\mathbf{g}$,
asignando cada cliente a un grupo $k$:

$$
g_i \in \{1, 2, 3\}
$$
""")

st.success("En la siguiente sección, veremos **cómo** el algoritmo K-Means logra encontrar estos centros y asignar los colores matemáticamente.")

if st.button("Siguiente página: Casos reales"):
    st.switch_page("pages/03_Casos_reales.py") # Asumiendo que usarás multipage
    # Si no usas multipage, aquí podrías cambiar una variable de estado session_state