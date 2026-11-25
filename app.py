import streamlit as st
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title="Clustering - App Didáctica",
    page_icon="🔍",
    layout="wide"
)

# --- 1. ENCABEZADO Y LOGO ---
col1, col2 = st.columns([1, 4])

with col1:
    # Lógica para cargar logo (ajusta el nombre del archivo si es necesario)
    logo_path = "assets/logo.png"  # <--- COLOCA TU LOGO AQUÍ
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        # Placeholder si no encuentran el logo
        st.write("📷 *[Inserta tu logo aquí]*")

with col2:
    st.title("Aprendizaje No Supervisado: Clustering")
    st.markdown("### 🎓 Una herramienta interactiva para entender el algoritmo K-Means")

st.divider()

# --- 2. DESCRIPCIÓN GENERAL ---
st.markdown("""
### 👋 ¡Bienvenido/a!

Esta aplicación ha sido diseñada para estudiantes que desean comprender **cómo las máquinas encuentran patrones** en los datos sin necesidad de etiquetas previas.
A diferencia de los métodos tradicionales de enseñanza, aquí podrás **tocar los datos**, visualizar los conceptos y experimentar con algoritmos reales.

**Objetivo:** Pasar de la intuición visual a la implementación técnica profesional.
""")

# --- 3. RUTA DE APRENDIZAJE ---
st.info("📌 **Guía de Navegación:** Usa el menú lateral para avanzar por los módulos en orden.")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("1. Conceptos")
    st.markdown("""
    * **Teoría:** Diferencia entre aprendizaje Supervisado y No Supervisado.
    * **Intuición:** ¿Cómo agrupamos objetos mentalmente?
    """)

with c2:
    st.subheader("2. Exploración")
    st.markdown("""
    * **Datos Reales:** Marketing, Biología y Demografía.
    * **EDA:** Visualización y detección de patrones naturales antes de modelar.
    """)

with c3:
    st.subheader("3. Algoritmo")
    st.markdown("""
    * **K-Means Mecánico:** Simulación paso a paso del algoritmo.
    * **Laboratorio:** Implementación en Python con Scikit-Learn y PCA.
    """)

st.divider()

# --- 4. CRÉDITOS Y AUTORÍA ---
# Usamos un contenedor o columnas para darle un toque formal al pie de página

footer_col1, footer_col2 = st.columns([1, 1])

with footer_col1:
    st.markdown("#### 🏛️ Institución")
    st.write("**Universidad del Caribe / Departamento de Ciencias Báscias e Ingenierías**")
    st.caption("Maestría en Analítica e Inteligencia de Negocios")

with footer_col2:
    st.markdown("#### 👨‍💻 Desarrollo")
    st.write("**Autor:** Dr. Héctor Fernando Gómez-García")
    st.caption("Versión 1.0 | Desarrollado con Streamlit & Python")

# Opcional: Logo en la barra lateral para que persista en toda la navegación
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width="stretch")
    st.sidebar.markdown("---")