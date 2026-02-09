import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils.datasets import (
    available_contexts, context_info, load_context_df,
    numeric_columns, default_numeric_columns
)

st.set_page_config(page_title="Lab: K-Means en Scikit-Learn", layout="wide")

st.title("Laboratorio: Aplicando lo aprendido")

st.markdown("""
Ha llegado el momento de la verdad. Vamos a usar la librería profesional **Scikit-Learn** para encontrar patrones.

**El flujo de trabajo profesional:**
1.  **Preprocesamiento:** Estandarizar escalas (¡vital para el mapa de calor!).
2.  **Modelado:** Ejecutar K-Means.
3.  **Perfilado:** Interpretar los grupos mediante sus "Centroides" (Z-Scores).
""")

# --- 1. CONFIGURACIÓN ---
st.sidebar.header("🛠️ Configuración")

context = st.sidebar.selectbox("Caso de Estudio", available_contexts())
info = context_info(context)
df = load_context_df(context)

# Hiperparámetros
k = st.sidebar.slider("Número de Clusters (k)", 2, 8, 3)
seed = st.sidebar.number_input("Semilla Aleatoria", 42)
scale = st.sidebar.checkbox("Estandarizar Datos", value=True, help="Recomendado. Convierte todo a la misma escala (Z-Scores).")

# Selección de Variables
st.sidebar.markdown("---")
num_cols = numeric_columns(df, exclude=info.get("id_cols", []))
defaults = default_numeric_columns(context, df)
sel_cols = st.sidebar.multiselect("Variables:", num_cols, default=defaults)

if len(sel_cols) < 2:
    st.error("Selecciona al menos 2 variables.")
    st.stop()

# --- 2. PROCESAMIENTO ---
X_raw = df[sel_cols].dropna()

# Escalamiento
scaler = StandardScaler()
if scale:
    X_processed = scaler.fit_transform(X_raw)
else:
    X_processed = X_raw.values

# Modelado
kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
labels = kmeans.fit_predict(X_processed)

# Métricas
inertia = kmeans.inertia_
sil_score = silhouette_score(X_processed, labels) if k > 1 else 0

# DataFrame FINAL (Datos Originales + Cluster) -> Para Descargar
df_final = X_raw.copy()
df_final['Cluster'] = labels.astype(str)

# DataFrame ESCALADO (Datos Z + Cluster) -> Para Heatmap
df_scaled_viz = pd.DataFrame(X_processed, columns=sel_cols, index=X_raw.index)
df_scaled_viz['Cluster'] = labels.astype(str)

# --- 3. RESULTADOS ---
st.divider()
st.header(f"📊 Resultados: {context}")

c1, c2 = st.columns([2, 1])
c1.metric("Inercia", f"{inertia:,.0f}")
c1.metric("Silueta", f"{sil_score:.3f}")

# Botón descarga (Datos ORIGINALES, para que Marketing los entienda)
csv = df_final.to_csv(index=False).encode('utf-8')
c2.download_button(
    "⬇️ Descargar CSV (Valores Reales)",
    data=csv,
    file_name=f"clusters_{context}.csv",
    mime="text/csv"
)

# --- 4. PERFILADO (CORREGIDO) ---
st.subheader("🕵️ Perfilamiento (Mapa de Calor)")

# Calculamos el promedio sobre los datos ESCALADOS
profile_z = df_scaled_viz.groupby('Cluster')[sel_cols].mean()

# Título dinámico
heatmap_title = "Promedios Estandarizados (Z-Scores)" if scale else "Promedios en Unidades Originales"

fig_heatmap = px.imshow(
    profile_z, 
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r", 
    # Fijamos el punto medio en 0 (el promedio global)
    zmin=-2 if scale else None,
    zmax=2 if scale else None,
    origin='lower',
    title=heatmap_title
)

st.plotly_chart(fig_heatmap, use_container_width=True)

if scale:
    st.info("""
    **Interpretación de Z-Scores:**
    * **0**: El grupo se comporta igual al promedio general.
    * **> 0 (Rojo)**: El grupo está **por encima** del promedio.
    * **< 0 (Azul)**: El grupo está **por debajo** del promedio.
    
    *Ejemplo: Un valor de +1.5 significa que es un grupo "muy alto" en esa variable.*
    """)

# --- 5. VISUALIZACIÓN GEOMÉTRICA (PCA) ---
st.divider()
st.subheader("🗺️ Mapa Visual de los Clústers")

if len(sel_cols) == 2:
    # Caso simple: 2 variables
    st.markdown("Como seleccionaste solo 2 variables, podemos graficarlas directamente.")
    fig_viz = px.scatter(
        df_final, x=sel_cols[0], y=sel_cols[1], 
        color="Cluster", 
        title="Dispersión (2 Variables Originales)",
        template="plotly_white"
    )
    st.plotly_chart(fig_viz, use_container_width=True)

else:
    # Caso complejo: > 2 variables (PCA)
    st.markdown(f"""
    Has seleccionado **{len(sel_cols)} variables**. No podemos graficar en {len(sel_cols)} dimensiones.
    Usamos **PCA (Análisis de Componentes Principales)** para comprimir la información en 2 ejes.
    """)
    
    # --- Explicación Pedagógica ---
    with st.expander("¿Qué son PC1 y PC2? (Clic para entender la magia)"):
        st.write("""
        Imagina que tienes datos de una persona: *Edad, Salario, Deuda, Ahorros, etc.* (muchas dimensiones).
        
        **PC1 y PC2 no son variables originales, son "Resúmenes":**
        1.  **PC1 (Componente Principal 1):** Es la combinación de tus variables que captura la **mayor variación** de los datos. (Ej: Podría ser una mezcla de *Salario + Ahorros*).
        2.  **PC2:** Captura la segunda mayor variación restante, asegurando que sea información nueva (perpendicular a PC1).
        
        **La Intuición:**
        Es como tomarle una foto a una nube de puntos 3D. PCA busca el "mejor ángulo" para que los puntos se vean lo más separados posible en la foto 2D, en lugar de verse amontonados.
        """)
        
    # Cálculo de PCA
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_processed)
    var_explained = pca.explained_variance_ratio_.sum()
    
    df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels.astype(str)
    
    # Gráfico
    fig_pca = px.scatter(
        df_pca, x='PC1', y='PC2', color='Cluster',
        title=f"Mapa PCA (Retiene el {var_explained:.1%} de la información original)",
        opacity=0.8,
        template="plotly_white",
        labels={"PC1": "PC1 (Mayor variabilidad)", "PC2": "PC2 (Segunda mayor var.)"}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    if var_explained < 0.6:
        st.warning("⚠️ Cuidado: Al reducir a 2D se está perdiendo mucha información (menos del 60%). Los grupos podrían parecer mezclados visualmente aunque en realidad estén separados en sus dimensiones originales.")