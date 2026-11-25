import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.datasets import (
    available_contexts,
    context_info,
    default_numeric_columns,
    load_context_df,
    numeric_columns,
)

st.set_page_config(page_title="Lab: K-Means en Scikit-Learn", layout="wide")

st.title("Laboratorio: aplicando lo aprendido")

st.markdown("""
Ahora con la librería profesional, **scikit-learn**. El flujo es siempre el mismo:

1. **Preparar:** poner todas las variables en la misma escala.
2. **Elegir k:** con evidencia, no por corazonada.
3. **Modelar:** ejecutar K-Means.
4. **Perfilar:** leer los centroides y ponerles un nombre de negocio.
""")

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.header("Configuración")

# key="contexto" es compartida con la página de exploración: el caso que el
# alumno eligió allá sigue elegido aquí.
context = st.sidebar.selectbox("Caso de estudio", available_contexts(), key="contexto")
info = context_info(context)
df = load_context_df(context)

k = st.sidebar.slider("Número de grupos (k)", 2, 8, 3)
seed = int(st.sidebar.number_input("Semilla aleatoria", value=42, step=1))
scale = st.sidebar.checkbox(
    "Estandarizar datos", value=True,
    help="Recomendado. Deja todas las variables en z-scores, con media 0 y desviación 1.",
)

st.sidebar.markdown("---")
id_cols = info.get("id_cols", [])
num_cols = numeric_columns(df, exclude=id_cols)
defaults = default_numeric_columns(context, df)
sel_cols = st.sidebar.multiselect("Variables:", num_cols, default=defaults)

if len(sel_cols) < 2:
    st.error("Selecciona al menos 2 variables.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. PROCESAMIENTO
# -----------------------------------------------------------------------------
X_raw = df[sel_cols].dropna()
X_processed = StandardScaler().fit_transform(X_raw) if scale else X_raw.values

kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
labels = kmeans.fit_predict(X_processed)

inertia = kmeans.inertia_
sil_score = silhouette_score(X_processed, labels)

# Los identificadores se arrastran: un CSV de segmentos sin cliente_id no sirve
# para llamar a nadie.
df_final = df.loc[X_raw.index, id_cols + sel_cols].copy()
df_final["Grupo"] = labels

df_scaled_viz = pd.DataFrame(X_processed, columns=sel_cols, index=X_raw.index)
df_scaled_viz["Grupo"] = labels

# -----------------------------------------------------------------------------
# 3. ¿CUÁNTOS GRUPOS?
# -----------------------------------------------------------------------------
st.divider()
st.header("1. ¿Cuántos grupos?")

st.markdown("""
`k` no sale de los datos: lo pones tú. Hay dos indicios que ayudan, y **no siempre coinciden**.

* **Inercia (codo):** siempre baja al subir k. Se busca el punto donde deja de bajar rápido.
* **Silueta:** mide qué tan separados quedan los grupos, de −1 a 1. Aquí sí hay un máximo.
""")


@st.cache_data(show_spinner=False)
def barrido_k(X: np.ndarray, k_max: int, seed: int) -> pd.DataFrame:
    filas = []
    for kk in range(2, k_max + 1):
        modelo = KMeans(n_clusters=kk, random_state=seed, n_init="auto").fit(X)
        filas.append({
            "k": kk,
            "Inercia": modelo.inertia_,
            "Silueta": silhouette_score(X, modelo.labels_),
        })
    return pd.DataFrame(filas)


k_max = min(8, len(X_raw) - 1)
barrido = barrido_k(X_processed, k_max, seed)

c_codo, c_sil = st.columns(2)

with c_codo:
    fig_codo = px.line(barrido, x="k", y="Inercia", markers=True,
                       title="Inercia: busca el codo", template="plotly_white")
    fig_codo.add_vline(x=k, line_dash="dash", line_color="#C8102E")
    st.plotly_chart(fig_codo, width="stretch")

with c_sil:
    fig_sil = px.line(barrido, x="k", y="Silueta", markers=True,
                      title="Silueta: busca el máximo", template="plotly_white")
    fig_sil.add_vline(x=k, line_dash="dash", line_color="#C8102E")
    st.plotly_chart(fig_sil, width="stretch")

k_mejor_sil = int(barrido.loc[barrido["Silueta"].idxmax(), "k"])
if k_mejor_sil == k:
    st.success(f"La silueta es máxima justo en **k = {k}**, que es el que tienes elegido.")
else:
    st.warning(
        f"La silueta es máxima en **k = {k_mejor_sil}**, no en el **k = {k}** que elegiste. "
        "Eso no quiere decir que estés mal: si los grupos de tu k se pueden explicar y accionar "
        "y los del otro no, gana el tuyo. **La métrica aconseja; la decisión es de negocio.**"
    )

# -----------------------------------------------------------------------------
# 4. POR QUÉ SE ESTANDARIZA
# -----------------------------------------------------------------------------
st.divider()
st.header("2. Por qué hay que estandarizar")

st.markdown("""
K-Means mide **distancias**. Si una variable viene en miles (un ingreso) y otra en decimales
(un índice de 0 a 1), la primera decide sola quién se parece a quién: la segunda no alcanza a
mover la distancia. Estandarizar le da a cada variable el mismo derecho a voto.
""")

escalas = X_raw.std()
razon = escalas.max() / escalas.min() if escalas.min() > 0 else np.inf
st.markdown(
    f"En **{context}**, la variable más dispersa (`{escalas.idxmax()}`) tiene una desviación "
    f"**{razon:,.0f} veces** mayor que la menos dispersa (`{escalas.idxmin()}`)."
)

with st.expander("Ver el mismo k con y sin estandarizar"):
    etiquetas_con = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(
        StandardScaler().fit_transform(X_raw))
    etiquetas_sin = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(
        X_raw.values)
    coincidencia = adjusted_rand_score(etiquetas_con, etiquetas_sin)

    comp = pd.DataFrame({
        "Estandarizado": pd.Series(etiquetas_con).value_counts().sort_index(),
        "Sin estandarizar": pd.Series(etiquetas_sin).value_counts().sort_index(),
    })
    comp.index.name = "Grupo"
    st.dataframe(comp, width="stretch")
    st.metric("Coincidencia entre ambas particiones (índice de Rand ajustado)",
              f"{coincidencia:.2f}")
    st.caption("1.00 es la misma partición; 0.00 es tanto parecido como el azar. "
               "Son los mismos datos, el mismo algoritmo y el mismo k: lo único que cambia "
               "es la escala, y sale otra segmentación.")

# -----------------------------------------------------------------------------
# 5. RESULTADOS
# -----------------------------------------------------------------------------
st.divider()
st.header(f"3. Resultados: {context}")

c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Inercia", f"{inertia:,.0f}")
c2.metric("Silueta", f"{sil_score:.3f}")
c3.download_button(
    "Descargar CSV (valores reales)",
    data=df_final.to_csv(index=False).encode("utf-8"),
    file_name=f"grupos_{context}.csv",
    mime="text/csv",
    width="stretch",
)
if id_cols:
    c3.caption(f"Incluye `{'`, `'.join(id_cols)}` para poder actuar sobre cada caso.")

# --- Perfilado ---
st.subheader("Perfilamiento: ¿quién es cada grupo?")

perfil = df_scaled_viz.groupby("Grupo")[sel_cols].mean()
perfil.index = [f"Grupo {i} (n={(labels == i).sum()})" for i in perfil.index]

tope = float(np.abs(perfil.values).max())
fig_heatmap = px.imshow(
    perfil,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-tope, zmax=tope,  # simétrico y sin recortar: el grupo extremo se ve extremo
    labels=dict(x="Variable", y="", color="z" if scale else "valor"),
    title="Promedios estandarizados (z-scores)" if scale else "Promedios en unidades originales",
)
fig_heatmap.update_layout(template="plotly_white", height=90 + 60 * len(perfil))
st.plotly_chart(fig_heatmap, width="stretch")

if scale:
    st.info("""
**Cómo se lee:** **0** es el promedio general; **rojo (> 0)** está por encima;
**azul (< 0)**, por debajo. Un +1.5 es «muy alto en esa variable».

Ponerle nombre a cada renglón —*«los que compran poco pero caro»*— es el verdadero
entregable. Un número de grupo no le sirve a nadie.
""")

# --- Mapa ---
st.subheader("Mapa visual de los grupos")

if len(sel_cols) == 2:
    st.markdown("Con dos variables podemos graficarlas directamente.")
    fig_viz = px.scatter(df_final, x=sel_cols[0], y=sel_cols[1],
                         color=df_final["Grupo"].astype(str),
                         hover_data=id_cols or None,
                         labels={"color": "Grupo"},
                         title="Dispersión (2 variables originales)",
                         template="plotly_white")
    st.plotly_chart(fig_viz, width="stretch")
else:
    st.markdown(f"""
Elegiste **{len(sel_cols)} variables** y no podemos dibujar en {len(sel_cols)} dimensiones.
**PCA** las comprime en dos ejes para poder verlas.
""")

    with st.expander("¿Qué son PC1 y PC2?"):
        st.write("""
PC1 y PC2 no son variables originales: son **resúmenes**.

1. **PC1** es la combinación de tus variables que captura la mayor variación de los datos.
2. **PC2** captura la segunda mayor, y es información nueva (perpendicular a PC1).

La intuición: es como fotografiar una nube de puntos en 3D buscando el ángulo desde el que
los puntos se ven lo más separados posible. **PCA se usa aquí solo para mirar**: el
agrupamiento se hizo con todas las variables, no con estos dos ejes.
""")

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_processed)
    var_explained = pca.explained_variance_ratio_.sum()

    df_pca = pd.DataFrame(comps, columns=["PC1", "PC2"], index=X_raw.index)
    df_pca["Grupo"] = labels.astype(str)
    for col in id_cols:
        df_pca[col] = df.loc[X_raw.index, col]

    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="Grupo",
        hover_data=id_cols or None,
        title=f"Mapa PCA (retiene el {var_explained:.1%} de la información original)",
        opacity=0.8, template="plotly_white",
        labels={"PC1": "PC1 (mayor variabilidad)", "PC2": "PC2 (segunda mayor)"},
    )
    st.plotly_chart(fig_pca, width="stretch")

    if var_explained < 0.6:
        st.warning("Cuidado: al reducir a 2D se pierde más del 40 % de la información. "
                   "Los grupos pueden verse mezclados en la foto aunque estén separados "
                   "en sus dimensiones originales.")

st.divider()
st.markdown("""
### La pregunta que falta

Esta app ya te dio los grupos. Lo que **no** te puede dar es la respuesta a:
*¿haríamos algo distinto con cada grupo?* Si la respuesta es no, la segmentación es correcta
y es inútil a la vez. Eso se resuelve en la libreta, con datos reales.
""")
