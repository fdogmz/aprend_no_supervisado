import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.kmeans_steps import (
    assign_clusters,
    inertia_sse,
    init_representatives,
    update_centroids,
)

st.set_page_config(page_title="K-Means Paso a Paso", layout="wide")

st.title("K-Means: la intuición mecánica")

st.markdown("""
El algoritmo K-Means es un baile de dos pasos que se repite hasta que nadie se mueve.
Vamos a verlo con nuestros consumidores (importancia del **Precio** vs. importancia de la
**Calidad**, en escala 1–7).
""")

COLORES = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A"]


# -----------------------------------------------------------------------------
# 1. DATOS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples // 3
    c1 = rng.normal(loc=[2.0, 6.0], scale=0.6, size=(n, 2))  # calidad alta, precio bajo
    c2 = rng.normal(loc=[6.0, 2.0], scale=0.6, size=(n, 2))  # precio alto, calidad baja
    c3 = rng.normal(loc=[3.5, 3.5], scale=0.7, size=(n, 2))  # intermedio
    X = np.clip(np.vstack([c1, c2, c3]), 1, 7)
    return pd.DataFrame(X, columns=["Precio", "Calidad"])


# -----------------------------------------------------------------------------
# 2. ESTADO
# -----------------------------------------------------------------------------
def reiniciar_estado(k: int, motivo: str | None = None) -> None:
    """Vuelve al punto de partida. Se llama al arrancar y cada vez que cambia k."""
    st.session_state.centroids = None
    st.session_state.labels = None
    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.k_estado = k
    st.session_state.convergio = False
    st.session_state.aviso_reinicio = motivo


if "df_kmeans" not in st.session_state:
    st.session_state.df_kmeans = make_dataset()

if "k_estado" not in st.session_state:
    reiniciar_estado(k=3)


def registrar_inercia(X, centroids, labels, etiqueta: str) -> None:
    st.session_state.history.append(
        {"paso": len(st.session_state.history), "momento": etiqueta,
         "Inercia (SSE)": inertia_sse(X, centroids, labels)}
    )


def una_iteracion(X, centroids, k, rng):
    """Asignar + mover, una vez. Devuelve (labels, centroids nuevos)."""
    labels = assign_clusters(X, centroids)
    centroids = update_centroids(X, labels, k, rng)
    return labels, centroids


# -----------------------------------------------------------------------------
# 3. CONTROLES
# -----------------------------------------------------------------------------
col_controls, col_viz = st.columns([1, 2], gap="large")

with col_controls:
    st.subheader("Controles")

    k = st.slider("Número de grupos (k)", 2, 5, 3)

    # El estado guarda cuántos centroides existen. Si el usuario mueve el
    # deslizador a mitad del algoritmo, el estado deja de corresponder a k y hay
    # que empezar de nuevo (antes esto reventaba con un IndexError).
    if k != st.session_state.k_estado:
        reiniciar_estado(k, motivo=f"Cambiaste k a {k}: el algoritmo vuelve a empezar.")

    semilla = st.number_input("Semilla de la inicialización", min_value=0, max_value=9999,
                              value=7, step=1,
                              help="Cambiarla mueve los puntos de arranque. El resultado final "
                                   "puede cambiar: K-Means no es determinista.")

    st.divider()

    X = st.session_state.df_kmeans.values
    iniciado = st.session_state.centroids is not None
    asignado = st.session_state.labels is not None

    if st.button("1. Inicializar (semillas)", width="stretch"):
        rng = np.random.default_rng(int(semilla))
        reiniciar_estado(k)
        st.session_state.centroids = init_representatives(X, k, rng)
        st.session_state.step = 1
        st.rerun()

    if st.button("2. Paso A: asignar puntos", disabled=not iniciado,
                 width="stretch") and iniciado:
        labels = assign_clusters(X, st.session_state.centroids)
        st.session_state.labels = labels
        st.session_state.step += 1
        registrar_inercia(X, st.session_state.centroids, labels, "tras asignar")
        st.rerun()

    if st.button("3. Paso B: mover centroides", disabled=not asignado,
                 width="stretch") and asignado:
        rng = np.random.default_rng(int(semilla))
        previos = st.session_state.centroids
        nuevos = update_centroids(X, st.session_state.labels, k, rng)
        st.session_state.convergio = bool(np.allclose(previos, nuevos))
        st.session_state.centroids = nuevos
        st.session_state.step += 1
        registrar_inercia(X, nuevos, st.session_state.labels, "tras mover")
        st.rerun()

    if st.button("Repetir hasta que nadie se mueva", disabled=not iniciado,
                 width="stretch") and iniciado:
        rng = np.random.default_rng(int(semilla))
        centroids = st.session_state.centroids
        for _ in range(50):
            labels, nuevos = una_iteracion(X, centroids, k, rng)
            st.session_state.step += 2
            registrar_inercia(X, centroids, labels, "tras asignar")
            registrar_inercia(X, nuevos, labels, "tras mover")
            if np.allclose(centroids, nuevos):
                centroids = nuevos
                break
            centroids = nuevos
        st.session_state.centroids = centroids
        st.session_state.labels = labels
        st.session_state.convergio = True
        st.rerun()

    st.divider()

    # --- Mensaje de estado ---
    if st.session_state.aviso_reinicio:
        st.warning(st.session_state.aviso_reinicio)
    elif st.session_state.convergio:
        st.success("**Convergió.** Los centroides ya no se mueven, así que las asignaciones "
                   "tampoco van a cambiar. Aquí termina el algoritmo.")
    elif st.session_state.step == 0:
        st.info("Pulsa **Inicializar**. Se eligen k puntos al azar como representantes "
                "provisionales. Nada más: todavía no hay grupos.")
    elif not asignado:
        st.warning("**Inicialización aleatoria.** Los puntos siguen grises porque **aún no "
                   "pertenecen a nadie**. Las X están donde cayó el azar, no en ningún centro.")
    elif st.session_state.step % 2 == 0:
        st.success("**Asignación hecha.** Cada punto tomó el color de la X más cercana. "
                   "Fíjate en que las X **no** están en el centro de su propio grupo.")
    else:
        st.info("**Centroides movidos.** Cada X saltó al promedio de sus puntos. Al moverse, "
                "puede que ahora haya puntos más cerca de otra X: vuelve al Paso A.")

# -----------------------------------------------------------------------------
# 4. VISUALIZACIÓN
# -----------------------------------------------------------------------------
with col_viz:
    st.subheader("Visualización en tiempo real")

    df = st.session_state.df_kmeans
    cents = st.session_state.centroids
    labels = st.session_state.labels

    fig = go.Figure()

    if labels is None:
        # Sin asignación todavía: TODOS grises. Pintarlos de un color sugeriría
        # que ya forman un grupo, que es justo lo contrario de lo que pasa.
        fig.add_trace(go.Scatter(
            x=df["Precio"], y=df["Calidad"], mode="markers",
            marker=dict(color="lightgray", size=8, opacity=0.6),
            name="Sin asignar",
        ))
    else:
        for i in range(k):
            datos_i = df[labels == i]
            fig.add_trace(go.Scatter(
                x=datos_i["Precio"], y=datos_i["Calidad"], mode="markers",
                marker=dict(color=COLORES[i % len(COLORES)], size=8, opacity=0.6),
                name=f"Grupo {i + 1} (n={len(datos_i)})",
            ))

    if cents is not None:
        fig.add_trace(go.Scatter(
            x=cents[:, 0], y=cents[:, 1], mode="markers",
            marker=dict(symbol="x", size=16, color="black",
                        line=dict(width=2, color="white")),
            name="Centroides",
        ))

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Importancia del Precio (1-7)", range=[0.5, 7.5]),
        yaxis=dict(title="Importancia de la Calidad (1-7)", range=[0.5, 7.5]),
        height=500, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="black", borderwidth=1),
    )
    st.plotly_chart(fig, width="stretch")

    if len(st.session_state.history) > 1:
        st.markdown("**Evolución del error (suma de distancias al cuadrado):**")
        hist = pd.DataFrame(st.session_state.history).set_index("paso")
        st.line_chart(hist["Inercia (SSE)"])
        st.caption("Baja en los dos pasos y nunca sube: asignar al más cercano no puede "
                   "empeorar el error, y mover la X al promedio tampoco.")

# -----------------------------------------------------------------------------
# 5. LA PARTE INCÓMODA: EL AZAR IMPORTA
# -----------------------------------------------------------------------------
st.divider()
st.subheader("¿Siempre sale lo mismo?")

st.markdown("""
K-Means arranca de puntos elegidos al azar, así que **no es determinista**: dos corridas del
mismo algoritmo, sobre los mismos datos y con el mismo k, pueden terminar en particiones
distintas. Corramos el algoritmo completo desde seis semillas diferentes.
""")

if st.button("Probar seis semillas"):
    X = st.session_state.df_kmeans.values
    filas = []
    for s in range(6):
        rng = np.random.default_rng(s)
        centroids = init_representatives(X, k, rng)
        for _ in range(50):
            labels, nuevos = una_iteracion(X, centroids, k, rng)
            if np.allclose(centroids, nuevos):
                centroids = nuevos
                break
            centroids = nuevos
        tam = sorted(np.bincount(labels, minlength=k).tolist(), reverse=True)
        filas.append({
            "Semilla": s,
            "Inercia final": round(inertia_sse(X, centroids, labels), 1),
            "Tamaño de los grupos": " / ".join(str(t) for t in tam),
        })
    res = pd.DataFrame(filas)
    st.dataframe(res, hide_index=True, width="stretch")

    if res["Inercia final"].nunique() == 1:
        st.info("Con estos datos y este k, las seis semillas llegaron al mismo sitio. "
                "**No siempre pasa**: prueba con otro k.")
    else:
        st.warning(f"Las semillas no llegaron al mismo sitio: la inercia final va de "
                   f"**{res['Inercia final'].min():,.1f}** a **{res['Inercia final'].max():,.1f}**. "
                   "Por eso `scikit-learn` repite la inicialización varias veces "
                   "(`n_init`) y se queda con la mejor.")

st.divider()
if st.button("Siguiente página: Laboratorio K-Means"):
    st.switch_page("pages/05_KMeans_sklearn.py")
