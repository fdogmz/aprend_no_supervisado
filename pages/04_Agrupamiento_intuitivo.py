import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="K-Means Paso a Paso", layout="wide")

st.title("K-Means: La intuición mecánica")

st.markdown("""
El algoritmo K-Means es como un baile en dos pasos que se repite hasta que nadie se mueve.
Vamos a verlo en acción con nuestros consumidores (Precio vs. Calidad).
""")

# -----------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES (Lógica del Algoritmo)
# -----------------------------------------------------------------------------
def make_dataset(n_samples=300, seed=42):
    rng = np.random.default_rng(seed)
    # Generamos 3 grupos claros
    c1 = rng.normal(loc=[2, 6], scale=0.6, size=(n_samples//3, 2)) # Calidad alta, Precio bajo
    c2 = rng.normal(loc=[6, 2], scale=0.6, size=(n_samples//3, 2)) # Precio alto, Calidad baja
    c3 = rng.normal(loc=[5, 5], scale=0.8, size=(n_samples//3, 2)) # Balanceado
    X = np.vstack([c1, c2, c3])
    # Recortamos a escala 1-7 (Likert)
    X = np.clip(X, 1, 7)
    return pd.DataFrame(X, columns=["Precio", "Calidad"])

def get_inertia(X, centroids, labels):
    # Suma de distancias al cuadrado
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            dist = np.linalg.norm(cluster_points - centroids[k], axis=1)**2
            inertia += np.sum(dist)
    return inertia

# -----------------------------------------------------------------------------
# 2. ESTADO DE LA SESIÓN (Persistencia)
# -----------------------------------------------------------------------------
if "df_kmeans" not in st.session_state:
    st.session_state.df_kmeans = make_dataset()

if "centroids" not in st.session_state:
    st.session_state.centroids = None # Array de centroides
    st.session_state.labels = None    # Array de etiquetas
    st.session_state.step = 0         # Contador de pasos
    st.session_state.history = []     # Para guardar la inercia

# -----------------------------------------------------------------------------
# 3. LAYOUT PRINCIPAL
# -----------------------------------------------------------------------------
col_controls, col_viz = st.columns([1, 2], gap="medium")

with col_controls:
    st.header("⚙️ Controles")
    
    # Configuración Inicial
    k = st.slider("Número de Clusters (K)", 2, 5, 3)
    
    st.divider()
    
    # BOTÓN 1: INICIALIZAR
    if st.button("1️⃣ Inicializar (Semillas)", use_container_width=True):
        X = st.session_state.df_kmeans.values
        # Elegimos k puntos al azar como centroides iniciales
        indices = np.random.choice(X.shape[0], k, replace=False)
        st.session_state.centroids = X[indices]
        st.session_state.labels = np.zeros(X.shape[0]) # Todos etiqueta 0 temporalmente
        st.session_state.step = 1
        st.session_state.history = []
    
    # Lógica para habilitar botones siguientes
    has_started = st.session_state.centroids is not None
    
    # BOTÓN 2: ASIGNAR
    if st.button("2️⃣ Paso A: Asignar Puntos", disabled=not has_started, use_container_width=True):
        X = st.session_state.df_kmeans.values
        cents = st.session_state.centroids
        # Calculamos distancias a cada centroide
        distances = np.linalg.norm(X[:, np.newaxis] - cents, axis=2)
        # Asignamos al más cercano
        new_labels = np.argmin(distances, axis=1)
        st.session_state.labels = new_labels
        st.session_state.step += 1
        
    # BOTÓN 3: ACTUALIZAR
    if st.button("3️⃣ Paso B: Mover Centroides", disabled=not has_started, use_container_width=True):
        X = st.session_state.df_kmeans.values
        labels = st.session_state.labels
        cents = st.session_state.centroids
        new_cents = []
        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                new_cents.append(points.mean(axis=0))
            else:
                new_cents.append(cents[i]) # Si un cluster queda vacío, se queda quieto
        
        st.session_state.centroids = np.array(new_cents)
        
        # Calcular inercia para gráfico
        inertia = get_inertia(X, np.array(new_cents), labels)
        st.session_state.history.append(inertia)
        st.session_state.step += 1

    st.markdown("---")
    
    # Explicación dinámica del estado
    if st.session_state.step == 0:
        st.info("👋 Dale a **Inicializar** para comenzar. Elegiremos 3 puntos al azar como 'líderes'.")
    elif st.session_state.step == 1:
        st.warning("⚠️ **Inicialización Aleatoria.** Mira los puntos grandes (X). Están en lugares aleatorios. Ahora debemos decidir qué puntos pertenecen a quién.")
    elif st.session_state.step % 2 == 0: # Después de asignar
        st.success("✅ **Asignación Completa.** Cada punto se ha pintado del color de su líder más cercano. \n\n¿Notas que los líderes (X) NO están en el centro de su grupo? Toca moverlos.")
    else: # Después de actualizar
        st.info("🔄 **Centroides Actualizados.** Los líderes (X) se movieron al centro de gravedad de sus puntos. \n\nAhora algunos puntos podrían estar más cerca de un nuevo líder. ¡Repite el paso de Asignar!")

with col_viz:
    st.subheader("Visualización en Tiempo Real")
    
    df = st.session_state.df_kmeans
    cents = st.session_state.centroids
    labels = st.session_state.labels
    
    # Gráfico Base con Plotly Graph Objects (Más control que Express)
    fig = go.Figure()
    
    # Colores para clusters
    colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A']
    
    # 1. Dibujar Puntos de Datos
    if labels is None:
        # Todos grises al principio
        fig.add_trace(go.Scatter(
            x=df['Precio'], y=df['Calidad'],
            mode='markers', marker=dict(color='lightgray', size=8, opacity=0.5),
            name='Datos'
        ))
    else:
        # Coloreados por cluster
        for i in range(k):
            cluster_data = df[labels == i]
            fig.add_trace(go.Scatter(
                x=cluster_data['Precio'], y=cluster_data['Calidad'],
                mode='markers',
                marker=dict(color=colors[i % len(colors)], size=8, opacity=0.6),
                name=f'Grupo {i+1}'
            ))

    # 2. Dibujar Centroides (Si existen)
    if cents is not None:
        fig.add_trace(go.Scatter(
            x=cents[:, 0], y=cents[:, 1],
            mode='markers',
            marker=dict(
                symbol='x', 
                size=15, 
                color='black', 
                line=dict(width=2, color='white')
            ),
            name='Centroides (Líderes)'
        ))

    # Configuración del Layout del Gráfico
    # ... (código anterior de la gráfica) ...

    # 3. Configuración del Layout del Gráfico
    fig.update_layout(
        xaxis=dict(title="Importancia del Precio (1-7)", range=[0.5, 7.5]),
        yaxis=dict(title="Importancia de la Calidad (1-7)", range=[0.5, 7.5]),
        width=700, height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        
        # --- AJUSTE DE LEYENDA ---
        legend=dict(
            yanchor="bottom",  # Anclamos la parte inferior de la caja de leyenda
            y=0.01,            # La ponemos casi pegada al eje X (0 es el borde)
            xanchor="left",    # Anclamos la parte izquierda
            x=0.01,            # Pegada al eje Y
            bgcolor="rgba(255, 255, 255, 0.8)", # Fondo semitransparente por si pisa algún punto
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico de Convergencia (Inercia)
    if len(st.session_state.history) > 1:
        st.write("📉 **Evolución del error (Compactación de los grupos):**")
        st.line_chart(st.session_state.history)

if st.button("Siguiente página: Laboratorio K-Means"):
    st.switch_page("pages/05_KMeans_sklearn.py") # Asumiendo que usarás multipage
    # Si no usas multipage, aquí podrías cambiar una variable de estado session_state