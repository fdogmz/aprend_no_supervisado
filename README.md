# Aprendizaje no supervisado: una introducción intuitiva

App didáctica en Streamlit para abrir el tema de **agrupamiento** antes de que el alumno
escriba código. Universidad del Caribe · Maestría en Analítica e Inteligencia de Negocios.

Publicada en <https://aprendnosupervisado-main-unicaribe.streamlit.app/>

## Para qué sirve y para qué no

La app es **la introducción**, no la clase. Da la intuición en unos 20 minutos y después el
trabajo se va a una libreta con datos reales, donde el alumno sí escribe el `StandardScaler`,
el `KMeans` y el PCA con sus manos. Si la app hace el modelado por él, no lo aprende.

| Página | Uso en clase | Minutos |
|---|---|---|
| 1. Aprendizaje supervisado | Repaso: dónde está la `Y` | 5 |
| 2. Aprendizaje no supervisado | **El gancho.** Se borra la `Y` y el toggle revela los grupos | 5 |
| 3. Casos reales | Exploración rápida; se puede saltar | 5 |
| 4. Agrupamiento intuitivo | **Demo proyectada.** El baile de dos pasos, paso a paso | 10 |
| 5. Laboratorio K-Means | *Después* de la libreta, para jugar con `k` | — |

Los tres conjuntos de datos son **simulados** (`simulated_datasets.py` los genera). Eso está
bien para construir la intuición, pero no sirve como evidencia de que el método funciona: para
eso está la libreta con datos reales.

## Ejecutar en local

```bash
python -m venv .venv
source .venv/bin/activate   # windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar los CSV de `data/`:

```bash
python simulated_datasets.py
```

## Estructura

```
app.py                      Portada y ruta de aprendizaje
pages/                      Las cinco páginas, en orden
utils/datasets.py           Catálogo de casos (archivo, ids, variables por omisión)
utils/kmeans_steps.py       K-Means paso a paso (lo usa la página 4)
utils/viz.py                Helpers de Plotly; hoy no lo importa ninguna página
simulated_datasets.py       Generador de los CSV de data/ (ejecución manual)
.streamlit/config.toml      Tema claro fijo, para proyectar
data/                       clientes.csv (120), estudiantes.csv (90), municipios.csv (35)
```

## Decisiones de diseño que conviene no deshacer

- **El tema está fijado en claro** (`.streamlit/config.toml`). Las gráficas usan
  `template="plotly_white"`; con el tema oscuro quedaban recuadros blancos sobre fondo negro.
- **Tras «Inicializar», los puntos se quedan grises.** Pintarlos de un color sugeriría que ya
  forman un grupo, que es justo lo contrario de lo que ocurre en ese paso.
- **Cambiar `k` a mitad del algoritmo reinicia el estado**, con aviso. El estado guarda `k`
  centroides; si `k` cambia sin reiniciar, la página revienta.
- **La inercia se registra después de cada medio paso**, con los centroides y las etiquetas que
  de verdad corresponden a ese momento. Así la curva nunca sube, que es el argumento de que el
  algoritmo converge.
- **El CSV descargable arrastra el identificador** (`cliente_id`, `municipio`). Una tabla de
  segmentos sin identificador no permite actuar sobre nadie.
- **El laboratorio barre `k` de 2 a 8** y dibuja inercia y silueta. Elegir `k` es la decisión
  del alumno; sin las dos curvas no tiene con qué justificarla.
- **La página 5 muestra la misma partición con y sin estandarizar**, con el índice de Rand
  ajustado entre ambas. Es la forma más barata de probar que la escala no es un detalle.
- **El caso elegido viaja entre las páginas 3 y 5** (`key="contexto"` compartida).
