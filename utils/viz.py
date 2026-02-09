from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    category_orders: dict | None = None,
    color_discrete_map: dict | None = None,
    **kwargs,
):
    """
    Scatter 2D con Plotly Express.
    Soporta category_orders y color_discrete_map para estabilizar leyendas/colores.
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        opacity=0.85,
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        **kwargs,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def scatter_with_centroids(
    df: pd.DataFrame,
    x: str,
    y: str,
    labels_col: str,
    C: np.ndarray,
    title: str = "",
    category_orders: dict | None = None,
    color_discrete_map: dict | None = None,
    **kwargs,
):
    """
    Scatter 2D con etiquetas (labels_col) y centroides superpuestos.

    category_orders y color_discrete_map permiten:
    - leyenda con orden fijo
    - colores fijos por categoría
    lo cual evita cambios de layout al pasar entre pasos.
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=labels_col,
        title=title,
        opacity=0.85,
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        **kwargs,
    )

    # centroides
    cdf = pd.DataFrame({x: C[:, 0], y: C[:, 1], "centroide": [f"C{j}" for j in range(len(C))]})
    fig2 = px.scatter(cdf, x=x, y=y, text="centroide")

    for tr in fig2.data:
        tr.update(marker=dict(size=14, symbol="x"), showlegend=False, mode="markers+text")

    fig.add_traces(fig2.data)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig
