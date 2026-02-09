from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CONTEXTS = {
    "Clientes (RFM)": {
        "file": "clientes.csv",
        "id_cols": ["cliente_id"],
        "default_cols": ["recencia_dias", "frecuencia_90d", "valor_monetario_90d", "ticket_promedio"],
        "description": (
            "Clientes con variables de comportamiento: **recencia**, **frecuencia** y **valor monetario** "
            "(más ticket promedio)."
        ),
    },
    "Estudiantes (calificaciones)": {
        "file": "estudiantes.csv",
        "id_cols": ["matricula"],
        "default_cols": [
            "Calculo", "Algebra", "Estadistica", "Discretas",
            "Redaccion", "ComprensionLectora", "Ingles", "Comunicacion",
            "Prom_Matematicas", "Prom_Lenguaje", "Prom_General"
        ],
        "description": (
            "Estudiantes con calificaciones simuladas en 8 asignaturas (4 matemáticas + 4 lenguaje) "
            "y promedios por área."
        ),
    },
    "Municipios (socioeconómico)": {
        "file": "municipios.csv",
        "id_cols": ["municipio"],
        "default_cols": [
            "tasa_analfabetismo_pct",
            "grado_promedio_estudios",
            "ingreso_medio_mensual_mxn",
            "razon_ingresos_p90_p10",
            "indice_gini",
            "pobreza_pct",
            "desempleo_pct",
            "densidad_pob_km2",
        ],
        "description": (
            "Municipios con indicadores demográficos y socioeconómicos simulados: analfabetismo, escolaridad, "
            "ingreso, desigualdad (Gini), pobreza, desempleo y densidad."
        ),
    },
}

def data_exists() -> bool:
    return DATA_DIR.exists()

def available_contexts() -> list[str]:
    return list(CONTEXTS.keys())

def context_info(context: str) -> dict:
    return CONTEXTS[context]

@st.cache_data(show_spinner=False)
def load_context_df(context: str) -> pd.DataFrame:
    info = context_info(context)
    path = DATA_DIR / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Asegúrate de tener los CSV en /data.")
    df = pd.read_csv(path)
    return df

def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def default_numeric_columns(context: str, df: pd.DataFrame) -> list[str]:
    info = context_info(context)
    defaults = [c for c in info["default_cols"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if defaults:
        return defaults
    # fallback: primeras numéricas
    nums = numeric_columns(df, exclude=info["id_cols"])
    return nums[: min(6, len(nums))]
