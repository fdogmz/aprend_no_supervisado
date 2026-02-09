# simulated_datasets.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------
# Utilidades
# ----------------------------
def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# 1) CLIENTES (RFM simplificado)
# ----------------------------
def simulate_customers(
    n: int = 120,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simula clientes con Recencia (días), Frecuencia (compras en 90 días),
    y Valor monetario (MXN en 90 días). Incluye ticket_promedio.

    Idea latente: 3 segmentos (no se expone por defecto).
    """
    rng = _make_rng(seed)

    # Segmentos latentes (para que haya estructura)
    # 0: inactivos (alta recencia, baja frecuencia, bajo gasto)
    # 1: regulares (media)
    # 2: leales/alto valor (baja recencia, alta frecuencia, alto gasto)
    seg = rng.choice([0, 1, 2], size=n, p=[0.35, 0.45, 0.20])

    # Recencia: gamma (positiva), segmentada
    rec = np.where(
        seg == 0, rng.gamma(shape=2.4, scale=25.0, size=n),
        np.where(seg == 1, rng.gamma(shape=2.2, scale=14.0, size=n),
                 rng.gamma(shape=2.0, scale=6.0, size=n))
    )
    rec = _clip(rec, 0, 180)  # 0-180 días
    rec = np.round(rec).astype(int)

    # Frecuencia (90 días): Poisson segmentado, penalizando algo por recencia
    base_freq = np.where(seg == 0, rng.poisson(1.8, n),
                         np.where(seg == 1, rng.poisson(4.5, n), rng.poisson(9.0, n)))
    # penalización suave: recencia alta -> menos frecuencia
    freq = base_freq - (rec / 90.0)
    freq = np.round(_clip(freq, 0, 40)).astype(int)

    # Valor monetario: depende de frecuencia y de un "ticket" lognormal
    # ticket típico por compra:
    ticket = np.where(
        seg == 0, rng.lognormal(mean=np.log(180), sigma=0.45, size=n),
        np.where(seg == 1, rng.lognormal(mean=np.log(320), sigma=0.45, size=n),
                 rng.lognormal(mean=np.log(650), sigma=0.50, size=n))
    )
    # valor total ~ frecuencia * ticket + ruido
    monetary = freq * ticket + rng.normal(0, 80, n)
    monetary = _clip(monetary, 0, None)
    monetary = np.round(monetary).astype(int)

    # ticket promedio observado (si freq=0, usar 0)
    ticket_avg = np.where(freq > 0, monetary / np.maximum(freq, 1), 0)
    ticket_avg = np.round(ticket_avg).astype(int)

    df = pd.DataFrame({
        "cliente_id": [f"C{str(i+1).zfill(4)}" for i in range(n)],
        "recencia_dias": rec,
        "frecuencia_90d": freq,
        "valor_monetario_90d": monetary,
        "ticket_promedio": ticket_avg,
        # "segmento_latente": seg,  # descomenta si quieres evaluar “verdad”
    })
    return df


# ----------------------------
# 2) ESTUDIANTES (8 asignaturas)
# ----------------------------
def simulate_students(
    n: int = 90,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simula calificaciones en 8 asignaturas:
    - Matemáticas (4): Calculo, Algebra, Estadistica, Discretas
    - Lenguaje (4): Redaccion, Comprension, Ingles, Comunicacion

    Se genera una habilidad latente cuantitativa y otra verbal,
    correlacionadas moderadamente.
    """
    rng = _make_rng(seed)

    # habilidades latentes (media 0, var 1)
    # correlación moderada
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.35],
                    [0.35, 1.0]])
    latent = rng.multivariate_normal(mean, cov, size=n)
    q = latent[:, 0]  # cuantitativa
    v = latent[:, 1]  # verbal

    # Dificultades/offsets por asignatura (en escala de 0-100)
    # Añadimos ruido idiosincrático
    def grade(base, wq, wv, noise_sd):
        g = base + 10*wq*q + 10*wv*v + rng.normal(0, noise_sd, n)
        return _clip(g, 40, 100)

    # Matemáticas
    calc = grade(base=72, wq=0.85, wv=0.10, noise_sd=6.5)
    alg  = grade(base=74, wq=0.80, wv=0.10, noise_sd=6.0)
    est  = grade(base=76, wq=0.55, wv=0.35, noise_sd=6.0)  # estadística mezcla
    disc = grade(base=73, wq=0.70, wv=0.20, noise_sd=6.5)

    # Lenguaje
    red  = grade(base=78, wq=0.10, wv=0.85, noise_sd=5.5)
    comp = grade(base=77, wq=0.10, wv=0.80, noise_sd=5.5)
    ing  = grade(base=76, wq=0.05, wv=0.75, noise_sd=6.0)
    com  = grade(base=79, wq=0.10, wv=0.70, noise_sd=5.5)

    df = pd.DataFrame({
        "matricula": [f"UC{rng.integers(1, 3)}01{str(i+1).zfill(4)}" for i in range(n)],
        "Calculo": np.round(calc, 1),
        "Algebra": np.round(alg, 1),
        "Estadistica": np.round(est, 1),
        "Discretas": np.round(disc, 1),
        "Redaccion": np.round(red, 1),
        "ComprensionLectora": np.round(comp, 1),
        "Ingles": np.round(ing, 1),
        "Comunicacion": np.round(com, 1),
    })

    # Promedios útiles para interpretación (opcionales)
    df["Prom_Matematicas"] = np.round(df[["Calculo","Algebra","Estadistica","Discretas"]].mean(axis=1), 1)
    df["Prom_Lenguaje"]    = np.round(df[["Redaccion","ComprensionLectora","Ingles","Comunicacion"]].mean(axis=1), 1)
    df["Prom_General"]     = np.round(df[["Prom_Matematicas","Prom_Lenguaje"]].mean(axis=1), 1)
    return df


# ----------------------------
# 3) MUNICIPIOS (demográfico/socioeconómico)
# ----------------------------
def simulate_municipalities(
    n: int = 35,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simula municipios con indicadores demográficos y socioeconómicos:
    - tasa de analfabetismo (%)
    - grado promedio de estudios (años)
    - ingreso_medio_mensual (MXN) y razon_ingresos (p90/p10)
    - indice_gini (0-1)
    - pobreza_pct (%)
    - desempleo_pct (%)
    - densidad_pob_km2
    """
    rng = _make_rng(seed)

    # factor latente de desarrollo (más alto = mejor)
    dev = rng.normal(0, 1, n)

    # densidad: lognormal (algunos municipios muy densos)
    dens = rng.lognormal(mean=np.log(180), sigma=0.9, size=n)
    dens = _clip(dens, 5, 12000)

    # escolaridad (años): aumenta con dev
    escolaridad = 8.5 + 1.6*dev + rng.normal(0, 0.6, n)
    escolaridad = _clip(escolaridad, 6.0, 13.5)

    # analfabetismo (%): baja con dev y con escolaridad
    analf = 8.0 - 2.0*dev - 0.6*(escolaridad - 9.0) + rng.normal(0, 1.0, n)
    analf = _clip(analf, 0.8, 18.0)

    # ingreso medio (MXN): sube con dev y algo con densidad (urbano)
    ingreso = 11000 + 2200*dev + 400*np.log1p(dens) + rng.normal(0, 1200, n)
    ingreso = _clip(ingreso, 6500, 32000)

    # razón de ingresos p90/p10: mayor desigualdad suele elevar razón
    razon = 6.0 - 0.8*dev + rng.normal(0, 0.7, n)
    razon = _clip(razon, 3.0, 12.0)

    # gini: relacionado con razón + ruido
    gini = 0.33 + 0.02*(razon - 5.0) + rng.normal(0, 0.015, n)
    gini = _clip(gini, 0.30, 0.55)

    # pobreza (%): baja con dev, sube con desigualdad
    pobreza = 45 - 10*dev + 18*(gini - 0.35) + rng.normal(0, 4.0, n)
    pobreza = _clip(pobreza, 8, 75)

    # desempleo (%): moderado, ligeramente más alto en zonas urbanas, ruido
    desempleo = 2.5 + 0.25*np.log1p(dens) + rng.normal(0, 0.8, n)
    desempleo = _clip(desempleo, 1.0, 10.0)

    df = pd.DataFrame({
        "municipio": [f"Municipio_{str(i+1).zfill(2)}" for i in range(n)],
        "densidad_pob_km2": np.round(dens, 1),
        "grado_promedio_estudios": np.round(escolaridad, 2),
        "tasa_analfabetismo_pct": np.round(analf, 2),
        "ingreso_medio_mensual_mxn": np.round(ingreso, 0).astype(int),
        "razon_ingresos_p90_p10": np.round(razon, 2),
        "indice_gini": np.round(gini, 3),
        "pobreza_pct": np.round(pobreza, 1),
        "desempleo_pct": np.round(desempleo, 1),
        # "desarrollo_latente": dev,  # opcional para validación
    })
    return df


# ----------------------------
# Guardado a CSV (para tu app Streamlit)
# ----------------------------
def main(out_dir: str = "data", seed: int = 123):
    import os
    os.makedirs(out_dir, exist_ok=True)

    clientes = simulate_customers(n=120, seed=seed)
    estudiantes = simulate_students(n=90, seed=seed + 1)
    municipios = simulate_municipalities(n=35, seed=seed + 2)

    clientes.to_csv(f"{out_dir}/clientes.csv", index=False)
    estudiantes.to_csv(f"{out_dir}/estudiantes.csv", index=False)
    municipios.to_csv(f"{out_dir}/municipios.csv", index=False)

    print("CSV generados en:", out_dir)
    print(" - clientes.csv")
    print(" - estudiantes.csv")
    print(" - municipios.csv")


if __name__ == "__main__":
    main(out_dir="data", seed=123)
