"""
M2_preprocessor.py — Preprocesamiento y estandarización global
==============================================================
Replica EXACTAMENTE la Fase 5 del script validado de la tesis:

    X_todos = df_bal[SCORES].values.astype(float)  ← solo tablas_ok
    media_g = X_todos.mean(axis=0)
    std_g   = X_todos.std(axis=0, ddof=1)
    std_g[std_g==0] = 1
    Xk_std  = (Xk - media_g) / std_g   ← z-score global por tabla

NOTA: la estandarización es OPCIONAL (checkbox en la UI).
Cuando standardize=False, los datos entran en escala original.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessResult:
    panel_raw:    dict = field(default_factory=dict)
    panel_scaled: dict = field(default_factory=dict)
    tablas:       list = field(default_factory=list)
    filas_por_tabla: dict = field(default_factory=dict)
    variable_names:  list = field(default_factory=list)
    n_tablas:    int  = 0
    n_variables: int  = 0
    standardized: bool = False
    global_means: Optional[np.ndarray] = None
    global_stds:  Optional[np.ndarray] = None
    zero_treatment: str = "keep"


def preprocess(panel: dict, standardize: bool, zero_treatment: str) -> PreprocessResult:
    """
    Fase 5 del script validado: estandarización z-score global.

    media_g y std_g se calculan sobre la CONCATENACIÓN de TODAS las tablas
    del panel (solo las que pasaron el filtro MIN_FILAS en m1).
    Esto es idéntico a calcularlos sobre df_bal del script.
    """
    result = PreprocessResult()
    result.zero_treatment = zero_treatment
    result.standardized   = standardize

    if not panel:
        return result

    tablas = sorted(panel.keys())
    result.tablas          = tablas
    result.n_tablas        = len(tablas)
    result.variable_names  = list(panel[tablas[0]].columns)
    result.n_variables     = len(result.variable_names)
    result.panel_raw       = {t: panel[t].copy() for t in tablas}
    result.filas_por_tabla = {t: list(panel[t].index) for t in tablas}

    if standardize:
        # Concatenar TODAS las tablas → X_todos (replica df_bal[SCORES].values)
        X_todos = np.vstack([panel[t].values.astype(float) for t in tablas])
        media_g = X_todos.mean(axis=0)
        std_g   = X_todos.std(axis=0, ddof=1)
        std_g   = np.where(std_g < 1e-12, 1.0, std_g)   # std_g[std_g==0] = 1

        result.global_means = media_g
        result.global_stds  = std_g

        for t in tablas:
            Xk = panel[t].values.astype(float)
            result.panel_scaled[t] = (Xk - media_g) / std_g
    else:
        result.global_means = None
        result.global_stds  = None
        for t in tablas:
            result.panel_scaled[t] = panel[t].values.astype(float)

    return result


def get_preprocessing_summary(result: PreprocessResult) -> str:
    lines = [
        f"Tablas (K)     : {result.n_tablas}",
        f"Variables (p)  : {result.n_variables}",
    ]
    if result.standardized:
        lines.append("Estandarización : Sí — Z-score global (μ y σ sobre todas las tablas)")
    else:
        lines.append("Estandarización : No — escala original")
    if result.zero_treatment == "impute_mean":
        lines.append("Ceros           : Imputados con media global antes del groupby")
    else:
        lines.append("Ceros           : Conservados como scores efectivos (solo NaN imputados)")
    for t in result.tablas:
        n = result.panel_scaled[t].shape[0]
        lines.append(f"  Tabla '{t}': {n} filas")
    return "\n".join(lines)
