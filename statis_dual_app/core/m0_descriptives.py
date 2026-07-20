"""
m0_descriptives.py — Estadísticas descriptivas genéricas
Compatible con modo longitudinal (panel) y modo dimensiones (bloques).
"""
import numpy as np
import pandas as pd
from typing import Optional


def compute_descriptives(df_raw: pd.DataFrame,
                         cols: list,
                         col_grupo: Optional[str] = None) -> dict:
    """
    Calcula estadísticas descriptivas para las columnas indicadas.

    Retorna dict con:
        'global'  : DataFrame (n_vars × stats) — estadísticas globales
        'por_grupo': dict {grupo: DataFrame} si col_grupo no es None
        'corr'    : DataFrame correlación de Pearson (n_vars × n_vars)
        'freq'    : dict {col: Series conteos} para detectar Likert/ordinales
    """
    sub = df_raw[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')

    # ── Estadísticas globales ─────────────────────────────────────────────────
    desc = sub.describe(percentiles=[.25, .5, .75]).T
    desc.columns = desc.columns.str.replace('%', 'pct')
    desc['cv']      = desc['std'] / desc['mean'].abs().replace(0, np.nan)
    desc['asimetria'] = sub.skew()
    desc['curtosis']  = sub.kurtosis()
    desc['n_missing'] = sub.isna().sum()
    desc['n_zeros']   = (sub == 0).sum()

    out = {
        'global':   desc.round(4),
        'corr':     sub.corr(method='pearson').round(4),
        'freq':     {c: sub[c].value_counts().sort_index() for c in cols},
        'por_grupo': {}
    }

    # ── Por grupo (si se indica) ──────────────────────────────────────────────
    if col_grupo and col_grupo in df_raw.columns:
        for g, grp in df_raw.groupby(col_grupo):
            sg = grp[cols].copy()
            for c in cols:
                sg[c] = pd.to_numeric(sg[c], errors='coerce')
            out['por_grupo'][g] = sg.describe(percentiles=[.25,.5,.75]).T.round(4)

    return out


def descriptives_to_df(desc_dict: dict) -> pd.DataFrame:
    """Convierte el resultado global a DataFrame listo para mostrar en tabla."""
    return desc_dict['global'][[
        'count','mean','std','min','25pct','50pct','75pct','max','cv',
        'asimetria','curtosis','n_missing','n_zeros'
    ]].rename(columns={
        'count':'N', 'mean':'Media', 'std':'D.E.',
        'min':'Mín', '25pct':'Q1', '50pct':'Mediana',
        '75pct':'Q3', 'max':'Máx', 'cv':'CV',
        'asimetria':'Asimetría', 'curtosis':'Curtosis',
        'n_missing':'NaN', 'n_zeros':'Ceros'
    })
