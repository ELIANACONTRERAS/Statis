"""
M5_intrastructure.py — Intraestructura STATIS Dual
===================================================
La intraestructura ahora se calcula dentro de m4 (run_interstructure)
siguiendo exactamente el script validado de la tesis:
  Xs_panel = vstack([Xk para cada tabla])   SIN αk
  F_panel  = Xs_panel @ U * (1/√λ)
  r(i,j)   = Pearson(Xs_panel[:,i], F_panel[:,j])  manual

Este módulo solo envuelve el resultado en IntrastructureResult
para compatibilidad con la UI y los gráficos.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from core.m4_interstructure import StatisDualResult


@dataclass
class IntrastructureResult:
    correlations:    Optional[np.ndarray]  = None   # (p × n_dims)
    variable_names:  list = field(default_factory=list)
    dim_labels:      list = field(default_factory=list)
    n_dims:          int  = 2
    df_correlations: Optional[pd.DataFrame] = None


def compute_intrastructure(statis: StatisDualResult) -> IntrastructureResult:
    """
    Envuelve la intraestructura ya calculada en run_interstructure.
    No recalcula — usa statis.intra_df directamente.
    """
    result = IntrastructureResult()
    result.variable_names = statis.variable_names
    result.n_dims         = statis.n_dims_retained
    result.dim_labels     = [f"Dim{d+1}" for d in range(statis.n_dims_retained)]

    if statis.intra_df is not None:
        result.df_correlations = statis.intra_df
        result.correlations    = statis.intra_df.values
    else:
        # Fallback: calcular manualmente si intra_df no existe
        blocks   = statis.blocks_result
        p        = blocks.n_variables
        n_dims   = statis.n_dims_retained
        U_ret    = statis.eigenvectors[:, :n_dims]
        lam_ret  = statis.eigenvalues[:n_dims]
        escala   = 1.0 / np.sqrt(lam_ret)

        Xs_panel = np.vstack([blocks.panel_scaled[t].astype(float)
                               for t in statis.tablas])
        F_panel  = Xs_panel @ U_ret * escala

        corr = np.zeros((p, n_dims), dtype=float)
        for j in range(n_dims):
            for i in range(p):
                x = Xs_panel[:, i]; y = F_panel[:, j]
                mx, my = x.mean(), y.mean()
                num    = np.sum((x-mx)*(y-my))
                denom  = np.sqrt(np.sum((x-mx)**2)*np.sum((y-my)**2))
                corr[i,j] = float(num/denom) if denom > 1e-14 else 0.0

        result.correlations    = corr
        result.df_correlations = pd.DataFrame(
            corr, index=statis.variable_names,
            columns=result.dim_labels
        ).round(4)

    return result
