"""
M4_interstructure.py — Interestructura, compromiso y proyecciones
=================================================================
Implementación EXACTA del script validado 2__STATIS_DUAL_DEFINITIVO_14-06_.ipynb

Estructura STATIS Dual:
  K tablas  = países (la entidad de estudio)
  n_k filas = años por país (pueden variar, mínimo 3)
  p columnas = variables (indicadores QS)

  Xk = (n_k × p)              tabla estandarizada del país k
  Vk = (1/Tk)·Xk'·Xk / ||Vk|| operador dual normalizado (p×p)

Pipeline (fases del script validado):
  F7.  Matriz RV (K×K)
  F8.  αk = primer eigenvector de RV, orientado + , normalizado Σ=1
  F9.  S = Σ αk·Vk             compromiso (p×p)
  F10. eigh(S) → λ, U          filtrar λ > 1e-10
  F11. Fk = Xk @ U * (1/√λ)   scores parciales  ← escala correcta
  F12. COS² sobre scores parciales apilados, luego media por tabla
  F13. Loadings = U * √λ       vectores HJ-Biplot
  F13. Intraestructura: Pearson(Xs_panel, F_panel) SIN αk

Referencia: Lavit et al. (1994) Comput. Stat. Data Anal. 18(1), 97-119
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.m0_config import QR_MAX_ITER, QR_TOLERANCE, VECTOR_SCALE
from core.m3_blocks import BlocksResult, manual_eigen_symmetric


# ─── Coeficiente RV ──────────────────────────────────────────────────────────

def _rv(Va: np.ndarray, Vb: np.ndarray) -> float:
    """
    RV(A,B) = tr(A·B) / sqrt(tr(A·A) · tr(B·B))
    Robert & Escoufier (1976). Rango [0,1].
    """
    num   = np.trace(Va @ Vb)
    denom = np.sqrt(np.trace(Va @ Va) * np.trace(Vb @ Vb))
    return float(num / denom) if denom > 1e-14 else 0.0


def build_rv_matrix(blocks: BlocksResult) -> np.ndarray:
    """Matriz RV simétrica K×K.
    En modo bloques usa operadores en espacio de individuos (n×n).
    """
    K  = len(blocks.tablas)
    RV = np.eye(K, dtype=float)
    _bloques = getattr(blocks, 'modo', 'longitudinal') == 'bloques'

    if _bloques:
        # Construir Sk = (1/n) Xk @ Xk.T normalizado → todos (n×n)
        n_ind = blocks.n
        Sk_dict = {}
        for t in blocks.tablas:
            Xs  = blocks.panel_scaled[t].astype(float)
            Sk  = (Xs @ Xs.T) / n_ind
            nrm = np.sqrt(np.trace(Sk @ Sk))
            Sk_dict[t] = Sk / nrm if nrm > 1e-10 else Sk
        for j, tj in enumerate(blocks.tablas):
            for k, tk in enumerate(blocks.tablas):
                if j < k:
                    rv = _rv(Sk_dict[tj], Sk_dict[tk])
                    RV[j, k] = rv
                    RV[k, j] = rv
    else:
        for j, tj in enumerate(blocks.tablas):
            for k, tk in enumerate(blocks.tablas):
                if j < k:
                    rv = _rv(blocks.Vk_dict[tj], blocks.Vk_dict[tk])
                    RV[j, k] = rv
                    RV[k, j] = rv
    return RV


def compute_alpha_k(RV: np.ndarray, tablas: list) -> tuple:
    """
    FIX #3: αk exactamente como el script validado.
    Primer eigenvector de RV → negar si suma < 0 → normalizar Σ=1.
    Sin abs() adicional.
    """
    vals_rv, vecs_rv = manual_eigen_symmetric(RV)
    alpha = vecs_rv[:, 0].copy()
    if alpha.sum() < 0:
        alpha = -alpha
    alpha = alpha / alpha.sum()

    alpha_s = pd.Series(alpha, index=tablas).sort_values(ascending=False)
    ranked  = [(t, float(a), r+1)
               for r, (t, a) in enumerate(alpha_s.items())]
    return alpha, ranked


# ─── Resultado ───────────────────────────────────────────────────────────────

@dataclass
class StatisDualResult:
    # Interestructura
    RV_matrix:    Optional[np.ndarray] = None
    alpha_k:      Optional[np.ndarray] = None
    alpha_k_rank: Optional[list]       = None

    # Compromiso
    Vs:           Optional[np.ndarray] = None
    eigenvalues:  Optional[np.ndarray] = None   # λ filtrados > 1e-10
    eigenvectors: Optional[np.ndarray] = None   # U filtrados

    # Varianza
    variance_explained:  Optional[np.ndarray] = None
    variance_cumulative: Optional[np.ndarray] = None

    # Scores parciales por tabla: F_k[tabla] = (n_k × n_dims)
    # Escala: Xk @ U * (1/√λ)  — igual que el script validado
    F_k: dict = field(default_factory=dict)

    # Biplot
    # F_rows: coordenadas medias de cada FILA (año) sobre las tablas (p × n_dims)
    # F_vars: loadings = U * √λ  (p × n_dims)
    F_rows: Optional[np.ndarray] = None
    F_vars: Optional[np.ndarray] = None

    # COS²: calculado sobre scores parciales apilados, media por tabla
    cos2_rows: Optional[np.ndarray] = None   # (n_filas_unicas × n_dims)
    cos2_mean: Optional[np.ndarray] = None   # (n_filas_unicas,)

    # COS² por tabla (país): media del COS² de sus filas (años)
    cos2_por_tabla: Optional[pd.DataFrame] = None

    # COS² "estilo R" (ade4::statis) — similitud de la tabla COMPLETA (Wk/Vk)
    # contra el consenso TRUNCADO a n_dims (mismo rango que Loadings/Biplot),
    # vía trace(Wk_norm · S_trunc_norm). Es una
    # métrica DISTINTA de cos2_por_tabla: compara operadores completos
    # (8×8 o n×n), no proyecciones sobre un eje específico. Ver conversación
    # de verificación cruzada con ade4::withinpca()+statis() — no está en el
    # texto de Lavit et al. (1994), es una extensión propia de ade4 sobre el
    # marco del RV-coeficiente (Robert & Escoufier, 1976).
    cos2_consenso: Optional[pd.DataFrame] = None

    # Intraestructura
    intra_df: Optional[pd.DataFrame] = None

    # Metadatos
    n_dims_retained: int  = 2
    biplot_type:     str  = "HJ"
    tablas:          list = field(default_factory=list)
    row_names:       list = field(default_factory=list)
    variable_names:  list = field(default_factory=list)
    filas_por_tabla: dict = field(default_factory=dict)
    años_tabla:      dict = field(default_factory=dict)   # {tabla: [filas ordenadas]}

    # coord_df: scores parciales apilados (como en el script)
    coord_df: Optional[pd.DataFrame] = None

    # centroides: media de Dim1,Dim2 por tabla (país) — para el biplot
    centroides: Optional[pd.DataFrame] = None

    blocks_result: Optional[object] = None

    @property
    def bloques(self):
        return self.tablas


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_interstructure(blocks: BlocksResult,
                       n_dims: int = 2,
                       biplot_type: str = "HJ") -> StatisDualResult:
    """
    Pipeline STATIS Dual idéntico al script validado de la tesis.
    Fases F7 a F13.
    """
    res = StatisDualResult()
    res.n_dims_retained = n_dims
    res.biplot_type     = biplot_type
    res.tablas          = blocks.tablas
    res.variable_names  = blocks.variable_names
    # En modo bloques, filas_por_tabla puede estar vacío — usar años_tabla
    _fpor = getattr(blocks, 'filas_por_tabla', {})
    _atab = getattr(blocks, 'años_tabla', {})
    # Asegurar que filas_por_tabla esté lleno para todas las tablas
    for t in blocks.tablas:
        if t not in _fpor or not _fpor[t]:
            _fpor[t] = _atab.get(t, [])
    blocks.filas_por_tabla = _fpor

    res.filas_por_tabla = _fpor
    res.blocks_result   = blocks

    # Filas únicas ordenadas
    seen, all_rows = set(), []
    for t in blocks.tablas:
        for f in _fpor[t]:
            if f not in seen:
                all_rows.append(f)
                seen.add(f)
    res.row_names = all_rows

    # años_tabla
    res.años_tabla = {t: list(_fpor[t]) for t in blocks.tablas}

    K = len(blocks.tablas)
    p = blocks.n_variables

    # ── F7. Matriz RV ─────────────────────────────────────────────────────────
    RV = build_rv_matrix(blocks)
    res.RV_matrix = RV

    # ── F8. αk — FIX #3: sin abs(), solo negar si sum<0 ──────────────────────
    alpha, ranked = compute_alpha_k(RV, blocks.tablas)
    res.alpha_k      = alpha
    res.alpha_k_rank = ranked

    # ── F9. Compromiso S = Σ αk·Vk ───────────────────────────────────────────
    _modo_bloques = getattr(blocks, 'modo', 'longitudinal') == 'bloques'
    Wk_norm_dict = {}   # tabla -> matriz normalizada (para cos2_consenso, F9b)
    if _modo_bloques:
        # Bloques con p_k diferente → compromiso en espacio de individuos (n×n)
        n_ind = blocks.n
        S = np.zeros((n_ind, n_ind), dtype=float)
        for i, t in enumerate(blocks.tablas):
            Xs  = blocks.panel_scaled[t].astype(float)
            Sk  = (Xs @ Xs.T) / n_ind
            nrm = np.sqrt(np.trace(Sk @ Sk))
            if nrm > 1e-10: Sk = Sk / nrm
            Wk_norm_dict[t] = Sk
            S += alpha[i] * Sk
    else:
        _modo_bloques = False
        S = np.zeros((p, p), dtype=float)
        for i, t in enumerate(blocks.tablas):
            Wk_norm_dict[t] = blocks.Vk_dict[t]   # ya normalizado en m3_blocks.py
            S += alpha[i] * blocks.Vk_dict[t]
    res.Vs = S

    # ── F10. Eigendescomposición de S ────────────────────────────────────────
    if _modo_bloques:
        # S es n×n (puede ser 334×334) — numpy LAPACK es exacto y ~100x más rápido
        vals_np, vecs_np = np.linalg.eigh(S)
        # eigh devuelve en orden ascendente — invertir
        idx_np   = np.argsort(vals_np)[::-1]
        vals_s   = vals_np[idx_np]
        vecs_s   = vecs_np[:, idx_np]
    else:
        # S es p×p (típicamente 8×8) — usar eigen manual para reproducibilidad
        vals_s, vecs_s = manual_eigen_symmetric(S, max_iter=QR_MAX_ITER, tol=QR_TOLERANCE)
    pos      = vals_s > 1e-10
    lambda_  = vals_s[pos]
    U        = vecs_s[:, pos]
    n_dim_total = len(lambda_)

    res.eigenvalues  = lambda_
    res.eigenvectors = U

    total = lambda_.sum()
    if total < 1e-14:
        total = 1.0
    res.variance_explained  = lambda_ / total * 100
    res.variance_cumulative = np.cumsum(res.variance_explained)

    # Retener n_dims (máx n_dim_total)
    n_dims = min(n_dims, n_dim_total)
    res.n_dims_retained = n_dims
    dim_names = [f"Dim{i+1}" for i in range(n_dims)]

    # ── F10b. cos2_consenso — "valor tipológico" estilo ade4::statis ────────
    # cos2_R[k] = trace(Wk_norm · S_trunc_norm) — replica statis.R con precisión:
    # ade4 NO compara contra el compromiso S completo (todas las dimensiones),
    # compara contra una reconstrucción TRUNCADA de S usando solo las primeras
    # 'nf' dimensiones (statis$C.li, reconstruido con solo nf ejes — por
    # defecto nf=3 si no se especifica). Comparar contra el S completo daba
    # valores sistemáticamente más altos para países con estructura relevante
    # en ejes 4+ (ej. Cuba, Nicaragua salían ~0.12 más altos que en R) —
    # verificado contrastando contra statis1$cos2 de R línea por línea.
    #
    # Aquí usamos n_dims (la misma cantidad de ejes que el usuario ya eligió
    # para Loadings/Biplot/Scree — "Dimensiones a retener") en vez de un nf
    # fijo, para que sea consistente con el resto de la app. Si se quiere
    # comparar exacto contra una corrida en R con un nf específico, hay que
    # igualar ese número (el script de ejemplo usó nf=3 por defecto, al no
    # especificarlo en statis(kta1, scann=FALSE)).
    n_trunc = min(n_dims, U.shape[1])
    U_trunc = U[:, :n_trunc]
    lam_trunc = lambda_[:n_trunc]
    S_trunc = (U_trunc * lam_trunc) @ U_trunc.T
    nrm_S = np.sqrt(np.trace(S_trunc @ S_trunc))
    S_norm = S_trunc / nrm_S if nrm_S > 1e-14 else S_trunc
    cos2_consenso_vals = {
        t: float(np.trace(Wk_norm_dict[t] @ S_norm)) for t in blocks.tablas
    }
    cos2_consenso_df = pd.DataFrame(
        sorted(cos2_consenso_vals.items(), key=lambda kv: -kv[1]),
        columns=["Tabla", "cos2_consenso"]
    ).set_index("Tabla")
    res.cos2_consenso = cos2_consenso_df

    U_ret   = U[:, :n_dims]          # (p × n_dims)
    lam_ret = lambda_[:n_dims]       # (n_dims,)

    # ── F11. Scores parciales — escala 1/√λ (Lavit 1994) ────────────────────
    # CRÍTICO: calcular también Fk con TODAS las dims para COS² correcto.
    # El script usa n_dim = len(lambda_) → coord_df tiene todas las dims,
    # y dist2 = sum(F²_ij para j=1..n_dim_total) → COS² << 1 por dim.
    escala_dim     = 1.0 / np.sqrt(lam_ret)    # (n_dims,)      para biplot
    escala_all     = 1.0 / np.sqrt(lambda_)    # (n_dim_total,) para COS²
    dim_names_all  = [f"Dim{i+1}" for i in range(n_dim_total)]

    rows_coord = []
    for t in blocks.tablas:
        Xs      = blocks.panel_scaled[t].astype(float)
        filas_t = res.años_tabla[t]
        if _modo_bloques:
            # U_ret es (n_ind × n_dims): los scores son directamente U*sqrt(λ)
            # Cada bloque comparte los mismos scores de individuos
            Fk_ret = U_ret * np.sqrt(lam_ret)       # (n_ind × n_dims)
            Fk_all = U     * np.sqrt(lambda_)        # (n_ind × n_dim_total)
        else:
            Fk_ret = Xs @ U_ret * escala_dim         # (n_k × n_dims)
            Fk_all = Xs @ U     * escala_all         # (n_k × n_dim_total)
        res.F_k[t] = Fk_ret
        for ti, fila in enumerate(filas_t):
            row = {"tabla": t, "fila": fila}
            for di, dn in enumerate(dim_names):
                row[dn] = float(Fk_ret[ti, di])
            for di2 in range(n_dim_total):
                row[f"_a{di2}"] = float(Fk_all[ti, di2])
            rows_coord.append(row)

    coord_df = pd.DataFrame(rows_coord)
    # Garantizar orden cronológico dentro de cada tabla
    coord_df["fila_int"] = pd.to_numeric(coord_df["fila"], errors="coerce")
    coord_df = coord_df.sort_values(["tabla", "fila_int"]).reset_index(drop=True)
    coord_df = coord_df.drop(columns=["fila_int"])
    res.coord_df = coord_df

    # F_rows: coordenadas medias de cada FILA (año) sobre todas las tablas
    accum = defaultdict(list)
    for t in blocks.tablas:
        filas_t = res.años_tabla[t]
        for ti, fila in enumerate(filas_t):
            accum[fila].append(res.F_k[t][ti, :])

    F_rows_list = []
    for fila in res.row_names:
        coords = np.array(accum[fila])
        F_rows_list.append(coords.mean(axis=0))
    res.F_rows = np.array(F_rows_list)   # (n_filas_unicas × n_dims)

    # ── F12. COS² — Fase 12 exacta al script ────────────────────────────────
    # dist2 = Σ F²_ij sobre TODAS las dims (n_dim_total), no solo n_dims
    # Igual que: F_all = coord_df[dim_names_all].values en el script
    all_cols = [f"_a{i}" for i in range(n_dim_total)]
    F_all    = coord_df[all_cols].values           # (K*T_k × n_dim_total)
    dist2    = np.sum(F_all**2, axis=1)
    dist2    = np.where(dist2 == 0, np.nan, dist2)
    cos2_all = F_all**2 / dist2[:, np.newaxis]    # (K*T_k × n_dim_total)
    cos2_ret = cos2_all[:, :n_dims]               # solo las n_dims retenidas

    cos2_df_full = pd.DataFrame(cos2_ret, columns=dim_names)
    cos2_df_full["tabla"] = coord_df["tabla"].values
    cos2_df_full["fila"]  = coord_df["fila"].values

    # COS² por tabla (media sobre sus filas)
    cos2_por_tabla = (cos2_df_full.groupby("tabla")[dim_names]
                      .mean()
                      .round(4)
                      .sort_values("Dim1", ascending=False))
    res.cos2_por_tabla = cos2_por_tabla

    # COS² por fila
    cos2_por_fila = (cos2_df_full.groupby("fila")[dim_names].mean())
    F_rows_cos2 = np.array([
        cos2_por_fila.loc[f].values if f in cos2_por_fila.index
        else np.zeros(n_dims)
        for f in res.row_names
    ])
    res.cos2_rows = F_rows_cos2
    res.cos2_mean = F_rows_cos2.mean(axis=1)

    # ── F13a. Loadings HJ-Biplot: U * √λ ─────────────────────────────────────
    lam_sqrt = np.sqrt(lam_ret)
    if _modo_bloques:
        # Cargas por bloque: Lk = Xk.T @ U_ind_ret / sqrt(n_ind)
        all_loads = []
        for t in blocks.tablas:
            Xs_k = blocks.panel_scaled[t].astype(float)
            Lk   = Xs_k.T @ U_ret / np.sqrt(Xs_k.shape[0])  # (p_k × n_dims)
            all_loads.append(Lk)
        F_vars_raw = np.vstack(all_loads)   # (sum_pk × n_dims)
    else:
        F_vars_raw = U_ret * lam_sqrt[np.newaxis, :]   # (p × n_dims)

    # Escalar vectores al VECTOR_SCALE del espacio de F_rows
    max_row = np.max(np.abs(res.F_rows)) if res.F_rows.size > 0 else 1.0
    max_var = np.max(np.abs(F_vars_raw))  if F_vars_raw.size > 0 else 1.0
    if max_var > 1e-14:
        F_vars_raw = F_vars_raw * (VECTOR_SCALE * max_row / max_var)
    res.F_vars = F_vars_raw

    # ── F13b. Intraestructura ─────────────────────────────────────────────────
    if _modo_bloques:
        # En modo bloques: Xs_panel apila columnas de todos los bloques
        Xs_panel = np.hstack([blocks.panel_scaled[t].astype(float)
                               for t in blocks.tablas])  # (n × sum_pk)
    else:
        Xs_panel = np.vstack([blocks.panel_scaled[t].astype(float)
                               for t in blocks.tablas])
    if _modo_bloques:
        # En modo bloques: scores son U_ret (n_ind × n_dims), ya calculados
        F_panel = U_ret * np.sqrt(lam_ret)    # (n_ind × n_dims)
        p_intra = Xs_panel.shape[1]           # sum_pk columnas
    else:
        F_panel = Xs_panel @ U_ret * escala_dim
        p_intra = p

    intra = np.zeros((p_intra, n_dims), dtype=float)
    for j in range(n_dims):
        for i in range(p_intra):
            x = Xs_panel[:, i]
            y = F_panel[:, j]
            mx, my = x.mean(), y.mean()
            num    = np.sum((x - mx) * (y - my))
            denom  = np.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
            intra[i, j] = float(num / denom) if denom > 1e-14 else 0.0

    res.intra_df = pd.DataFrame(intra,
                                 index=blocks.variable_names,
                                 columns=dim_names).round(4)

    # Centroides por tabla: media de Dim1,Dim2 (igual que el script)
    dim_labels = [f"Dim{i+1}" for i in range(n_dims)]
    centroides = (coord_df.groupby("tabla")[dim_labels]
                  .mean()
                  .rename(columns={f"Dim{i+1}": f"C{i+1}"
                                   for i in range(n_dims)}))
    res.centroides = centroides

    return res
