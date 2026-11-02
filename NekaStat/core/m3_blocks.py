"""
M3_blocks.py — Construcción de Vk y solver numérico manual
===========================================================
Responsabilidades:
  • Calcular Vk = (1/Tk) · Xck' · Xck   (Lavit et al. 1994, p. 107, Ejemplo 1: Vk = (1/I)·Xk'Xk)
    donde Xck = Xk - media(Xk)  →  CENTRADO DENTRO DE CADA TABLA
  • Implementar QR iterativo manual con Gram-Schmidt modificado
  • Descomposición propia manual de matrices simétricas definidas positivas
  • Preparar las matrices para M4 (interestructura) y M5 (intraestructura)

NOTA CRÍTICA: No se usa scipy, sklearn ni numpy.linalg.eig/eigh.
El solver QR iterativo es el mismo validado en el script de 979 líneas.

FIX (v1.0.1) — Centrado dentro de tabla:
=========================================
La versión anterior calculaba Vk = (1/Tk)·Xk'·Xk directamente sobre Xk
SIN restar la media propia de cada tabla. Eso convierte a Vk en una matriz
de momentos de segundo orden NO centrados, que se descompone como:

    Xk'Xk/Tk  =  Covarianza_interna(tabla)  +  dk·dk'

donde dk es el vector de desvío de la media de la tabla respecto a la media
global. Ese segundo término mezcla el NIVEL de la tabla (p. ej. el promedio
general de un país en el ranking) con su estructura de covarianza real,
inflando artificialmente el RV entre tablas que simplemente comparten un
nivel parecido, aunque su dinámica interna sea distinta.

STATIS / STATIS Dual está definido matemáticamente sobre datos centrados
por tabla (operadores de Escoufier). Este fix agrega ese centrado antes de
construir Vk en build_blocks() y reestructura build_blocks_bloques() para
que el centrado por bloque sea SIEMPRE obligatorio, independientemente de
la opción de escalado (parcial / total / ninguna) que elija el usuario.
"""

import numpy as np
from dataclasses import dataclass, field

from core.m0_config import QR_MAX_ITER, QR_TOLERANCE
from core.m2_preprocessor import PreprocessResult


# ─── Solver QR manual ────────────────────────────────────────────────────────

def _gram_schmidt_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Descomposición QR por Gram-Schmidt modificado.
    A: matriz (m × n), m ≥ n
    Devuelve Q (m×n ortogonal), R (n×n triangular superior)
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(np.dot(v, v))
        if abs(R[j, j]) < 1e-14:
            Q[:, j] = 0.0
        else:
            Q[:, j] = v / R[j, j]

    return Q, R


def manual_eigen_symmetric(M: np.ndarray,
                            max_iter: int = QR_MAX_ITER,
                            tol: float = QR_TOLERANCE) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendescomposición de matriz simétrica M (p×p) por algoritmo QR iterativo.
    Devuelve (eigenvalores, eigenvectores) ordenados de mayor a menor.

    Garantías:
    - Solo usa numpy básico (dot, sqrt, abs)
    - Converge para matrices simétricas definidas positivas
    - Idéntico al solver del script validado
    """
    p = M.shape[0]
    Ak = M.copy().astype(float)
    Qk = np.eye(p, dtype=float)   # acumula los eigenvectores

    for _ in range(max_iter):
        Q, R = _gram_schmidt_qr(Ak)
        Ak_new = R @ Q
        Qk = Qk @ Q

        # Criterio de convergencia: off-diagonal de Ak
        off_diag = np.sqrt(np.sum(Ak_new**2) - np.sum(np.diag(Ak_new)**2))
        if off_diag < tol:
            Ak = Ak_new
            break
        Ak = Ak_new

    eigenvalues = np.diag(Ak)
    eigenvectors = Qk

    # Ordenar de mayor a menor
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# ─── SVD manual por eigendescomposición ─────────────────────────────────────

def manual_svd(X: np.ndarray,
               max_iter: int = QR_MAX_ITER,
               tol: float = QR_TOLERANCE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD manual de X (n×p) vía eigendescomposición de X'X.
    Devuelve U (n×p), S (p,), Vt (p×p).

    X = U · diag(S) · Vt
    """
    n, p = X.shape
    XtX = X.T @ X   # (p×p) simétrica semidefinida positiva

    # Eigenvectores de X'X → V
    eigenvalues, V = manual_eigen_symmetric(XtX, max_iter, tol)

    # Valores singulares: σ = sqrt(λ), clamp a 0
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

    # U: columnas = X·v / σ
    U = np.zeros((n, p), dtype=float)
    for j in range(p):
        if singular_values[j] > 1e-14:
            U[:, j] = X @ V[:, j] / singular_values[j]
        else:
            U[:, j] = 0.0

    Vt = V.T
    return U, singular_values, Vt


# ─── Construcción de Vk ──────────────────────────────────────────────────────

@dataclass
class BlocksResult:
    """Resultado de M3: matrices Vk y sus eigendescomposiciones."""
    Vk_dict: dict = field(default_factory=dict)       # {tabla: ndarray (p×p)}
    eigenvalues_k: dict = field(default_factory=dict) # {tabla: ndarray (p,)}
    eigenvectors_k: dict = field(default_factory=dict)# {tabla: ndarray (p×p)}
    Tk_dict: dict = field(default_factory=dict)       # {tabla: int n_k}

    # Metadatos heredados de M2
    tablas: list = field(default_factory=list)
    variable_names: list = field(default_factory=list)
    filas_por_tabla: dict = field(default_factory=dict)
    n_variables: int = 0
    panel_scaled: dict = field(default_factory=dict)  # referencia a M2 (lectura)
    panel_centered: dict = field(default_factory=dict)  # NUEVO: Xk centrado por tabla (lectura/depuración)
    bouroche_total: bool = False  # NUEVO: si se aplicó el reescalado de Bouroche (1975)


def build_blocks(prep: PreprocessResult, bouroche_total: bool = False) -> BlocksResult:
    """
    Construye Vk = (1/Tk) · Xck' · Xck para cada tabla k,
    donde Xck = Xk - media(Xk) es el CENTRADO DENTRO DE LA TABLA.

    Lavit et al. 1994, p. 107 (Ejemplo 1): normalización por Tk (número de filas del tabla k).
    El centrado dentro de tabla es un requisito estructural del operador de
    Escoufier — sin él, Vk deja de ser una matriz de covarianza y mezcla
    nivel con estructura (ver nota de FIX en el encabezado del módulo).

    Parámetro
    ---------
    prep : PreprocessResult de M2 — contiene panel_scaled con Xk como arrays numpy
           (ya escalado globalmente si standardize=True, pero SIN centrar por tabla)
    bouroche_total : bool
           Si True, aplica el SEGUNDO paso de la estandarización "total" de
           Bouroche (1975) — la misma referencia que cita ade4::withinpca()
           para su opción scaling="total". Tras centrar dentro de tabla, los
           residuos se vuelven a dividir por una desviación estándar GLOBAL
           calculada sobre esos mismos residuos (agrupando todas las tablas).
           Esto NO es necesario para que Vk sea una covarianza válida (el
           centrado ya garantiza eso) — es una decisión adicional de
           ponderación entre variables, opcional, para verificar/comparar
           resultados contra ade4::withinpca(scal="total").
           Si False (default), el comportamiento es idéntico al de antes.

    Devuelve
    --------
    BlocksResult con Vk, eigenvalores y eigenvectores por tabla
    """
    result = BlocksResult()
    result.tablas = prep.tablas
    result.variable_names = prep.variable_names
    result.n_variables = prep.n_variables
    result.filas_por_tabla = prep.filas_por_tabla
    # Copia propia (no alias) de prep.panel_scaled: más abajo la
    # sobrescribimos con la versión centrada (y opcionalmente re-escalada),
    # y no queremos mutar por referencia el diccionario original de
    # PreprocessResult (m2), que podría seguir usándose en otra parte con
    # su significado original (solo z-score global, sin centrar por tabla).
    result.panel_scaled = dict(prep.panel_scaled)

    # ── Pasada 1: centrado dentro de tabla (siempre, obligatorio) ────────────
    Xk_c_dict = {}
    for b in prep.tablas:
        Xk = prep.panel_scaled[b].astype(float)   # (n_k × p) — escalado global, sin centrar por tabla
        Tk = Xk.shape[0]
        result.Tk_dict[b] = Tk

        media_tabla = Xk.mean(axis=0)
        Xk_c_dict[b] = Xk - media_tabla

    # ── Paso opcional: reescalado de Bouroche (1975) ─────────────────────────
    # Segundo paso de la estandarización "total" citada en ade4::withinpca():
    # divide los residuos (ya centrados por tabla) por una sd GLOBAL,
    # calculada sobre la concatenación de todos los residuos de todas las
    # tablas. Esto iguala el peso de las variables cuya varianza es
    # principalmente "entre tablas" (poco residuo tras centrar) con las que
    # varían principalmente "dentro de cada tabla" (mucho residuo).
    if bouroche_total:
        residuos_todos = np.vstack([Xk_c_dict[b] for b in prep.tablas])
        sd_residual_global = residuos_todos.std(axis=0, ddof=1)
        sd_residual_global = np.where(sd_residual_global < 1e-12, 1.0, sd_residual_global)
        for b in prep.tablas:
            Xk_c_dict[b] = Xk_c_dict[b] / sd_residual_global

    # ── Pasada 2: guardar centrado final y construir Vk ──────────────────────
    for b in prep.tablas:
        Xk_c = Xk_c_dict[b]
        Tk = result.Tk_dict[b]
        result.panel_centered[b] = Xk_c

        # IMPORTANTE: panel_scaled también se actualiza al centrado (y, si
        # aplica, al reescalado de Bouroche). m4_interstructure.py (F13b,
        # intraestructura; y el fallback de m5_intrastructure.py) leen
        # blocks.panel_scaled directamente para proyectar sobre el
        # compromiso. Si panel_scaled se quedara con la versión SIN centrar
        # (solo z-score global de m2), la intraestructura quedaría
        # inconsistente con Vk/compromiso (que sí usan el centrado),
        # desalineando la proyección Xs_panel @ U_ret. Sobrescribirlo aquí
        # garantiza que cualquier consumidor de panel_scaled — actual o
        # futuro — reciba datos ya centrados (y re-escaladas si aplica) por
        # tabla, sin tener que tocar m4/m5 ni recordar usar panel_centered
        # en cada punto de consumo.
        result.panel_scaled[b] = Xk_c

        # Vk = (1/Tk) · Xck' · Xck  normalizado por su norma (script validado)
        Vk  = (1.0 / Tk) * (Xk_c.T @ Xk_c)
        nrm = np.sqrt(np.trace(Vk @ Vk))
        Vk  = Vk / nrm if nrm > 1e-14 else Vk
        result.Vk_dict[b] = Vk

        # Eigendescomposición manual
        evals, evecs = manual_eigen_symmetric(Vk)
        result.eigenvalues_k[b] = evals
        result.eigenvectors_k[b] = evecs

    result.bouroche_total = bouroche_total
    return result


# ─── STATIS modo bloques (diferentes variables, mismos individuos) ────────────

def build_blocks_bloques(raw_blocks: dict,
                          standardize: bool = True,
                          std_type: str = "parcial") -> "BlocksResult":
    """
    raw_blocks: {nombre: np.ndarray (n × p_k)}

    El CENTRADO por columna dentro de cada bloque es SIEMPRE obligatorio
    (requisito estructural de STATIS), independientemente de std_type.
    std_type controla únicamente el ESCALADO (normalización de varianza):

        "parcial" — cada bloque usa su propia sd por columna (además de su
                    propia media). Equivale a estandarización PARCIAL en
                    ade4/SensoMineR de R.
        "total"   — todos los bloques se centran con su PROPIA media, pero
                    se escalan con una sd ÚNICA calculada sobre la
                    concatenación de todos los bloques. Equivale a
                    estandarización TOTAL en ade4 (scal="total"), donde el
                    centrado siempre es "within" y solo la escala es global.
        "ninguna" — se centra por columna dentro de cada bloque, pero no
                    se escala (sin división por sd). Antes esta opción no
                    centraba en absoluto; ahora sí lo hace, porque el
                    centrado no es opcional en STATIS.
    """
    import numpy as np

    nombres  = list(raw_blocks.keys())
    matrices = list(raw_blocks.values())
    n        = matrices[0].shape[0]

    for nm, X in zip(nombres, matrices):
        if X.shape[0] != n:
            raise ValueError(f"Bloque '{nm}' tiene {X.shape[0]} filas, esperaba {n}.")

    # ── Paso 1: CENTRADO por bloque — siempre, sin excepción ────────────────
    medias_bloque = [X.astype(float).mean(axis=0) for X in matrices]
    Xc_list = [X.astype(float) - mu for X, mu in zip(matrices, medias_bloque)]

    # ── Paso 2: ESCALADO — depende de std_type, opera sobre los datos YA centrados ──
    Xs_list = []

    if not standardize or std_type == "ninguna":
        Xs_list = [Xc.copy() for Xc in Xc_list]

    elif std_type == "total":
        # sd única calculada sobre la concatenación de los bloques YA centrados
        X_all = np.hstack(Xc_list)               # (n × Σpk)
        sd_all = X_all.std(axis=0, ddof=1)
        sd_all = np.where(sd_all < 1e-10, 1.0, sd_all)
        col = 0
        for Xc in Xc_list:
            p_k = Xc.shape[1]
            Xs_list.append(Xc / sd_all[col:col + p_k])
            col += p_k

    else:  # "parcial" (default) — cada bloque con su propia sd
        for Xc in Xc_list:
            sd = Xc.std(axis=0, ddof=1)
            sd = np.where(sd < 1e-10, 1.0, sd)
            Xs_list.append(Xc / sd)

    # Operadores duales Wk = (1/n) Xk' Xk / ||Wk||  (Xk ya centrado y escalado)
    Wk_list = []
    for Xs in Xs_list:
        W = (Xs.T @ Xs) / n
        nrm = np.sqrt(np.trace(W @ W))
        Wk_list.append(W / nrm if nrm > 1e-10 else W)

    # Empaquetar en BlocksResult — compatible con m4_interstructure
    result = BlocksResult()
    result.tablas       = nombres
    result.panel_scaled = {nm: Xs for nm, Xs in zip(nombres, Xs_list)}
    result.panel_centered = {nm: Xc for nm, Xc in zip(nombres, Xc_list)}
    # Vk_dict = alias de Wk para compatibilidad con build_rv_matrix y m4
    result.Vk_dict      = {nm: Wk for nm, Wk in zip(nombres, Wk_list)}
    result.Tk_dict      = {nm: n  for nm  in nombres}
    # filas_por_tabla: en modo bloques las "filas" son los índices de individuos
    result.filas_por_tabla = {nm: list(range(n)) for nm in nombres}
    result.años_tabla      = {nm: list(range(n)) for nm in nombres}
    result.row_names       = list(range(n))
    # n_variables = máximo p_k entre bloques (para pre-reservar arrays en m4)
    result.n_variables = max(X.shape[1] for X in matrices)
    # variable_names: lista plana de todas las variables de todos los bloques
    # (se sobreescribe en el worker con los nombres reales)
    result.variable_names = [f"var_{i}" for i in range(result.n_variables)]
    result.variable_names_per_block = {nm: None for nm in nombres}
    result.n    = n
    result.modo = "bloques"

    return result
