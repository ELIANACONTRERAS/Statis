"""
M1_loader.py — Carga y validación del panel STATIS Dual
========================================================
Responsabilidades:
  • Leer Excel (.xlsx, .xls) o CSV (.csv, .txt separado por coma/punto y coma/tab)
  • Detectar y separar columnas numéricas vs cualitativas
  • Validar criterios mínimos según Lavit et al. (1994):
      - K ≥ MIN_TABLAS  (tablas distintos)
      - n_k ≥ MIN_FILAS  (filas por tabla)
      - p ≥ MIN_VARIABLES (variables numéricas)
      - cada tabla debe ser cuadrado posible: n_k ≥ p
  • Devolver un objeto DatasetInfo con todo lo necesario para la UI y el análisis

FIX (v1.0.1) — Manejo de NaN en modo bloques:
==============================================
build_blocks_from_bloques() extraía las columnas de cada bloque directamente
con `df[cols].values.astype(float)`, sin ningún tratamiento de valores
faltantes. Si una celda era NaN, numpy.mean() (usado más adelante en
m3_blocks.py para el centrado) propaga el NaN a la media de toda la columna,
contaminando Vk, RV, alpha_k y el compromiso completo para ese bloque —
en cascada, de forma silenciosa (sin lanzar ningún error visible).

Esto replica, para el modo bloques, el mismo tratamiento que ya existía en
build_clean_panel() para el modo longitudinal: imputar NaN con la media de
la propia columna, dentro del propio bloque (consistente con el criterio de
"Panel A" ya usado en el resto de la app).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from core.m0_config import MIN_TABLAS, MIN_FILAS, MIN_VARIABLES


# ─── Estructura de resultado ─────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    level: str        # "error" | "warning" | "info"
    message: str


@dataclass
class DatasetInfo:
    """Resultado completo de M1. La UI y los módulos posteriores consumen esto."""
    # Raw
    df_raw: pd.DataFrame = field(repr=False)
    filepath: str = ""
    filename: str = ""

    # Columnas detectadas
    all_columns: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)

    # Mapeo elegido por el usuario (se completa en la UI, M1 solo detecta candidatos)
    col_tabla: Optional[str] = None      # columna que define los tablas (ej. "año")
    col_fila: Optional[str] = None        # columna que identifica filas (ej. "pais")
    cols_variables: list = field(default_factory=list)  # variables numéricas a usar

    # Resumen estructural
    n_tablas: int = 0
    n_filas_por_tabla: dict = field(default_factory=dict)   # {tabla: n}
    n_variables: int = 0
    n_total_filas: int = 0

    # Estado
    is_valid: bool = False
    issues: list = field(default_factory=list)    # lista de ValidationIssue

    # Configuración de análisis (la UI la completa)
    standardize: bool = True
    zero_treatment: str = "keep"     # "keep" | "impute_mean"
    biplot_type: str = "HJ"          # "HJ" | "symmetric"
    n_dims: int = 2

    # Reescalado adicional de Bouroche (1975), citado en la documentación de
    # ade4::withinpca como la referencia de la estandarización "total":
    # tras centrar dentro de tabla, vuelve a dividir los residuos por una
    # desviación estándar GLOBAL calculada sobre esos mismos residuos
    # (agrupando todas las tablas). Es opcional — el centrado dentro de
    # tabla (obligatorio para STATIS) ya es correcto sin este paso; esto
    # solo cambia el peso relativo entre variables según cuánta de su
    # varianza es "entre tablas" vs "dentro de cada tabla". Solo aplica al
    # modo longitudinal (build_blocks()), no al modo bloques.
    bouroche_total: bool = False

    # ── Modo de análisis ──────────────────────────────────────────────────────
    # "longitudinal" (clásico): col_tabla + col_fila + cols_variables
    # "bloques"     (nuevo):    bloques_variables = [(nombre,[cols]), ...]
    modo: str = "longitudinal"
    bloques_variables: list = field(default_factory=list)
    col_id_individuo: Optional[str] = None
    # Estandarización modo bloques: "parcial" | "total" | "ninguna"
    std_type: str = "parcial"


# ─── Funciones de lectura ────────────────────────────────────────────────────

def _detect_separator(filepath: str) -> str:
    """Detecta el separador de un archivo CSV/TXT leyendo las primeras líneas."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    candidates = {
        ",": sample.count(","),
        ";": sample.count(";"),
        "\t": sample.count("\t"),
        "|": sample.count("|"),
    }
    return max(candidates, key=candidates.get)


def load_file(filepath: str) -> tuple[pd.DataFrame, str]:
    """
    Carga el archivo y devuelve (DataFrame, mensaje_error).
    Si hay error devuelve (None, mensaje).
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    try:
        if ext in (".xlsx", ".xls", ".xlsm"):
            df = pd.read_excel(filepath, sheet_name=0)
        elif ext in (".csv", ".txt"):
            sep = _detect_separator(filepath)
            df = pd.read_csv(filepath, sep=sep, encoding="utf-8", on_bad_lines="skip")
        else:
            return None, f"Formato no soportado: {ext}. Use .xlsx, .xls, .csv o .txt"

        # Limpiar nombres de columnas
        df.columns = [str(c).strip() for c in df.columns]

        # Eliminar filas completamente vacías
        df = df.dropna(how="all").reset_index(drop=True)

        if df.empty:
            return None, "El archivo está vacío o no contiene datos válidos."

        return df, ""

    except Exception as e:
        return None, f"Error al leer el archivo: {str(e)}"


# ─── Detección de tipos de columnas ─────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> tuple[list, list]:
    """
    Clasifica columnas en numéricas y cualitativas.
    Una columna es numérica si tiene dtype numérico Y más de 2 valores únicos.
    Devuelve (numeric_cols, categorical_cols).
    """
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        # Convertir StringDtype (pandas 2.x) antes de evaluar
        series = df[col]
        try:
            series = series.astype("object")
        except Exception:
            pass

        if pd.api.types.is_numeric_dtype(series):
            # Verificar que no sea un ID disfrazado de número
            n_unique = series.nunique()
            if n_unique >= 2:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            # Intentar conversión forzada
            converted = pd.to_numeric(series, errors="coerce")
            pct_converted = converted.notna().mean()
            if pct_converted >= 0.80:
                # ≥80% convertible → numérica (los demás son anotaciones)
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ─── Construcción de DatasetInfo inicial ────────────────────────────────────

def build_dataset_info(filepath: str) -> DatasetInfo:
    """
    Punto de entrada principal de M1.
    Carga el archivo y devuelve un DatasetInfo con detección automática.
    La validación completa se ejecuta en validate_mapping() una vez
    que el usuario asigna tabla / fila / variables.
    """
    info = DatasetInfo(df_raw=pd.DataFrame())
    info.filepath = filepath
    info.filename = Path(filepath).name

    df, error = load_file(filepath)
    if error:
        info.is_valid = False
        info.issues.append(ValidationIssue("error", error))
        return info

    info.df_raw = df
    info.all_columns = list(df.columns)
    info.n_total_filas = len(df)

    numeric_cols, cat_cols = detect_column_types(df)
    info.numeric_columns = numeric_cols
    info.categorical_columns = cat_cols

    info.issues.append(ValidationIssue(
        "info",
        f"Archivo cargado: {info.filename} — "
        f"{len(df)} filas × {len(df.columns)} columnas — "
        f"{len(numeric_cols)} numéricas, {len(cat_cols)} cualitativas"
    ))

    return info


# ─── Validación con mapeo definido por el usuario ───────────────────────────

def validate_mapping(info: DatasetInfo) -> DatasetInfo:
    """
    Ejecutar después de que el usuario asignó col_tabla, col_fila, cols_variables.
    Verifica todos los criterios mínimos STATIS Dual y actualiza info.is_valid.
    """
    info.issues = [i for i in info.issues if i.level == "info"]   # conservar infos previas
    errors = []
    warnings = []

    df = info.df_raw

    # ── Verificar que las columnas elegidas existen ──────────────────────────
    if not info.col_tabla:
        errors.append("No se definió la TABLAS.")
    if not info.col_fila:
        errors.append("No se definió la columna de FILAS (entidades).")
    if not info.cols_variables or len(info.cols_variables) < MIN_VARIABLES:
        errors.append(f"Seleccione al menos {MIN_VARIABLES} variables numéricas.")

    if errors:
        for e in errors:
            info.issues.append(ValidationIssue("error", e))
        info.is_valid = False
        return info

    # ── Verificar solapamiento de roles ─────────────────────────────────────
    roles = set([info.col_tabla, info.col_fila]) & set(info.cols_variables)
    if roles:
        errors.append(f"Columna(s) {roles} asignadas a más de un rol.")
        for e in errors:
            info.issues.append(ValidationIssue("error", e))
        info.is_valid = False
        return info

    # ── Forzar tipos numéricos en variables seleccionadas ───────────────────
    for col in info.cols_variables:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Tablas ──────────────────────────────────────────────────────────────
    tablas = sorted(df[info.col_tabla].dropna().unique())
    info.n_tablas = len(tablas)
    info.n_variables = len(info.cols_variables)

    if info.n_tablas < MIN_TABLAS:
        errors.append(
            f"Se requieren ≥ {MIN_TABLAS} tablas. "
            f"La columna '{info.col_tabla}' tiene solo {info.n_tablas} valores únicos."
        )

    # ── Filas por tabla ─────────────────────────────────────────────────────
    info.n_filas_por_tabla = {}
    filas_insuficientes = []

    for b in tablas:
        mask = df[info.col_tabla] == b
        sub = df.loc[mask, info.cols_variables].dropna(how="all")
        n_k = len(sub)
        info.n_filas_por_tabla[b] = n_k

        if n_k < MIN_FILAS:
            filas_insuficientes.append(f"  • Tabla '{b}': {n_k} filas (mínimo {MIN_FILAS})")

        # Advertir solo si n_k < p (Vk de rango deficiente)
        # NOTA: filas diferentes entre tablas es VÁLIDO en STATIS Dual
        if n_k < info.n_variables:
            warnings.append(
                f"Tabla '{b}': {n_k} filas < {info.n_variables} variables. "
                f"Vk será de rango deficiente — considere reducir variables."
            )

    if filas_insuficientes:
        errors.append(
            f"Tablas con filas insuficientes (< {MIN_FILAS}):\n" +
            "\n".join(filas_insuficientes)
        )

    # ── Columnas constantes o todo-NaN en algún tabla ───────────────────────
    for b in tablas:
        mask = df[info.col_tabla] == b
        sub = df.loc[mask, info.cols_variables]
        for col in info.cols_variables:
            if sub[col].nunique(dropna=True) <= 1:
                warnings.append(
                    f"Variable '{col}' en tabla '{b}' es constante o todo-NaN. "
                    f"Puede causar inestabilidad numérica."
                )

    # ── Empaquetar resultados ────────────────────────────────────────────────
    for e in errors:
        info.issues.append(ValidationIssue("error", e))
    for w in warnings:
        info.issues.append(ValidationIssue("warning", w))

    info.is_valid = len(errors) == 0

    if info.is_valid:
        info.issues.append(ValidationIssue(
            "info",
            f"✓ Panel válido: K={info.n_tablas} tablas, "
            f"p={info.n_variables} variables. "
            f"Filas por tabla: {dict(info.n_filas_por_tabla)}"
        ))

    return info


# ─── Construcción del panel limpio para el análisis ─────────────────────────

def build_clean_panel(info: DatasetInfo) -> dict:
    """
    Replica EXACTAMENTE las fases 3 y 4 del script validado de la tesis.

    Fase 3 — Tratamiento de valores especiales (según criterio):
      zero_treatment == "keep":
        df[vars].fillna(df[vars].mean())   ← Panel A: imputa solo NaN,
                                              ceros se mantienen como efectivos
      zero_treatment == "impute_mean":
        df[vars].replace(0, NaN).fillna(mean)  ← trata ceros como faltantes

    Fase 3b — Promedio por (tabla, fila):
        df.groupby([col_tabla, col_fila])[vars].mean()
        → Promedia múltiples observaciones con misma tabla+fila
        → Ej: universidades de un país en un año → un solo valor por país-año

    Fase 4 — Filtro mínimo de filas por tabla (min_filas=3):
        tablas_ok = tablas con n_filas_únicas >= 3
        → Elimina tablas con menos de 3 filas DESPUÉS del groupby

    Returns: {tabla: DataFrame con index=filas, cols=variables}
    """
    df = info.df_raw.copy()

    # Convertir variables a numérico
    for col in info.cols_variables:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fase 3 — Tratamiento según criterio (antes del groupby)
    if info.zero_treatment == "impute_mean":
        # Trata ceros como faltantes + imputa con media global
        df[info.cols_variables] = df[info.cols_variables].replace(0, np.nan)
        for col in info.cols_variables:
            df[col] = df[col].fillna(df[col].mean())
    else:
        # Panel A (por defecto): solo imputa NaN reales, ceros se conservan
        for col in info.cols_variables:
            df[col] = df[col].fillna(df[col].mean())

    # Fase 3b — Promedio por (tabla, fila) — replica groupby del script
    df_cy = (
        df.groupby([info.col_tabla, info.col_fila])[info.cols_variables]
        .mean()
        .reset_index()
    )
    # Garantizar tipo consistente en col_fila (evitar mezcla int/float)
    try:
        df_cy[info.col_fila] = df_cy[info.col_fila].astype(int)
    except (ValueError, TypeError):
        pass

    # Fase 4 — Filtro: solo tablas con ≥ MIN_FILAS filas únicas
    filas_por_tabla = df_cy.groupby(info.col_tabla)[info.col_fila].nunique()
    tablas_ok = sorted(
        filas_por_tabla[filas_por_tabla >= MIN_FILAS].index.tolist()
    )

    df_bal = df_cy[df_cy[info.col_tabla].isin(tablas_ok)].copy()

    # Construir panel {tabla: DataFrame}
    panel = {}
    for t in tablas_ok:
        sub = (
            df_bal[df_bal[info.col_tabla] == t]
            .set_index(info.col_fila)[info.cols_variables]
            .sort_index()
        )
        panel[t] = sub

    return panel


# ─── Modo bloques: panel de bloques de variables ──────────────────────────────

def validate_bloques(info: "DatasetInfo") -> None:
    """Valida el modo bloques y rellena info.n_tablas, etc."""
    from core.m0_config import MIN_TABLAS

    if not info.bloques_variables:
        raise ValueError("No se definieron bloques de variables.")
    if len(info.bloques_variables) < MIN_TABLAS:
        raise ValueError(
            f"Se necesitan al menos {MIN_TABLAS} bloques para STATIS "
            f"(con solo 2, alpha_k siempre da 50%/50% y no aporta información)."
        )
    df = info.df_raw
    for nombre, cols in info.bloques_variables:
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Columna '{c}' no encontrada (bloque '{nombre}').")
            if not pd.api.types.is_numeric_dtype(df[c]):
                raise ValueError(f"Columna '{c}' no es numérica (bloque '{nombre}').")
        if len(cols) < 2:
            raise ValueError(f"El bloque '{nombre}' necesita al menos 2 variables.")
    info.n_tablas   = len(info.bloques_variables)
    info.n_variables = max(len(cols) for _, cols in info.bloques_variables)
    info.n_total_filas = len(df)
    info.is_valid   = True


def build_blocks_from_bloques(info: "DatasetInfo") -> dict:
    """
    Modo bloques: devuelve {nombre_bloque: np.ndarray (n × p_k)}.
    Cada bloque es la submatriz de variables correspondiente.

    FIX: antes de convertir a numpy, se fuerza tipo numérico y se imputan
    los NaN con la media de su propia columna DENTRO del bloque (mismo
    criterio "Panel A" que build_clean_panel usa en modo longitudinal).
    Sin este paso, un solo valor faltante propaga NaN a toda la media del
    bloque en m3_blocks.py, contaminando Vk, RV, alpha_k y el compromiso
    de forma silenciosa (sin ningún error visible).

    Si una columna llega a estar COMPLETAMENTE vacía dentro de un bloque
    (todos NaN), no hay media posible para imputar — en ese caso se
    conserva el NaN y se deja que validate_bloques() / la capa de
    validación de la UI lo señale como advertencia, en vez de imputar
    silenciosamente con un valor arbitrario (p. ej. 0).
    """
    df = info.df_raw
    result = {}
    for nombre, cols in info.bloques_variables:
        sub = df[cols].apply(pd.to_numeric, errors="coerce")

        medias = sub.mean()   # media por columna, ignorando NaN (comportamiento de pandas)
        sub = sub.fillna(medias)

        X = sub.values.astype(float)
        result[nombre] = X
    return result


# ─── Puntajes brutos (dato original, previo a STATIS) ───────────────────────

def build_raw_scores_table(info: "DatasetInfo") -> pd.DataFrame:
    """
    Tabla de puntajes BRUTOS por (tabla, fila) — el dato original tal como
    viene del archivo cargado, ANTES de estandarizar/imputar y ANTES de
    cualquier proyección STATIS (no son coordenadas ni loadings de la
    descomposición espectral).

    Genérica: usa exactamente las columnas que el usuario mapeó en modo
    longitudinal (info.col_tabla, info.col_fila, info.cols_variables), sin
    asumir nombres fijos como 'AR_score', 'país' o 'año'. Funciona igual
    con cualquier base de datos que cumpla el mapeo TABLA/FILA/variables.

    Si hay varias observaciones para la misma combinación (tabla, fila),
    se promedian (misma agregación que Fase 3b del pipeline STATIS), para
    que la tabla resultante tenga una sola fila por combinación.
    """
    if getattr(info, "modo", "longitudinal") != "longitudinal":
        raise ValueError(
            "build_raw_scores_table solo aplica al modo longitudinal "
            "(requiere TABLA + FILA + variables definidos)."
        )

    df        = info.df_raw.copy()
    col_tabla = info.col_tabla
    col_fila  = info.col_fila
    cols_vars = list(info.cols_variables)

    for c in cols_vars:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tabla = (
        df.groupby([col_tabla, col_fila], as_index=False)[cols_vars]
          .mean()
          .sort_values([col_tabla, col_fila])
          .reset_index(drop=True)
    )
    return tabla
