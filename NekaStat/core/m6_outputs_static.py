"""
M6_outputs_static.py — Gráficos estáticos (matplotlib/seaborn)
===============================================================
Genera todas las figuras para publicación Q1:
  G01 · HJ-Biplot / Biplot simétrico (compromiso)
  G02 · Trayectorias por entidad (grilla 4×n)
  G03 · Pesos αk (barras horizontales)
  G04 · COS² heatmap
  G05 · Heatmap RV
  G06 · Scree plot (varianza explicada)
  G07 · Intraestructura heatmap

Estándares:
  - 600 dpi, Arial / fallback automático
  - Doble columna 7.0 in (simple) / 14.0 in (doble)
  - Navy/Gold/Teal palette (M0_config)
  - Sin scipy, sin sklearn

Cada función devuelve el objeto Figure de matplotlib (para mostrar en UI
y para exportar a PNG + TIFF desde la UI).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend no interactivo — compatible con PyQt6
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False

# Configuración global de matplotlib — sin tocar font.family para evitar
# conflictos con Qt en Windows (QFont::setPointSize error)
matplotlib.rcParams.update({
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       96,
})

from core.m0_config import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT,
    COLOR_BG, COLOR_TEXT, COLOR_TEXT_LIGHT,
    EXPORT_DPI, FIG_WIDTH_SINGLE, FIG_WIDTH_DOUBLE,
    TRAJECTORY_PALETTE, FONT_PRIORITY,
    COLOR_ECUADOR, COLOR_CERCANOS, COLOR_LATAM, COLOR_VARS,
    COLOR_TIER_ALTO, COLOR_TIER_MEDIO, COLOR_TIER_BAJO,
)
from core.m4_interstructure import StatisDualResult, biplot_label, biplot_cita, CALC_VERSION
from core.m5_intrastructure import IntrastructureResult
from core.m3_blocks import manual_eigen_symmetric
# manual_eigen implementado inline con numpy para evitar imports circulares


# ─── Utilidades comunes ──────────────────────────────────────────────────────

def _get_font():
    """No modifica font.family — evita QFont::setPointSize en Windows."""
    return None

def _style_axes(ax, title="", xlabel="", ylabel=""):
    """Estilo base para todos los ejes."""
    ax.set_title(title, color=COLOR_PRIMARY, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=COLOR_TEXT, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E0")
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, color="#CBD5E0")


def _add_crosshair(ax):
    """Líneas de referencia en 0,0."""
    ax.axhline(0, color="#A0AEC0", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#A0AEC0", linewidth=0.8, linestyle="--", zorder=1)


# ─── G01 · HJ-Biplot / Biplot simétrico ─────────────────────────────────────

# Paleta de 30 colores para países
PALETA_PAISES = [
    '#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261',
    '#6A4C93','#1D3557','#E76F51','#264653','#A8DADC',
    '#F72585','#4361EE','#3A86FF','#8338EC','#FB5607',
    '#006400','#FF006E','#8AC926','#1982C4','#FFCA3A',
    '#6A994E','#BC6C25','#540B0E','#023E8A','#7209B7','#B5179E',
    '#0077B6','#D62828','#F77F00','#52B788',
]


def _etiqueta_punto_safe(ax, x, y, texto, color, fontsize=8, lim=1.0):
    """Etiqueta desplazada radialmente con fondo blanco."""
    r = np.hypot(x, y)
    nx, ny = (x/r, y/r) if r > 1e-9 else (1.0, 0.0)
    off = 0.04 * lim * 2
    ha  = "left"   if nx >= 0 else "right"
    va  = "bottom" if ny >= 0 else "top"
    ax.text(x + nx*off, y + ny*off, texto,
            fontsize=fontsize, color=color, fontweight="bold",
            ha=ha, va=va, zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.80, edgecolor="none"))


def plot_hjbiplot(statis: StatisDualResult,
                  dim_x: int = 0,
                  dim_y: int = 1) -> plt.Figure:
    """
    HJ-Biplot del Compromiso STATIS Dual.
    - Un punto por tabla (centroide = media de sus filas)
    - Cada punto lleva un NÚMERO pequeño (no el nombre completo) y una
      columna lateral lista "N — Nombre" en el mismo color del punto.
      Esto evita solapamiento de texto sin importar cuántas tablas haya
      ni qué tan largos sean sus nombres (país o universidad) — con
      texto inline (incluso reubicado con adjustText) no hay forma de
      evitar el solape cuando hay muchos puntos densos y nombres largos;
      un índice numérico sí lo garantiza siempre.
    - Vectores de variables en negro
    - Sin trayectorias, sin leyenda de buckets
    - Color DISTINTO por tabla (PALETA_PAISES). Todas las tablas —incluido
      Ecuador si aplica— se dibujan exactamente igual: mismo marcador,
      mismo tamaño. Ninguna se resalta por separado.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    ve       = statis.variance_explained
    d1_pct   = ve[dim_x]
    d2_pct   = ve[dim_y]
    G        = statis.F_vars
    var_names= statis.variable_names

    # Centroides por tabla (media de coord_df agrupado por tabla)
    coord_df = statis.coord_df
    dim1_col = f"Dim{dim_x+1}"
    dim2_col = f"Dim{dim_y+1}"
    centroides = (coord_df.groupby("tabla")[[dim1_col, dim2_col]]
                  .mean()
                  .rename(columns={dim1_col:"C1", dim2_col:"C2"}))

    # ── Reescalado de centroides para GARANTIZAR visibilidad ─────────────────
    # coord_df guarda las coordenadas de las filas según statis.biplot_type
    # (m4, F11): "HJ" → Fk = wk·Xk·U (HJ del compromiso); "GH" → Fk = Xk·U/√λ.
    # Son las mismas que usan trayectorias y el export "Coords".
    # Multiplicar por λ (Fk·(1/√λ)·λ = Fk·√λ) buscaba igualar la escala de
    # individuos a la de variables (√λ, convención HJ-Biplot) — pero si λ es
    # pequeño, esto puede dejar los centroides casi invisibles frente a los
    # vectores (que NO se reescalan por este mismo factor), sin que ningún
    # error aparezca — solo puntos apretados cerca del origen.
    #
    # En vez de esa multiplicación algebraica (frágil ante la magnitud real
    # de λ en cada corrida), autoescalamos los centroides para ocupar una
    # fracción fija y garantizada del gráfico (mismo criterio ya usado para
    # los vectores vía 'fac' más abajo) — prioriza que los países SIEMPRE
    # se vean, preservando sus distancias/ángulos relativos entre sí.
    # ── FIX CRÍTICO: los "centroides" (promedio de cada tabla) son SIEMPRE
    # (0,0) por construcción matemática, no por un bug de escala o dibujo.
    #
    # Cada tabla se centra restando su PROPIA media antes de proyectar
    # (Xk_c = Xk - media(Xk), el fix de centrado obligatorio de STATIS).
    # Como Fk = Xk_c · U · escala es una transformación LINEAL de Xk_c, el
    # promedio de Fk sobre esas mismas filas hereda la media cero de Xk_c:
    #
    #   promedio(Fk, filas propias) = promedio(Xk_c) · U · escala = 0
    #
    # Esto se cumple SIEMPRE, para cualquier tabla, sin excepción — no hay
    # reescalado posible que lo corrija (amplificar cero sigue dando cero).
    # Por eso ningún país aparecía: los 24 puntos caían apilados en el
    # origen. La corrección real es no promediar — graficar los puntos
    # INDIVIDUALES (cada combinación tabla-fila, ej. país-año), que sí
    # varían y sí se alejan del origen.
    dim1_col = f"Dim{dim_x+1}"
    dim2_col = f"Dim{dim_y+1}"
    puntos = coord_df[["tabla", "fila", dim1_col, dim2_col]].rename(
        columns={dim1_col: "C1", dim2_col: "C2"}
    ).copy()

    # ── Reescalado de puntos para GARANTIZAR visibilidad ─────────────────
    c1_raw = puntos["C1"].values.astype(float)
    c2_raw = puntos["C2"].values.astype(float)
    lmax_c = max(np.sqrt((c1_raw**2 + c2_raw**2)).max(), 1e-10)
    fac_c  = 0.75 / lmax_c   # ocupa ~75% del radio del gráfico
    puntos["C1"] = c1_raw * fac_c
    puntos["C2"] = c2_raw * fac_c

    tablas_sorted = sorted(puntos["tabla"].unique())
    color_tabla = {t: PALETA_PAISES[i % len(PALETA_PAISES)]
                   for i, t in enumerate(tablas_sorted)}
    numero_tabla = {t: i + 1 for i, t in enumerate(tablas_sorted)}

    # Límite del gráfico
    all_vals = (list(puntos["C1"]) + list(puntos["C2"]) +
                list(G[:, dim_x]) + list(G[:, dim_y]))
    lim_b = max(abs(v) for v in all_vals) * 1.30
    lim_b = max(lim_b, 0.1)

    # Figura con 2 columnas: biplot (ancha) + lista numerada (angosta).
    # El ancho de la columna de lista se adapta a la longitud real de los
    # nombres, y el alto total de la figura crece con la cantidad de tablas
    # — así funciona igual con K=24 países o K=22 (o más) universidades.
    n_tablas   = len(tablas_sorted)
    max_name_len = max((len(str(t)) for t in tablas_sorted), default=10)
    legend_w   = max(2.8, 1.0 + max_name_len * 0.075)
    fig_h      = max(9, 3.0 + n_tablas * 0.22)
    fig = plt.figure(figsize=(9 + legend_w, fig_h), dpi=96)
    gs  = fig.add_gridspec(1, 2, width_ratios=[9, legend_w], wspace=0.02)
    ax        = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)

    # ── Puntos individuales (tabla-fila, ej. país-año) ──────────────────────
    # Un punto por cada fila real de coord_df — NO un promedio por tabla
    # (ver nota arriba: ese promedio es siempre cero por construcción).
    # Coloreados por tabla, todos del mismo tamaño/forma.
    for tabla, sub in puntos.groupby("tabla"):
        col = color_tabla.get(tabla, "#4A90D9")
        num = numero_tabla[tabla]
        ax.scatter(sub["C1"], sub["C2"], s=55, color=col, marker="o",
                   edgecolors="white", linewidths=0.6, alpha=0.85, zorder=5)
        # Etiqueta numérica en el centroide VISUAL (promedio de las
        # posiciones YA graficadas) solo para ubicar el número — no se usa
        # para ningún cálculo, es puramente para orientar la lectura.
        cx_lbl, cy_lbl = sub["C1"].mean(), sub["C2"].mean()
        ax.scatter(cx_lbl, cy_lbl, s=90, color=col, marker="o",
                   edgecolors="white", linewidths=0.9, zorder=6)
        ax.text(cx_lbl, cy_lbl, str(num), fontsize=6.5, color="white",
                fontweight="bold", ha="center", va="center", zorder=7)

    # ── Columna lateral: lista numerada "N — Nombre" en el color del punto ──
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    y0 = 0.99
    dy = 1.0 / max(n_tablas, 1)
    for tabla in tablas_sorted:
        col = color_tabla[tabla]
        num = numero_tabla[tabla]
        ax_legend.text(0.0, y0, f"{num}", fontsize=7.5, color=col,
                       fontweight="bold", ha="left", va="top",
                       transform=ax_legend.transAxes)
        ax_legend.text(0.09, y0, str(tabla), fontsize=7.5, color=col,
                       ha="left", va="top", transform=ax_legend.transAxes)
        y0 -= dy

    # ── Vectores de variables en negro ───────────────────────────────────────
    lmax = np.sqrt((G**2).sum(axis=1)).max()
    fac  = (lim_b * 0.72) / lmax if lmax > 1e-10 else 1.0
    # Factor total aplicado a las cargas U·√λ (F_vars ya viene reescalado en m4)
    _L0 = getattr(statis, "loadings", None)
    if _L0 is not None and np.size(_L0) and np.abs(_L0).max() > 1e-14:
        c_vec = fac * float(np.abs(G).max() / np.abs(_L0).max())
    else:
        c_vec = fac
    fig.text(0.01, 0.005,
             f"Puntos multiplicados por k = {fac_c:.4g} y vectores (U·√λ) por c = {c_vec:.4g} "
             "solo para su visualización; los factores comunes no alteran ángulos ni proyecciones "
             "relativas. Coordenadas reales en la hoja 'Coords'.",
             fontsize=7, color="#555555", ha="left", va="bottom")
    for j, vname in enumerate(var_names):
        gx = G[j, dim_x] * fac
        gy = G[j, dim_y] * fac
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color="black",
                                    lw=1.6,
                                    mutation_scale=12),
                    zorder=7)
        nombre = vname.replace("_score", "")
        r2 = np.hypot(gx, gy)
        nx2, ny2 = (gx/r2, gy/r2) if r2 > 1e-9 else (1.0, 0.0)
        off2 = 0.07 * lim_b
        ax.text(gx + nx2*off2, gy + ny2*off2, nombre,
                fontsize=9, color="black", fontweight="bold",
                ha="left" if nx2 >= 0 else "right",
                va="bottom" if ny2 >= 0 else "top",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xlim(-lim_b, lim_b)
    ax.set_ylim(-lim_b, lim_b)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dimensión {dim_x+1}  ({d1_pct:.1f}% varianza)", fontsize=12)
    ax.set_ylabel(f"Dimensión {dim_y+1}  ({d2_pct:.1f}% varianza)", fontsize=12)
    etiqueta = biplot_label(statis)
    fig.suptitle(f"{etiqueta} del Compromiso — K={len(statis.tablas)} tablas\n"
                 f"{biplot_cita(statis)} · Lavit et al. (1994)",
                 fontsize=12, fontweight="semibold")
    ax.spines[["top","right"]].set_visible(False)

    # Márgenes fijos (no solo tight_layout): con aspect="equal" + título de
    # 2 líneas, tight_layout() no siempre reserva suficiente espacio arriba/
    # abajo, y al exportar sin bbox_inches="tight" (fix de figuras achatadas)
    # el título y la etiqueta del eje Dim1 quedaban cortados.
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.09)
    return fig


# ─── G01-VERIF · Verificación matemática del compromiso por concatenación ──
# Gráfico ADICIONAL, independiente de plot_hjbiplot (que queda intacto).
# Demuestra Zc'·Zc = Wc reconstruyendo el compromiso por una vía distinta a
# la ya validada (S = Σ αk·Vk calculado de forma incremental en m4).
#
#   Zc = [√α1·Z1 | √α2·Z2 | ... | √αK·ZK]   (concatenación vertical, ya que
#         las tablas comparten las mismas p variables — Vk = (1/Tk)Xk'Xk)
#   Zc' · Zc = Σ αk · Zk'·Zk = Wc
#
# Cada Zk se construye sobre blocks.panel_scaled[t], que YA viene centrado
# por tabla (m3_blocks.build_blocks sobrescribe panel_scaled con Xk_c, ver
# nota en ese módulo) y, opcionalmente, reescalado por Bouroche (1975) si
# el usuario activó esa opción — misma transformación desde el inicio,
# para que todo sea coherente con lo que realmente calculó m3_blocks.
#
# IMPORTANTE (mismo fix que plot_hjbiplot): el promedio de los puntos de
# una tabla es SIEMPRE cero por construcción (Xk_c tiene media cero antes
# de proyectar), así que aquí también se grafican los puntos INDIVIDUALES
# (tabla-fila), no un centroide por tabla.

def build_concatenated_compromise(statis: StatisDualResult):
    """
    Reconstruye el compromiso Wc por concatenación (ver nota arriba) y lo
    contrasta numéricamente contra el Wc ya validado (S = Σ αk·Vk de m4).

    Devuelve (Zc, S_concatenacion, S_original, diff_max, filas_totales,
    U_verif, lam_verif) — U_verif y lam_verif salen de una eigendescomposición
    FRESCA e independiente de Zc'·Zc (mismo solver manual QR de m3_blocks),
    no reutiliza los eigenvectores ya calculados en m4.
    """
    blocks = statis.blocks_result
    alpha_dict = {t: a for t, a, _ in statis.alpha_k_rank}

    Zc_partes  = []
    filas_totales = []
    S_original = np.zeros((blocks.n_variables, blocks.n_variables))

    for t in blocks.tablas:
        Xk = blocks.panel_scaled[t].astype(float)  # ya centrado por tabla (y Bouroche si aplica)
        Tk = blocks.Tk_dict[t]
        ak = alpha_dict[t]

        Vk_bruto = (1.0 / Tk) * (Xk.T @ Xk)
        nrm = np.sqrt(np.trace(Vk_bruto @ Vk_bruto))
        Vk_normalizado = Vk_bruto / nrm if nrm > 1e-14 else Vk_bruto

        S_original += ak * Vk_normalizado

        factor = np.sqrt(ak / (Tk * nrm)) if nrm > 1e-14 else 0.0
        Zk = Xk * factor
        Zc_partes.append(Zk)
        filas_totales.extend([(t, f) for f in blocks.filas_por_tabla[t]])

    Zc = np.vstack(Zc_partes)
    S_concatenacion = Zc.T @ Zc

    diff_max = float(np.max(np.abs(S_concatenacion - S_original)))

    lam_verif, U_verif = manual_eigen_symmetric(S_concatenacion)

    return (Zc, S_concatenacion, S_original, diff_max, filas_totales,
            U_verif, lam_verif)


def plot_biplot_verificacion(statis: StatisDualResult,
                             dim_x: int = 0,
                             dim_y: int = 1) -> plt.Figure:
    """
    Biplot de VERIFICACIÓN — gráfico adicional, no reemplaza plot_hjbiplot.

    Reconstruye el compromiso Wc mediante concatenación de las tablas
    estandarizadas ponderadas por √αk (ver build_concatenated_compromise),
    con una eigendescomposición fresca e independiente de la ya validada.
    Muestra en el título la diferencia máxima entre ambas construcciones de
    Wc — debe ser ≈ 0, confirmando la identidad Zc'·Zc = Σ αk·Vk = Wc.

    Grafica puntos INDIVIDUALES (tabla-fila), no centroides — el promedio
    por tabla es matemáticamente cero aquí también, por la misma razón que
    en plot_hjbiplot (datos centrados por tabla antes de proyectar).
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    (Zc, S_concat, S_orig, diff_max, filas_totales,
     U_verif, lam_verif) = build_concatenated_compromise(statis)

    n_dims = max(dim_x, dim_y) + 1
    lam_ret = lam_verif[:n_dims]
    U_ret   = U_verif[:, :n_dims]
    var_total = float(np.sum(lam_verif[lam_verif > 0]))
    ve = (lam_ret / var_total * 100.0) if var_total > 1e-14 else lam_ret * 0

    # Coordenadas de TODOS los puntos individuales — F_verif = Zc @ U_verif
    F_verif = Zc @ U_ret
    puntos = pd.DataFrame(F_verif[:, [dim_x, dim_y]], columns=["C1", "C2"])
    puntos["tabla"] = [t for t, _ in filas_totales]

    # Loadings de variables en la MISMA escala HJ-Biplot (√λ) que el biplot
    # ya validado, para que ambos gráficos sean comparables visualmente.
    G_verif = U_verif[:, :n_dims] * np.sqrt(np.maximum(lam_ret, 0))

    tablas_sorted = sorted(puntos["tabla"].unique())
    color_tabla = {t: PALETA_PAISES[i % len(PALETA_PAISES)]
                   for i, t in enumerate(tablas_sorted)}
    numero_tabla = {t: i + 1 for i, t in enumerate(tablas_sorted)}

    all_vals = (list(puntos["C1"]) + list(puntos["C2"]) +
                list(G_verif[:, dim_x]) + list(G_verif[:, dim_y]))
    lim_b = max(abs(v) for v in all_vals) * 1.30
    lim_b = max(lim_b, 0.1)

    n_tablas = len(tablas_sorted)
    max_name_len = max((len(str(t)) for t in tablas_sorted), default=10)
    legend_w = max(2.8, 1.0 + max_name_len * 0.075)
    fig_h = max(9, 3.0 + n_tablas * 0.22)
    fig = plt.figure(figsize=(9 + legend_w, fig_h), dpi=96)
    gs  = fig.add_gridspec(1, 2, width_ratios=[9, legend_w], wspace=0.02)
    ax        = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)

    for tabla, sub in puntos.groupby("tabla"):
        col = color_tabla.get(tabla, "#4A90D9")
        num = numero_tabla[tabla]
        ax.scatter(sub["C1"], sub["C2"], s=55, color=col, marker="o",
                   edgecolors="white", linewidths=0.6, alpha=0.85, zorder=5)
        cx_lbl, cy_lbl = sub["C1"].mean(), sub["C2"].mean()
        ax.scatter(cx_lbl, cy_lbl, s=90, color=col, marker="o",
                   edgecolors="white", linewidths=0.9, zorder=6)
        ax.text(cx_lbl, cy_lbl, str(num), fontsize=6.5, color="white",
                fontweight="bold", ha="center", va="center", zorder=7)

    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(0, 1)
    y0 = 0.99
    dy = 1.0 / max(n_tablas, 1)
    for tabla in tablas_sorted:
        col = color_tabla[tabla]
        num = numero_tabla[tabla]
        ax_legend.text(0.0, y0, f"{num}", fontsize=7.5, color=col,
                       fontweight="bold", ha="left", va="top",
                       transform=ax_legend.transAxes)
        ax_legend.text(0.09, y0, str(tabla), fontsize=7.5, color=col,
                       ha="left", va="top", transform=ax_legend.transAxes)
        y0 -= dy

    lmax = np.sqrt((G_verif**2).sum(axis=1)).max()
    fac  = (lim_b * 0.72) / lmax if lmax > 1e-10 else 1.0
    var_names = statis.variable_names
    for j, vname in enumerate(var_names):
        gx = G_verif[j, dim_x] * fac
        gy = G_verif[j, dim_y] * fac
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.6, mutation_scale=12), zorder=7)
        nombre = vname.replace("_score", "")
        r2 = np.hypot(gx, gy)
        nx2, ny2 = (gx/r2, gy/r2) if r2 > 1e-9 else (1.0, 0.0)
        off2 = 0.07 * lim_b
        ax.text(gx + nx2*off2, gy + ny2*off2, nombre,
                fontsize=9, color="black", fontweight="bold",
                ha="left" if nx2 >= 0 else "right",
                va="bottom" if ny2 >= 0 else "top", zorder=8,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xlim(-lim_b, lim_b)
    ax.set_ylim(-lim_b, lim_b)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dimensión {dim_x+1}  ({ve[dim_x]:.1f}% varianza)", fontsize=12)
    ax.set_ylabel(f"Dimensión {dim_y+1}  ({ve[dim_y]:.1f}% varianza)", fontsize=12)
    ax.set_title(
        "Verificación matemática del compromiso — Zc'·Zc = Σαk·Vk (Wc)\n"
        f"Diferencia máx. entre construcciones: {diff_max:.2e}  "
        "(Lavit et al., 1994)",
        fontsize=11, fontweight="semibold")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.09)
    return fig


def plot_trajectories_grid(statis: StatisDualResult,
                            dim_x: int = 0,
                            dim_y: int = 1,
                            n_cols: int = 4) -> plt.Figure:
    """
    Mapa de trayectorias longitudinales — un solo plano.
    Todas las tablas (países) con sus filas (años) conectadas cronológicamente.
    Igual al G7 del script validado de la tesis.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    ve       = statis.variance_explained
    d1_pct   = ve[dim_x]
    d2_pct   = ve[dim_y]
    coord_df = statis.coord_df
    dim1_col = f"Dim{dim_x+1}"
    dim2_col = f"Dim{dim_y+1}"

    tablas_sorted = sorted(statis.tablas)
    # Paleta viva — sin pasteles
    PALETA_VIVA = [
        '#E63946','#1D3557','#2A9D8F','#E9C46A','#6A4C93',
        '#F4A261','#264653','#457B9D','#C1121F','#2D6A4F',
        '#8338EC','#FB5607','#3A86FF','#FFBE0B','#06D6A0',
        '#118AB2','#073B4C','#EF233C','#4CC9F0','#F72585',
        '#7209B7','#560BAD','#480CA8','#3A0CA3','#3F37C9',
    ]
    color_tabla = {t: PALETA_VIVA[i % len(PALETA_VIVA)]
                   for i, t in enumerate(tablas_sorted)}

    all_x = list(coord_df[dim1_col])
    all_y = list(coord_df[dim2_col])
    lim_b = max(max(abs(v) for v in all_x), max(abs(v) for v in all_y)) * 1.25
    lim_b = max(lim_b, 0.1)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=96)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", lw=0.5, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)

    for tabla in statis.tablas:
        temp = (coord_df[coord_df["tabla"] == tabla]
                .sort_values("fila").reset_index(drop=True))
        col   = color_tabla[tabla]
        xg    = temp[dim1_col].values
        yg    = temp[dim2_col].values
        filas = temp["fila"].values

        # Línea de trayectoria (punteada, sin flecha superpuesta —
        # una punta de flecha es una forma rellena y sigue viéndose
        # "sólida" aunque el arrowprops tenga linestyle="--")
        ax.plot(xg, yg, color=col, lw=1.4, ls="--", alpha=0.85, zorder=4)

        # Puntos por año
        ax.scatter(xg, yg, s=30, color=col,
                   edgecolors="white", linewidths=0.4, zorder=6)

        # Etiqueta del país en el último punto
        ux, uy = float(xg[-1]), float(yg[-1])
        r = np.hypot(ux, uy)
        nx, ny = (ux/r, uy/r) if r > 1e-9 else (1.0, 0.0)
        off = 0.04 * lim_b
        ax.text(ux + nx*off, uy + ny*off, str(tabla),
                fontsize=7.5, color=col, fontweight="normal",
                ha="left" if nx >= 0 else "right",
                va="bottom" if ny >= 0 else "top",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="white", alpha=0.75, edgecolor="none"))

    # ── Vectores de variables (loadings) ─────────────────────────────────────
    nd    = statis.n_dims_retained
    U_ret = statis.eigenvectors[:, :nd]
    lam_r = np.sqrt(np.maximum(statis.eigenvalues[:nd], 0))
    L_mat = U_ret * lam_r[np.newaxis, :]           # (p × nd)
    ix = min(dim_x, nd-1)
    iy = min(dim_y, nd-1)
    L2 = np.column_stack([L_mat[:, ix], L_mat[:, iy]])  # (p × 2)
    lmax_L = np.sqrt((L2**2).sum(axis=1)).max()
    fac_L  = (lim_b * 0.60) / lmax_L if lmax_L > 1e-10 else 1.0

    for j, vname in enumerate(statis.variable_names):
        gx = float(L2[j, 0]) * fac_L
        gy = float(L2[j, 1]) * fac_L
        if abs(gx) < 1e-10 and abs(gy) < 1e-10:
            continue
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                   lw=1.1, mutation_scale=12), zorder=8)
        nombre = vname.replace("_score", "")
        r = np.hypot(gx, gy)
        nx2, ny2 = (gx/r, gy/r) if r > 1e-9 else (1.0, 0.0)
        ax.text(gx + nx2*0.06*lim_b, gy + ny2*0.06*lim_b, nombre,
                fontsize=9, color="black", fontweight="bold",
                ha="left" if nx2 >= 0 else "right",
                va="bottom" if ny2 >= 0 else "top",
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.85, edgecolor="none"))

    fig.text(0.01, 0.005,
             f"Vectores (U·√λ) multiplicados por c = {fac_L:.4g} para su visualización; "
             "c no altera ángulos ni proyecciones relativas. Los puntos están en su escala real.",
             fontsize=7, color="#555555", ha="left", va="bottom")
    ax.set_xlim(-lim_b, lim_b)
    ax.set_ylim(-lim_b, lim_b)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dimensión {dim_x+1}  ({d1_pct:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"Dimensión {dim_y+1}  ({d2_pct:.1f}% varianza)", fontsize=11)
    ax.set_title(f"Trayectorias longitudinales ({biplot_label(statis)}) — "
                 f"K={len(statis.tablas)} tablas\n"
                 f"Lavit et al. (1994) · {biplot_cita(statis)}",
                 fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig

# ─── Helper: bucket de color Ecuador / cercanos / resto LATAM ───────────────
# Idéntico al script validado (notebook de sensibilidad): distancia euclidiana
# de centroides (Dim1,Dim2) respecto a Ecuador, top-5 más cercanos.

def _top_cercanos_a_entidad(centroides, entidad: str = "Ecuador",
                             n: int = 5, dim_cols=("Dim1", "Dim2")) -> list:
    """Devuelve la lista de las n tablas (países) más cercanas a `entidad`
    por distancia euclidiana de centroides. Si `entidad` no está presente,
    devuelve lista vacía (no rompe el gráfico, solo no resalta naranja)."""
    if centroides is None or entidad not in centroides.index:
        return []
    cols = [c for c in dim_cols if c in centroides.columns]
    if not cols:
        return []
    c_ref = centroides.loc[entidad, cols].values.astype(float)
    dist  = np.sqrt(((centroides[cols].values.astype(float) - c_ref) ** 2).sum(axis=1))
    dist_s = pd.Series(dist, index=centroides.index).drop(entidad, errors="ignore")
    return dist_s.sort_values().head(n).index.tolist()


def _color_por_bucket(pais: str, top_cercanos: list, entidad: str = "Ecuador") -> str:
    """Rojo Ecuador · Naranja cercanos · Azul resto LATAM (paleta del script)."""
    if str(pais).upper() == entidad.upper():
        return COLOR_ECUADOR
    if pais in top_cercanos:
        return COLOR_CERCANOS
    return COLOR_LATAM


# ─── G03 · Pesos αk ─────────────────────────────────────────────────────────

def plot_alpha_k(statis: StatisDualResult) -> plt.Figure:
    """
    Barras horizontales de αk ordenadas de mayor a menor.
    Color por TERCIL de magnitud (alto/medio/bajo) — no por identidad de país
    ni cercanía a Ecuador. Ecuador se resalta siempre en rojo, sin importar
    en qué tercil caiga su valor.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    data_sorted = sorted(statis.alpha_k_rank, key=lambda x: x[1], reverse=True)
    labels = [str(b) for b, _, _ in data_sorted]
    values = [a for _, a, _ in data_sorted]
    n = len(values)
    tercio = max(1, n // 3)

    def _tier_color(i):
        if i < tercio:
            return COLOR_TIER_ALTO
        elif i < 2 * tercio:
            return COLOR_TIER_MEDIO
        return COLOR_TIER_BAJO

    colors = [
        COLOR_ECUADOR if lbl.upper() == "ECUADOR" else _tier_color(i)
        for i, lbl in enumerate(labels)
    ]

    # Ancho DINÁMICO según la longitud real de las etiquetas — no un valor
    # fijo pensado solo para nombres de país. Con nombres largos (ej.
    # universidades) el ancho crece automáticamente para que el margen de
    # etiquetas no se coma el espacio del título y las barras, sin importar
    # qué base de datos se cargue.
    max_label_len = max((len(l) for l in labels), default=10)
    fig_w = max(FIG_WIDTH_SINGLE, 3.0 + max_label_len * 0.10)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, len(labels) * 0.35)),
                            dpi=96)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)

    bars = ax.barh(range(len(labels)), values, color=colors,
                   height=0.65, edgecolor="white", linewidth=0.5)

    # Etiquetas de valor
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, color=COLOR_TEXT)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Etiqueta de Ecuador en rojo y negrita (si existe esa fila; en modo
    # universidades simplemente no habrá coincidencia y no pasa nada)
    for i, lbl in enumerate(labels):
        if lbl.upper() == "ECUADOR":
            ax.get_yticklabels()[i].set_color(COLOR_ECUADOR)
            ax.get_yticklabels()[i].set_fontweight("bold")

    mean_line = np.mean(values)
    ax.axvline(mean_line, color="#888888", linestyle="--",
               linewidth=1.0, label=f"Media = {mean_line:.4f}")

    # Leyenda: SOLO la línea de la media (sin parches de bucket) — ubicada
    # arriba-derecha, zona de barras cortas, para no solaparse con los datos.
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    _style_axes(ax,
                title="Pesos αk — Similitud estructural al consenso\n"
                      "(RV coefficient, Lavit et al. 1994)",
                xlabel="αk",
                ylabel="Tabla")
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ─── G03b · COS² Consenso — "Typological value" (ade4::statis) ──────────────

def plot_cos2_consenso(statis: StatisDualResult) -> plt.Figure:
    """
    Barras horizontales de cos2_consenso ordenadas de mayor a menor.
    Responde: "¿esta tabla se parece a la estructura COMPLETA del consenso?"
    (a diferencia de COS² por eje, que responde "¿confío en su posición en
    este plano 2D?"). Mismo estilo visual que plot_alpha_k, para comparar
    ambos rankings a simple vista.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    if getattr(statis, "cos2_consenso", None) is None:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, 3), dpi=96)
        ax.text(0.5, 0.5, "cos2_consenso no disponible para este análisis",
                ha="center", va="center", fontsize=10, color=COLOR_TEXT)
        ax.axis("off")
        return fig

    df_sorted = statis.cos2_consenso.sort_values("cos2_consenso", ascending=False)
    labels = [str(t) for t in df_sorted.index]
    values = list(df_sorted["cos2_consenso"].values)
    n = len(values)
    tercio = max(1, n // 3)

    def _tier_color(i):
        if i < tercio:
            return COLOR_TIER_ALTO
        elif i < 2 * tercio:
            return COLOR_TIER_MEDIO
        return COLOR_TIER_BAJO

    colors = [
        COLOR_ECUADOR if lbl.upper() == "ECUADOR" else _tier_color(i)
        for i, lbl in enumerate(labels)
    ]

    max_label_len = max((len(l) for l in labels), default=10)
    fig_w = max(FIG_WIDTH_SINGLE, 3.0 + max_label_len * 0.10)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, len(labels) * 0.35)), dpi=96)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)

    bars = ax.barh(range(len(labels)), values, color=colors,
                   height=0.65, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, color=COLOR_TEXT)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i, lbl in enumerate(labels):
        if lbl.upper() == "ECUADOR":
            ax.get_yticklabels()[i].set_color(COLOR_ECUADOR)
            ax.get_yticklabels()[i].set_fontweight("bold")

    mean_line = np.mean(values)
    ax.axvline(mean_line, color="#888888", linestyle="--",
               linewidth=1.0, label=f"Media = {mean_line:.4f}")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    _style_axes(ax,
                title="COS² Consenso — ¿Se parece esta tabla al consenso (rango truncado)?\n"
                      "trace(Wk_norm·S_norm) — análogo al 'Typological value' de ade4::statis",
                xlabel="cos2_consenso",
                ylabel="Tabla")
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ─── G04 · COS² heatmap ──────────────────────────────────────────────────────

def plot_cos2(statis: StatisDualResult) -> plt.Figure:
    """
    Heatmap COS² por TABLA (país) × dimensión.
    Igual que el script validado: media del COS² de cada año por país.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    # Usar cos2_por_tabla si existe, si no usar cos2_rows
    if statis.cos2_por_tabla is not None:
        import pandas as pd
        data      = statis.cos2_por_tabla.values
        row_names = list(statis.cos2_por_tabla.index)
        dim_labels = list(statis.cos2_por_tabla.columns)
    else:
        data      = statis.cos2_rows
        row_names = statis.row_names
        dim_labels = [f"Dim{d+1}" for d in range(statis.n_dims_retained)]

    # Ancho generoso: antes max(3, dim*1.2) daba solo ~3" — demasiado angosto
    # para el título de 2 líneas y los nombres de país largos en el eje Y.
    # Ahora suma también la longitud real de las etiquetas de fila (país
    # o universidad), para funcionar igual con cualquier base.
    max_row_len = max((len(str(r)) for r in row_names), default=10)
    fig_w = max(FIG_WIDTH_SINGLE,
                2.0 + max_row_len * 0.09 + len(dim_labels) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w,
                                     max(4, len(row_names) * 0.38)),
                            dpi=96)
    fig.patch.set_facecolor(COLOR_BG)

    sns.heatmap(data, ax=ax,
                xticklabels=dim_labels,
                yticklabels=row_names,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "COS²"})

    ax.set_title("COS² — Calidad de representación por tabla\n"
                 "(media por tabla de cos² = (z'u)²/‖z‖², coordenadas principales)",
                 fontsize=10, fontweight="bold", color=COLOR_PRIMARY, pad=10)
    ax.set_xlabel("Dimensión", fontsize=10)
    ax.set_ylabel("Tabla", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ─── G05 · Heatmap RV ───────────────────────────────────────────────────────

def plot_rv_heatmap(statis: StatisDualResult) -> plt.Figure:
    """Heatmap de la matriz RV (K×K)."""
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    RV = statis.RV_matrix
    labels = [str(b) for b in statis.tablas]

    # Tamaño DINÁMICO: cuadrado 7x7 fijo era demasiado chico con muchas
    # tablas y/o etiquetas largas (universidades). Escala con la longitud
    # de etiqueta y con K, para funcionar igual con cualquier base.
    # FIX: con K grande (20-24+ tablas) los números quedaban muy apretados
    # y difíciles de distinguir — se aumentó el factor de escala por tabla
    # (0.18 → 0.32) para dar más espacio a cada celda.
    max_label_len = max((len(l) for l in labels), default=10)
    fig_size = max(FIG_WIDTH_SINGLE,
                   2.0 + max_label_len * 0.09 + len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=96)
    fig.patch.set_facecolor(COLOR_BG)

    mask = np.eye(len(labels), dtype=bool)   # ocultar diagonal
    # FIX: tamaño de fuente un poco mayor (7 → 8.5) — con el fig_size ya
    # ampliado arriba, ahora hay espacio de sobra sin que se amontonen.
    sns.heatmap(RV, ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".2f", annot_kws={"size": 8.5},
                cmap="Blues", vmin=0, vmax=1,
                mask=mask,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "RV"})

    # FIX: contraste de texto por celda — antes todos los números salían
    # del mismo color (por defecto de seaborn), ilegibles tanto en celdas
    # FIX (a pedido del usuario): el contraste automático (blanco/negro
    # según qué tan oscura fuera la celda) resultó en números en blanco
    # sobre celdas que en la práctica no eran lo bastante oscuras para
    # que el blanco se leyera bien — texto invisible. Se simplifica a
    # texto NEGRO fijo para todas las celdas, sin excepción.
    n = RV.shape[0]
    for i, txt in enumerate(ax.texts):
        row, col = divmod(i, n)
        if mask[row, col]:
            continue
        txt.set_color("#111111")
        txt.set_fontweight("medium")

    ax.set_title("Matriz RV — Interestructura STATIS Dual\n"
                 "(coeficiente RV entre tablas)",
                 fontsize=10, fontweight="bold", color=COLOR_PRIMARY, pad=10)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ─── G06 · Scree plot ───────────────────────────────────────────────────────

def plot_scree(statis: StatisDualResult) -> plt.Figure:
    """Scree plot: varianza explicada y acumulada."""
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    ve  = statis.variance_explained
    vc  = statis.variance_cumulative
    dims = [f"Dim{i+1}" for i in range(len(ve))]

    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_SINGLE, 4), dpi=96)
    ax2 = ax1.twinx()
    fig.patch.set_facecolor(COLOR_BG)
    ax1.set_facecolor(COLOR_BG)

    bars = ax1.bar(dims, ve, color=COLOR_ACCENT, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label="Varianza explicada (%)")
    # Etiqueta de % en cada barra
    for bar, v in zip(bars, ve):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color=COLOR_PRIMARY, fontweight="bold")
    ax2.plot(dims, vc, color=COLOR_SECONDARY, marker="o",
             linewidth=2, markersize=6, label="Varianza acumulada (%)")

    ax1.set_ylabel("Varianza explicada (%)", color=COLOR_ACCENT, fontsize=10)
    ax2.set_ylabel("Varianza acumulada (%)", color=COLOR_SECONDARY, fontsize=10)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(0, 105)   # siempre desde 0 hasta 100
    ax1.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)
    ax1.set_xlabel("Dimensión", fontsize=10)

    # Línea 80% referencia
    ax2.axhline(80, color="#A0AEC0", linestyle="--",
                linewidth=0.9, label="80% referencia")

    # Detectar codo automático (mayor diferencia entre dims consecutivas)
    diffs = np.diff(ve)
    codo  = int(np.argmin(diffs)) + 1  # dim donde la caída es más brusca
    ax2.axvline(dims[codo], color=COLOR_ACCENT, linestyle=":",
                linewidth=1.5, alpha=0.8, label=f"Codo sugerido: Dim{codo+1}")

    ax1.set_title(f"Scree plot — Varianza explicada por dimensión\n"
                  f"Compromiso STATIS Dual  ·  Codo sugerido: Dim{codo+1}  ·  "
                  f"Dim1+Dim2 acumula {vc[1]:.1f}%",
                  fontsize=10, fontweight="bold", color=COLOR_PRIMARY, pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax1.spines[["top"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)
    return fig


# ─── G07 · Intraestructura heatmap ──────────────────────────────────────────

def plot_intrastructure(intra: IntrastructureResult) -> plt.Figure:
    """Heatmap de correlaciones Pearson variable × dimensión."""
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    data   = intra.correlations    # (p × n_dims)
    rows   = intra.variable_names
    cols   = intra.dim_labels

    # Mismo fix que plot_cos2: max(4, cols*1.2) quedaba angosto para el
    # título de 2 líneas. Ancho mínimo = estándar de columna simple (7.0").
    fig_w = max(FIG_WIDTH_SINGLE, 3.5 + len(cols) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w,
                                     max(4, len(rows) * 0.5)),
                            dpi=96)
    fig.patch.set_facecolor(COLOR_BG)

    # Anotaciones con signo
    annot = np.vectorize(lambda x: f"{x:+.2f}")(data)

    sns.heatmap(data, ax=ax,
                xticklabels=cols, yticklabels=rows,
                annot=annot, fmt="", annot_kws={"size": 8.5},
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.8, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson r"})

    ax.set_title("Intraestructura — r(Xvar, Fdim)\n"
                 "Correlaciones Pearson indicadores × dimensiones STATIS Dual",
                 fontsize=10, fontweight="bold", color=COLOR_PRIMARY, pad=10)
    ax.set_xlabel("Dimensión", fontsize=10)
    ax.set_ylabel("Variable", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ─── Exportación ─────────────────────────────────────────────────────────────

def export_figure(fig: plt.Figure, filepath: str):
    """
    Exporta la figura en PNG (600 dpi) y TIFF (600 dpi).
    filepath: ruta base sin extensión, ej. "/ruta/G01_HJBiplot"

    NOTA IMPORTANTE (fix "figuras achatadas"):
    Antes se usaba bbox_inches="tight", que recalcula el bounding box
    según el contenido renderizado y DESCARTA el figsize/aspect ratio
    que cada función plot_* definió (p. ej. FIG_WIDTH_SINGLE = 7.0" para
    el estándar de columna simple de revistas Q1). Eso hacía que cada
    figura se guardara con una proporción ancho/alto distinta e
    impredecible, y al insertarla en una plantilla con marco de imagen
    fijo se veía "achatada"/deformada.

    Ahora se guarda SIN recorte automático: el PNG/TIFF respeta
    exactamente las pulgadas de figsize × EXPORT_DPI (ej. figsize=(7,5)
    a 600 dpi → 4200×3000 px), que es lo que exige el estándar de
    figuras de publicación. plt.tight_layout(), ya llamado dentro de
    cada plot_*, se encarga de que títulos/leyendas queden dentro del
    área del figsize sin necesidad de recortar el lienzo.
    """
    path = Path(filepath)
    fig.savefig(str(path) + ".png",  dpi=EXPORT_DPI,
                facecolor=fig.get_facecolor())
    fig.savefig(str(path) + ".tiff", dpi=EXPORT_DPI,
                facecolor=fig.get_facecolor())


# ─── G08 · Interestructura — posición de tablas en espacio RV ───────────────

def plot_interstructure_rv(statis: StatisDualResult) -> plt.Figure:
    """
    Gráfico A del Panel 4: Posición de tablas (países) en el espacio RV.
    Proyecta los eigenvectores de RV como círculo unitario con vectores por tabla.
    Igual al subplot A) del script validado.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    RV   = statis.RV_matrix
    tablas = statis.tablas
    alpha  = statis.alpha_k
    ve     = statis.variance_explained

    # Eigendescomposición de RV para obtener coordenadas en el espacio interestructura
    # Eigendescomposición con el algoritmo propio (iteración QR, m3_blocks),
    # el mismo que calcula los pesos αk; ordenada descendente.
    vals_rv, vecs_rv = manual_eigen_symmetric(RV)

    # Coordenadas de cada tabla en Dim1/Dim2 del espacio RV
    # = primeras 2 columnas de eigenvectores escalados por sqrt(eigenvalor)
    lam1 = np.sqrt(max(vals_rv[0], 0))
    lam2 = np.sqrt(max(vals_rv[1], 0)) if len(vals_rv) > 1 else 0
    coords_x = vecs_rv[:, 0] * lam1
    coords_y = vecs_rv[:, 1] * lam2 if lam2 > 0 else np.zeros(len(tablas))

    # Asegurar orientación positiva (mayoría de αk positivos)
    if coords_x.dot(alpha) < 0:
        coords_x = -coords_x
    if coords_y.mean() < 0:
        coords_y = -coords_y

    tablas_sorted = sorted(statis.tablas)
    color_tabla = {t: PALETA_PAISES[i % len(PALETA_PAISES)]
                   for i, t in enumerate(tablas_sorted)}

    lim = max(np.abs(coords_x).max(), np.abs(coords_y).max()) * 1.25
    lim = max(lim, 0.1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=96)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Círculo unitario de referencia
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta)*lim*0.75, np.sin(theta)*lim*0.75,
            color="#CCCCCC", lw=0.8, ls="--", alpha=0.6)

    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)
    ax.grid(True, color="#E5E5E5", lw=0.5, zorder=0)

    # Vectores desde el origen hasta cada tabla
    for i, tabla in enumerate(tablas):
        cx, cy = float(coords_x[i]), float(coords_y[i])
        col = color_tabla.get(tabla, "#4A90D9")
        ax.annotate("",
                    xy=(cx, cy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.4, mutation_scale=9), zorder=5)
        r = np.hypot(cx, cy)
        nx, ny = (cx/r, cy/r) if r > 1e-9 else (1.0, 0.0)
        off = 0.04 * lim
        ax.text(cx + nx*off, cy + ny*off, str(tabla),
                fontsize=7.5, color=col,
                ha="left" if nx >= 0 else "right",
                va="bottom" if ny >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=6)

    pct1 = vals_rv[0] / vals_rv.sum() * 100 if vals_rv.sum() > 0 else 0
    pct2 = vals_rv[1] / vals_rv.sum() * 100 if len(vals_rv) > 1 else 0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim 1 ({pct1:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Dim 2 ({pct2:.1f}%)", fontsize=11)
    ax.set_title("A) Interestructura\nPosición de tablas en el espacio RV",
                 fontsize=11, fontweight="bold", color=COLOR_PRIMARY)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


# ─── G09 · Compromiso — Loadings (vectores) ──────────────────────────────────

def plot_loadings(statis: StatisDualResult,
                  dim_x: int = 0,
                  dim_y: int = 1) -> plt.Figure:
    """
    Gráfico C del Panel 4: Loadings Lj = Uj * sqrt(λj) en círculo de correlación.
    Solo vectores de variables, sin puntos de países.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    ve       = statis.variance_explained
    G        = statis.F_vars
    var_names= statis.variable_names

    # Loadings = U * sqrt(λ) usando atributos ya calculados en statis
    nd   = statis.n_dims_retained
    U_l  = statis.eigenvectors[:, :nd]          # (p × nd)
    lam_l= np.sqrt(np.maximum(statis.eigenvalues[:nd], 0))  # (nd,)
    L_all= U_l * lam_l[np.newaxis, :]          # (p × nd)
    # Tomar las 2 dims de visualización
    ix = min(dim_x, nd-1); iy = min(dim_y, nd-1)
    G_2d = np.column_stack([L_all[:, ix], L_all[:, iy]])  # (p × 2)
    lmax = np.sqrt((G_2d**2).sum(axis=1)).max()
    fac  = 1.0 / lmax if lmax > 1e-10 else 1.0
    G_norm = G_2d * fac

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Círculo unitario
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#CCCCCC", lw=0.8, ls="--", alpha=0.6)

    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.7, alpha=0.6)
    ax.grid(True, color="#E5E5E5", lw=0.5, zorder=0)

    for j, vname in enumerate(var_names):
        gx = float(G_norm[j, 0])   # dim_x normalizado
        gy = float(G_norm[j, 1])   # dim_y normalizado
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                   lw=1.8, mutation_scale=12), zorder=5)
        nombre = vname.replace("_score", "")
        r = np.hypot(gx, gy)
        nx, ny = (gx/r, gy/r) if r > 1e-9 else (1.0, 0.0)
        off = 0.07
        ax.text(gx + nx*off, gy + ny*off, nombre,
                fontsize=10, color="black", fontweight="bold",
                ha="left" if nx >= 0 else "right",
                va="bottom" if ny >= 0 else "top",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", alpha=0.88, edgecolor="none"))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim {dim_x+1} ({ve[dim_x]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Dim {dim_y+1} ({ve[dim_y]:.1f}%)", fontsize=11)
    ax.set_title(f"C) Compromiso — Loadings Lj = Uj·√λj\n"
                 f"Representación de variables compromiso (Lavit et al. 1994)",
                 fontsize=11, fontweight="bold", color=COLOR_PRIMARY)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


# ─── G10 · Trayectoria individual por tabla ──────────────────────────────────

def plot_single_trajectory(statis: StatisDualResult,
                            tabla: str,
                            dim_x: int = 0,
                            dim_y: int = 1) -> plt.Figure:
    """
    Trayectoria temporal de una sola tabla con etiquetas de año y zoom adaptado.
    Igual al G8 del script: una figura por país.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    if tabla not in statis.F_k:
        raise ValueError(f"Tabla '{tabla}' no encontrada.")

    ve      = statis.variance_explained
    G       = statis.F_vars
    Fk      = statis.F_k[tabla]
    filas_t = statis.filas_por_tabla[tabla]
    var_names = statis.variable_names

    xs = Fk[:, dim_x]
    ys = Fk[:, dim_y]

    # ── Escala del gráfico ─────────────────────────────────────────────────
    # Los límites se fijan a partir de los puntos de la trayectoria, no con
    # mínimos fijos: con el HJ-Biplot del compromiso las coordenadas de una
    # tabla pueden ser del orden de 0,1 y un mínimo de 1,0 las dejaría
    # apiñadas en el origen. Los vectores se reescalan después con UN factor
    # común c (fac_L), que conserva ángulos, orden de proyecciones y
    # distancias relativas (Gower, Lubbe & le Roux, 2011).
    r_pts = float(np.max(np.hypot(xs, ys))) if len(xs) else 0.0
    lim_pts = max(float(np.max(np.abs(xs))), float(np.max(np.abs(ys))), 1e-12)
    # Los vectores se dibujan con longitud máxima = radio máximo de los puntos
    # (o el 80 % del límite si la trayectoria es casi un punto).
    lim = lim_pts * 1.45

    COLORES_FILAS = ['#1D3557','#457B9D','#2A9D8F','#E9C46A',
                     '#F4A261','#E76F51','#6A4C93']

    # Ancho dinámico: el título incluye el nombre de la tabla (país o
    # universidad); con nombres largos, 9" fijo no alcanza y se corta.
    fig_w = max(9, 3.0 + len(str(tabla)) * 0.13)
    fig, ax = plt.subplots(figsize=(fig_w, 8), dpi=96)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", lw=0.5, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.7)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.7)

    # Línea de trayectoria (punteada, sin flecha superpuesta — misma
    # razón que en el gráfico grid: la punta de flecha es una forma
    # rellena y rompe visualmente el patrón punteado)
    ax.plot(xs, ys, color="#1D3557", lw=2.0, ls="--", alpha=0.7, zorder=3)

    # Puntos y etiquetas por año
    # Desplazamiento radial desde el centroide — evita solapamiento cuando puntos son cercanos
    n_pts = len(filas_t)
    cx_mean = float(np.mean(xs)); cy_mean = float(np.mean(ys))
    for t, fila in enumerate(filas_t):
        col = COLORES_FILAS[t % len(COLORES_FILAS)]
        ax.scatter(float(xs[t]), float(ys[t]), s=80, color=col,
                   edgecolors="white", linewidths=0.9, zorder=5)
        # Dirección desde centroide al punto → etiqueta hacia afuera
        dpx = float(xs[t]) - cx_mean; dpy = float(ys[t]) - cy_mean
        r = np.hypot(dpx, dpy)
        if r > 1e-9:
            nx_l = dpx/r; ny_l = dpy/r
        else:
            # Si el punto está en el centroide, usar ángulo según índice
            angle = 2 * np.pi * t / max(n_pts, 1)
            nx_l = np.cos(angle); ny_l = np.sin(angle)
        off = 0.12 * lim             # desplazamiento proporcional a la escala
        lx = float(xs[t]) + nx_l * off
        ly = float(ys[t]) + ny_l * off
        ax.annotate(str(fila),
                    xy=(float(xs[t]), float(ys[t])),
                    xytext=(lx, ly),
                    fontsize=9, color=col, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=col,
                                    lw=0.8, alpha=0.5),
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", alpha=0.88,
                              edgecolor=col, linewidth=0.8))

    # Vectores de variables — usar loadings reales U*sqrt(λ), escalados al 65% del espacio
    n_d   = statis.n_dims_retained
    U_ret = statis.eigenvectors[:, :n_d]
    lam_r = np.sqrt(np.maximum(statis.eigenvalues[:n_d], 0))
    L_mat = U_ret * lam_r[np.newaxis, :]     # loadings (p × n_dims)
    # Seleccionar las dos dimensiones de visualización
    lx_col = dim_x if dim_x < n_d else 0
    ly_col = dim_y if dim_y < n_d else min(1, n_d-1)
    L2 = L_mat[:, [lx_col, ly_col]]          # (p × 2)
    lmax_L = np.sqrt((L2**2).sum(axis=1)).max()
    largo_obj = r_pts if r_pts > 0.25 * lim else 0.8 * lim
    fac_L  = largo_obj / lmax_L if lmax_L > 1e-10 else 1.0

    for j, vname in enumerate(var_names):
        gx = float(L2[j, 0]) * fac_L
        gy = float(L2[j, 1]) * fac_L
        if abs(gx) < 1e-10 and abs(gy) < 1e-10:
            continue
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                   lw=1.1, mutation_scale=11), zorder=6)
        nombre = vname.replace("_score", "")
        r = np.hypot(gx, gy)
        nx, ny = (gx/r, gy/r) if r > 1e-9 else (1.0, 0.0)
        ax.text(gx + nx*0.06*lim, gy + ny*0.06*lim, nombre,
                fontsize=9, color="black", fontweight="bold",
                ha="left" if nx >= 0 else "right",
                va="bottom" if ny >= 0 else "top",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim {dim_x+1} ({ve[dim_x]:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"Dim {dim_y+1} ({ve[dim_y]:.1f}% varianza)", fontsize=11)
    ax.set_title(f"Trayectoria — {tabla} ({biplot_label(statis)})\nDim{dim_x+1} × Dim{dim_y+1}",
                 fontsize=12, fontweight="bold", color=COLOR_PRIMARY)
    # Nota del factor de escala de los vectores (debe declararse en la figura)
    fig.text(0.01, 0.01,
             f"Vectores (U·√λ) multiplicados por c = {fac_L:.4g} para su visualización; "
             "c no altera ángulos ni proyecciones relativas.",
             fontsize=7.5, color="#555555", ha="left", va="bottom")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.11)
    return fig


# ─── Exportación Excel completa ──────────────────────────────────────────────

def export_results_excel(statis, intra, filepath: str):
    """
    Exporta todos los resultados a Excel con el mismo formato que
    SEN_Resultados_3Paneles.xlsx — una hoja por resultado.
    """
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment,
                                  Border, Side, numbers)
    from openpyxl.utils import get_column_letter
    import pandas as pd

    wb = Workbook()
    wb.remove(wb.active)

    # Estilos
    hdr_fill = PatternFill("solid", start_color="0D2545")
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    sub_fill = PatternFill("solid", start_color="1A7A8A")
    sub_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    cell_font= Font(name="Arial", size=10)
    bold_font= Font(name="Arial", bold=True, size=10)
    thin     = Side(style="thin", color="CBD5E0")
    border   = Border(left=thin, right=thin, top=thin, bottom=thin)
    center   = Alignment(horizontal="center", vertical="center")

    def style_header(cell, sub=False):
        cell.fill  = sub_fill if sub else hdr_fill
        cell.font  = sub_font if sub else hdr_font
        cell.alignment = center
        cell.border= border

    def style_data(cell, bold=False):
        cell.font  = bold_font if bold else cell_font
        cell.border= border
        cell.alignment = Alignment(horizontal="center", vertical="center")

    def write_df(ws, df, title, start_row=1, index=True):
        """Escribe un DataFrame con header estilizado."""
        ws.cell(start_row, 1, title).font = Font(name="Arial", bold=True, size=11)
        ws.cell(start_row, 1).fill = PatternFill("solid", start_color="EDF2F7")
        r = start_row + 2

        # Cabecera
        cols = ([df.index.name or ""] if index else []) + list(df.columns)
        for c, col in enumerate(cols, 1):
            cell = ws.cell(r, c, str(col))
            style_header(cell)
            ws.column_dimensions[get_column_letter(c)].width = max(14, len(str(col))+4)
        r += 1

        # Datos
        for idx_val, row in df.iterrows():
            c = 1
            if index:
                cell = ws.cell(r, c, str(idx_val))
                style_data(cell, bold=True)
                c += 1
            for val in row:
                cell = ws.cell(r, c, round(float(val), 6) if isinstance(val, float) else val)
                style_data(cell)
                c += 1
            r += 1
        return r

    # ── Hoja 1: Resumen global ───────────────────────────────────────────────
    ws = wb.create_sheet("Resumen_Global")
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 20

    info = [
        ("Análisis", "STATIS Dual"),
        ("K tablas", str(len(statis.tablas))),
        ("Tablas", ", ".join(str(t) for t in sorted(statis.tablas))),
        ("Variables (p)", str(len(statis.variable_names))),
        ("Variables", ", ".join(statis.variable_names)),
        ("Filas únicas", str(len(statis.row_names))),
        ("Dims retenidas", str(statis.n_dims_retained)),
        ("Tipo de biplot", f"{biplot_label(statis)} — {biplot_cita(statis)}"),
        ("Versión del cálculo", CALC_VERSION),
        ("COS²", "Clásico: (z_i'u_j)²/‖z_i‖², coordenadas principales (Lebart et al. 1995)"),
        ("Var. explicada Dim1 (%)", f"{statis.variance_explained[0]:.4f}"),
        ("Var. explicada Dim2 (%)", f"{statis.variance_explained[1]:.4f}" if len(statis.variance_explained)>1 else ""),
        ("Var. acumulada Dim1+Dim2 (%)", f"{statis.variance_cumulative[min(1,len(statis.variance_cumulative)-1)]:.4f}"),
        ("Referencia", "Lavit et al. (1994) Comput. Stat. Data Anal. 18(1), 97-119"),
    ]
    ws.cell(1,1,"Resultados STATIS Dual").font = Font(name="Arial", bold=True, size=13)
    ws.cell(1,1).fill = PatternFill("solid", start_color="0D2545")
    ws.cell(1,1).font = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    for r, (k, v) in enumerate(info, 3):
        ws.cell(r, 1, k).font = bold_font
        ws.cell(r, 2, v).font = cell_font

    # ── Hoja 2: αk ──────────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Alphak")
    ak_df = pd.DataFrame(
        [(t, round(a,6), rk) for t,a,rk in statis.alpha_k_rank],
        columns=["Tabla", "αk", "Rango"]
    ).set_index("Tabla")
    write_df(ws2, ak_df, "Pesos αk — Similitud estructural al consenso (Lavit et al. 1994)")

    # ── Hoja 3: COS² ─────────────────────────────────────────────────────────
    ws3 = wb.create_sheet("COS2")
    if statis.cos2_por_tabla is not None:
        write_df(ws3, statis.cos2_por_tabla,
                 "COS² por tabla — media de cos²(i,j) = (z_i'u_j)²/‖z_i‖² sobre las filas de cada tabla, "
                 "en coordenadas principales; igual con HJ y GH "
                 "(calidad de representación clásica: Benzécri 1973; Lebart et al. 1995)")

    # ── Hoja 3b: COS²_Consenso — "valor tipológico" estilo ade4::statis ──────
    # Métrica DISTINTA de COS2: compara la tabla COMPLETA (Wk/Vk) contra el
    # consenso TRUNCADO a n_dims (mismo rango que Loadings/Biplot/Scree) —
    # no una proyección sobre un eje específico, ni el consenso de rango
    # completo. Replica statis.R (C.li construido con solo 'nf' ejes, luego
    # apply(sep*w, 2, sum)) — ver nota completa en m4_interstructure.py F10b.
    # Útil para responder "¿esta tabla se parece al patrón global visible en
    # el biplot que estoy mostrando?", propio de la etapa de INTERESTRUCTURA —
    # a diferencia de COS2 (arriba), que responde "¿confío en la posición de
    # esta tabla EN UN EJE ESPECÍFICO del biplot 2D?".
    ws3b = wb.create_sheet("COS2_Consenso")
    if getattr(statis, "cos2_consenso", None) is not None:
        write_df(ws3b, statis.cos2_consenso,
                 "COS² respecto al consenso TRUNCADO a n_dims_retenidas — "
                 "trace(Wk_norm·S_trunc_norm) (análogo al 'Typological value' "
                 "de ade4::statis; usa el mismo número de ejes que Loadings/"
                 "Biplot/Scree — ver 'Dims retenidas' en Resumen_Global; "
                 "no aparece textualmente en Lavit et al. 1994 con este nombre, "
                 "pero sí la lógica de aproximación de rango h, sección "
                 "'Graphical representation of the compromise', p.106)")

    # ── Hoja 4: Coordenadas (scores parciales) ───────────────────────────────
    ws4 = wb.create_sheet("Coords")
    if statis.coord_df is not None:
        dim_cols = [c for c in statis.coord_df.columns if c.startswith("Dim")]
        coords_out = statis.coord_df[["tabla","fila"] + dim_cols].copy()
        coords_out.columns = ["Tabla","Fila"] + dim_cols
        write_df(ws4, coords_out.set_index("Tabla"),
                 "Coordenadas de las filas — " + (getattr(statis, "coord_formula", "")
                 or "Fk = Xk @ U · (1/√λ) (biplot GH, Gabriel 1971)"))

    # ── Hoja 5: Varianza explicada ───────────────────────────────────────────
    ws5 = wb.create_sheet("Varianza")
    var_df = pd.DataFrame({
        "Dim": [f"Dim{i+1}" for i in range(len(statis.eigenvalues))],
        "Lambda": [round(v,6) for v in statis.eigenvalues],
        "Var. Exp. (%)": [round(v,4) for v in statis.variance_explained],
        "Var. Acum. (%)": [round(v,4) for v in statis.variance_cumulative],
    }).set_index("Dim")
    write_df(ws5, var_df, "Varianza explicada — Eigenvalores del compromiso S")

    # ── Hoja 6: Intraestructura ──────────────────────────────────────────────
    ws6 = wb.create_sheet("Intraestructura")
    if intra is not None and intra.df_correlations is not None:
        write_df(ws6, intra.df_correlations,
                 "Intraestructura — r(Xvar, Fdim) de Pearson entre cada indicador escalado y las coordenadas de las observaciones (círculo de correlaciones del ACP; Lebart et al. 1995). Invariante al tipo de biplot.")

    # ── Hoja 7: Matriz RV ────────────────────────────────────────────────────
    ws7 = wb.create_sheet("Matriz_RV")
    if statis.RV_matrix is not None:
        rv_df = pd.DataFrame(
            np.round(statis.RV_matrix, 4),
            index=[str(t) for t in statis.tablas],
            columns=[str(t) for t in statis.tablas]
        )
        write_df(ws7, rv_df, "Matriz RV — Coeficiente RV entre tablas (Robert & Escoufier 1976)")

    # ── Hoja 8: Loadings ─────────────────────────────────────────────────────
    ws8 = wb.create_sheet("Loadings")
    n_dims = statis.n_dims_retained
    dim_names = [f"Dim{i+1}" for i in range(n_dims)]
    load_data = {}
    for di, dn in enumerate(dim_names):
        load_data[dn] = [round(float(statis.F_vars[j, di] /
                               (np.max(np.abs(statis.F_vars)) + 1e-10) *
                               np.sqrt(statis.eigenvalues[di])), 6)
                         for j in range(len(statis.variable_names))]
    # Usar directamente los eigenvalores para loadings = U * sqrt(lambda)
    U_ret = statis.eigenvectors[:, :n_dims]
    lam_ret = statis.eigenvalues[:n_dims]
    load_mat = U_ret * np.sqrt(lam_ret)
    load_df = pd.DataFrame(
        np.round(load_mat, 6),
        index=statis.variable_names,
        columns=dim_names
    )
    write_df(ws8, load_df,
             "Loadings del compromiso — Lj = Uj·√λj (Lavit et al. 1994) — "
             "NOTA: representa solo variables; no es HJ-Biplot (requiere individuos+variables juntos)")

    # ── Hoja 9: Biplot_Coords — pares ordenados EXACTOS del biplot validado ──
    # IMPORTANTE: el biplot valida grafica puntos INDIVIDUALES (tabla-fila,
    # ej. país-año), no un centroide por tabla — el promedio de una tabla es
    # SIEMPRE (0,0) por construcción (los datos se centran por tabla antes
    # de proyectar sobre el compromiso; ver nota en plot_hjbiplot). Por eso
    # esta hoja tiene el MISMO contenido que "Coords" (scores parciales) —
    # es exactamente el dato que el biplot dibuja, sin ningún reescalado
    # adicional con significado analítico (el biplot solo aplica un factor
    # de zoom puramente visual para que los puntos se vean bien en pantalla,
    # que no tiene interpretación y por eso no se exporta aquí).
    ws9 = wb.create_sheet("Biplot_Coords")
    if statis.coord_df is not None:
        dim1_col, dim2_col = "Dim1", "Dim2"
        puntos_out = statis.coord_df[["tabla","fila",dim1_col,dim2_col]].copy()
        puntos_out.columns = ["Tabla","Fila","Dim1","Dim2"]
        write_df(ws9, puntos_out.set_index("Tabla"),
                 "Pares ordenados del biplot del compromiso — puntos individuales "
                 "tabla-fila (idéntico a 'Coords'; el biplot no usa centroides "
                 "porque su promedio por tabla es matemáticamente cero) — "
                 f"{biplot_label(statis)} ({biplot_cita(statis)}); cargas "
                 "U·√λ (Lavit et al. 1994)")

    # ── Hoja 10: Verif_Biplot_Coords — pares ordenados del biplot de VERIFICACIÓN ──
    # Mismos puntos que dibuja plot_biplot_verificacion (pestaña "🧮 Verif.
    # Compromiso"): reconstruidos por concatenación Zc = [√α1·Z1|...|√αK·ZK]
    # con eigendescomposición fresca e independiente de la ya validada.
    # Puntos INDIVIDUALES (tabla-fila), no centroides, por la misma razón
    # que en Biplot_Coords. Incluye también los loadings de esa verificación
    # y la diferencia numérica entre ambas construcciones del compromiso
    # (debe ser ≈ 0). Solo aplica en modo longitudinal.
    if getattr(statis, "blocks_result", None) is not None:
        try:
            (Zc, S_concat, S_orig, diff_max, filas_totales,
             U_verif, lam_verif) = build_concatenated_compromise(statis)

            F_verif = Zc @ U_verif[:, :2]
            df_verif = pd.DataFrame(F_verif, columns=["Dim1", "Dim2"])
            df_verif["Tabla"] = [t for t, _ in filas_totales]
            df_verif["Fila"]  = [f for _, f in filas_totales]
            df_verif = df_verif[["Tabla","Fila","Dim1","Dim2"]].round(6)

            ws10 = wb.create_sheet("Verif_Biplot_Coords")
            write_df(ws10, df_verif.set_index("Tabla"),
                     "Pares ordenados del biplot de VERIFICACIÓN por concatenación "
                     "Zc'·Zc = Wc — puntos individuales tabla-fila — diferencia máx. "
                     f"vs. compromiso validado: {diff_max:.2e} (debe ser ≈ 0)")

            G_verif = U_verif[:, :2] * np.sqrt(np.maximum(lam_verif[:2], 0))
            load_verif_df = pd.DataFrame(
                np.round(G_verif, 6),
                index=statis.variable_names,
                columns=["Dim1", "Dim2"]
            )
            write_df(ws10, load_verif_df,
                     "Loadings de variables — biplot de verificación (misma escala √λ)",
                     start_row=len(df_verif) + 5)
        except Exception:
            pass  # Verificación no aplica (p.ej. modo bloques) — no bloquea el export

    # ── Trayectorias de las variables (Lavit et al., 1994, p. 107) ──────────
    vt = getattr(statis, "var_traj", None)
    if vt is not None and len(vt):
        ws11 = wb.create_sheet("Tray_Variables_Lavit")
        vt_out = vt.copy()
        vt_out["variable"] = vt_out["variable"].astype(str).str.replace("_score", "", regex=False)
        num_cols = [c for c in vt_out.columns if c.startswith("Dim")] + ["dist_compromiso"]
        vt_out[num_cols] = vt_out[num_cols].round(6)
        write_df(ws11, vt_out.set_index("tabla"),
                 "Posición de cada variable según cada tabla — Gk = Wk*·U·Λ^(-1/2) "
                 "(Lavit et al. 1994, p. 107). Σ αk·Gk = U·√λ (cargas del compromiso). "
                 "dist_compromiso = distancia euclídea a la carga de la variable en el plano retenido. "
                 "Si las tablas son periodos es una trayectoria temporal; si son entidades, "
                 "describe cuánto se aparta cada entidad del compromiso en cada variable.")

    wb.save(filepath)
    return filepath


def plot_variable_trajectories(statis: StatisDualResult, variable: str = None,
                               dim_x: int = 0, dim_y: int = 1,
                               n_etiquetas: int = 3) -> plt.Figure:
    """
    Trayectorias de las variables (Lavit et al., 1994, p. 107).
    Cada variable se dibuja en su posición del compromiso (U·√λ, flecha) y en
    su posición según cada tabla, Gk = Wk*·U·Λ^(-1/2) (puntos unidos a la
    punta de la flecha). Σ αk·Gk = U·√λ, por lo que la flecha es la media
    ponderada de los puntos. Ambos están en la misma escala: no se reescalan.

    variable = None → todas las variables; se etiquetan, por variable, las
    n_etiquetas tablas más alejadas del compromiso.
    variable = nombre → solo esa variable, con todas las tablas etiquetadas.
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font
    vt = getattr(statis, "var_traj", None)
    if vt is None or not len(vt):
        raise ValueError("Trayectorias de variables no disponibles (solo modo longitudinal).")
    nd = statis.n_dims_retained
    ix, iy = min(dim_x, nd - 1), min(dim_y, nd - 1)
    cx, cy = f"Dim{ix+1}", f"Dim{iy+1}"
    L = np.asarray(statis.eigenvectors[:, :nd]) * np.sqrt(np.maximum(statis.eigenvalues[:nd], 0))
    nombres = list(statis.variable_names)
    if variable is not None:
        if variable not in nombres:
            alt = [v for v in nombres if v.replace("_score", "") == variable]
            if not alt:
                raise ValueError(f"Variable '{variable}' no encontrada.")
            variable = alt[0]
        sel = [variable]
    else:
        sel = nombres
    colores = {v: PALETA_PAISES[i % len(PALETA_PAISES)] for i, v in enumerate(nombres)}

    fig, ax = plt.subplots(figsize=(11, 9), dpi=96)
    ax.set_facecolor("#F8F8F8"); fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", lw=0.5, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.7); ax.axvline(0, color="#BBBBBB", ls="--", lw=0.7)

    vals = [0.0]
    for v in sel:
        j = nombres.index(v); col = colores[v]
        sub = vt[vt["variable"] == v]
        lx, ly = float(L[j, ix]), float(L[j, iy])
        for _, rw in sub.iterrows():
            ax.plot([lx, rw[cx]], [ly, rw[cy]], color=col, lw=0.6, alpha=0.45, zorder=2)
        ax.scatter(sub[cx], sub[cy], s=28 if variable is None else 45, color=col,
                   edgecolors="white", linewidths=0.5, alpha=0.9, zorder=3)
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4, mutation_scale=13), zorder=5)
        ax.text(lx, ly, "  " + v.replace("_score", ""), fontsize=10, fontweight="bold",
                color=col, zorder=6, va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"))
        etiq = sub if variable is not None else sub.nlargest(n_etiquetas, "dist_compromiso")
        for _, rw in etiq.iterrows():
            ax.annotate(str(rw["tabla"])[:28], (rw[cx], rw[cy]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=col, zorder=6)
        vals += list(sub[cx]) + list(sub[cy]) + [lx, ly]
    lim = max(abs(float(x)) for x in vals) * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ve = statis.variance_explained
    ax.set_xlabel(f"Dim {ix+1} ({ve[ix]:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"Dim {iy+1} ({ve[iy]:.1f}% varianza)", fontsize=11)
    titulo = "todas las variables" if variable is None else variable.replace("_score", "")
    ax.set_title(f"Trayectorias de las variables — {titulo}\n"
                 f"Lavit et al. (1994, p. 107): Gk = Wk*·U·Λ^(-1/2); flecha = compromiso U·√λ",
                 fontsize=12, fontweight="bold", color=COLOR_PRIMARY)
    nota = ("Cada punto es la posición de la variable según una tabla; la flecha es su media ponderada por αk. "
            + ("Se etiquetan las tablas más alejadas del compromiso." if variable is None else ""))
    fig.text(0.01, 0.005, nota, fontsize=7, color="#555555", ha="left", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.subplots_adjust(bottom=0.09)
    return fig


def plot_hjbiplot_bloques(statis, dim_x: int = 0, dim_y: int = 1) -> plt.Figure:
    """
    HJ-Biplot para modo bloques/dimensiones.
    - Individuos como puntos azules (F_rows)
    - Vectores coloreados por bloque (F_vars)
    - Sin centroides por tabla (no aplica en modo bloques)
    """
    font = _get_font()
    if font:
        plt.rcParams["font.family"] = font

    ve   = statis.variance_explained
    G    = statis.F_vars        # (sum_pk × 2)
    Frow = statis.F_rows        # (n × 2)
    var_names = statis.variable_names

    # Identificar qué variables pertenecen a qué bloque
    bloques_info = getattr(statis, 'dimensiones_info', None) or                    getattr(statis, 'bloques_info', None)

    # Colores por bloque
    BLOQUE_COLS = ["#E63946", "#1A7A8A", "#2A9D8F", "#E9C46A", "#457B9D"]
    var_colors = {}
    if bloques_info:
        for bi, (nm, cols) in enumerate(bloques_info):
            col = BLOQUE_COLS[bi % len(BLOQUE_COLS)]
            for c in cols:
                var_colors[c] = (col, nm)
    else:
        for vn in (var_names or []):
            var_colors[vn] = ("#334155", "Variables")

    # Límite común para individuos y vectores
    all_vals = (list(Frow[:, dim_x]) + list(Frow[:, dim_y]) +
                list(G[:, dim_x])    + list(G[:, dim_y]))
    lim_b = max(abs(v) for v in all_vals) * 1.20
    lim_b = max(lim_b, 0.01)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=96)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", lw=0.6, zorder=0)
    ax.axhline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="#BBBBBB", ls="--", lw=0.8, alpha=0.7)

    # Individuos
    n_ind = Frow.shape[0]
    ax.scatter(Frow[:, dim_x], Frow[:, dim_y],
               s=18, color="#4A90D9", alpha=0.50,
               edgecolors="white", linewidths=0.4, zorder=3,
               label=f"Individuos (n={n_ind})")

    # Vectores por bloque
    lmax = np.sqrt((G**2).sum(axis=1)).max()
    fac  = (lim_b * 0.85) / lmax if lmax > 1e-10 else 1.0

    legend_bloques = {}
    for j, vname in enumerate(var_names or []):
        col, bloque_nm = var_colors.get(vname, ("#334155", "Variables"))
        gx = G[j, dim_x] * fac
        gy = G[j, dim_y] * fac
        if abs(gx) < 1e-12 and abs(gy) < 1e-12:
            continue
        ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.4, mutation_scale=10), zorder=6)
        nombre = str(vname).replace("_score", "")
        r2 = np.hypot(gx, gy)
        nx2 = gx/r2 if r2 > 1e-9 else 1.0
        ny2 = gy/r2 if r2 > 1e-9 else 0.0
        off2 = 0.06 * lim_b
        ax.text(gx + nx2*off2, gy + ny2*off2, nombre,
                fontsize=8.5, color=col, fontweight="bold",
                ha="left" if nx2 >= 0 else "right",
                va="bottom" if ny2 >= 0 else "top", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.85, edgecolor="none"))
        if bloque_nm not in legend_bloques:
            legend_bloques[bloque_nm] = col

    # Leyenda por bloque
    import matplotlib.patches as mpatches
    handles = [ax.scatter([], [], s=30, color="#4A90D9",
                          label=f"Individuos (n={n_ind})")]
    for nm, col in legend_bloques.items():
        handles.append(mpatches.Patch(color=col, label=nm))
    ax.legend(handles=handles, fontsize=9, framealpha=0.9,
              loc="upper left", edgecolor="#CCCCCC")

    d1_pct = ve[dim_x] if ve is not None and len(ve) > dim_x else 0
    d2_pct = ve[dim_y] if ve is not None and len(ve) > dim_y else 0
    ax.set_xlim(-lim_b, lim_b); ax.set_ylim(-lim_b, lim_b)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dimensión {dim_x+1}  ({d1_pct:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"Dimensión {dim_y+1}  ({d2_pct:.1f}% varianza)", fontsize=11)
    K = len(bloques_info) if bloques_info else "?"
    ax.set_title(
        f"HJ-Biplot del Compromiso — Modo Dimensiones  (n={n_ind} individuos)\n"
        f"Galindo Villardón (1986) · Lavit et al. (1994)",
        fontsize=12, fontweight="bold", color="#0D2545", pad=12
    )
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.09)
    return fig
