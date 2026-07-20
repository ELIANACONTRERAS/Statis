"""
m6_descriptives_plot.py — Gráficos de estadísticas descriptivas.
Funciona para cualquier conjunto de variables numéricas.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_desc_heatmap(desc_df: pd.DataFrame,
                      title: str = "Estadísticas descriptivas") -> plt.Figure:
    """Heatmap de la tabla de estadísticas (variables × stats)."""
    cols_show = ['Media', 'D.E.', 'Mín', 'Q1', 'Mediana', 'Q3', 'Máx', 'CV', 'Asimetría']
    available = [c for c in cols_show if c in desc_df.columns]
    data = desc_df[available].astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(available)*1.1),
                                    max(5, len(data)*0.45)), dpi=120)
    # Normalizar por columna para el color
    normed = (data - data.min()) / (data.max() - data.min() + 1e-10)
    im = ax.imshow(normed.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index.tolist(), fontsize=8)

    for i in range(len(data)):
        for j, col in enumerate(available):
            val = data.iloc[i, j]
            txt = f'{val:.3f}' if abs(val) < 100 else f'{val:.1f}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7.5, color='black' if normed.iloc[i,j] < 0.6 else 'white')

    ax.set_title(title, fontsize=11, fontweight='bold', color='#0D2545', pad=10)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_desc_boxplot(df_raw: pd.DataFrame,
                      cols: list,
                      title: str = "Distribución de variables",
                      col_grupo: Optional[str] = None) -> plt.Figure:
    """Boxplot de todas las variables. Agrupa por col_grupo si se indica."""
    sub = df_raw[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')

    n = len(cols)
    ncols_p = min(6, n)
    nrows_p = int(np.ceil(n / ncols_p))
    fig, axes = plt.subplots(nrows_p, ncols_p,
                              figsize=(ncols_p*2.2, nrows_p*2.8), dpi=110)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    palette = ['#1A7A8A','#E63946','#E9C46A','#2A9D8F','#457B9D','#F4A261']

    for i, col in enumerate(cols):
        ax = axes[i]
        data_col = sub[col].dropna()
        ax.boxplot(data_col, patch_artist=True,
                   boxprops=dict(facecolor=palette[i % len(palette)], alpha=0.7),
                   medianprops=dict(color='black', lw=2),
                   whiskerprops=dict(color='#334155'),
                   capprops=dict(color='#334155'),
                   flierprops=dict(marker='o', markersize=3,
                                   markerfacecolor='#E63946', alpha=0.5))
        ax.set_title(col, fontsize=8.5, fontweight='bold', color='#0D2545')
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines[['top','right']].set_visible(False)
        # Anotar mediana
        med = float(data_col.median())
        ax.text(1.25, med, f'{med:.2f}', va='center', fontsize=7, color='#334155')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight='bold', color='#0D2545', y=1.01)
    plt.tight_layout()
    return fig


def plot_desc_correlation(corr_df: pd.DataFrame,
                           title: str = "Matriz de correlación") -> plt.Figure:
    """Heatmap de correlación con máscara triangular."""
    n = len(corr_df)
    fig, ax = plt.subplots(figsize=(max(7, n*0.55), max(6, n*0.55)), dpi=110)

    mask = np.triu(np.ones_like(corr_df.values, dtype=bool), k=1)
    vals = corr_df.values.copy()
    vals[mask] = np.nan

    im = ax.imshow(vals, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, label='r Pearson')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    lbs = list(corr_df.columns)
    ax.set_xticklabels(lbs, fontsize=max(5, 9 - n//8), rotation=45, ha='right')
    ax.set_yticklabels(lbs, fontsize=max(5, 9 - n//8))

    for i in range(n):
        for j in range(n):
            if not mask[i, j] or i == j:
                v = corr_df.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=max(5, 8 - n//10),
                            color='black' if abs(v) < 0.6 else 'white')

    ax.set_title(title, fontsize=11, fontweight='bold', color='#0D2545', pad=10)
    plt.tight_layout()
    return fig


def plot_desc_histograms(df_raw: pd.DataFrame,
                          cols: list,
                          title: str = "Histogramas") -> plt.Figure:
    """Histograma + KDE por variable."""
    sub = df_raw[cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')

    n = len(cols)
    ncols_p = min(6, n)
    nrows_p = int(np.ceil(n / ncols_p))
    fig, axes = plt.subplots(nrows_p, ncols_p,
                              figsize=(ncols_p*2.4, nrows_p*2.6), dpi=110)
    axes = np.array(axes).flatten() if n > 1 else [axes]
    palette = ['#1A7A8A','#E63946','#E9C46A','#2A9D8F','#457B9D','#F4A261']

    for i, col in enumerate(cols):
        ax = axes[i]
        data_col = sub[col].dropna()
        n_bins = min(20, max(5, len(data_col.unique())))
        ax.hist(data_col, bins=n_bins,
                color=palette[i % len(palette)], alpha=0.7, edgecolor='white')
        try:
            from scipy import stats as sp
            xr = np.linspace(data_col.min(), data_col.max(), 100)
            kde = sp.gaussian_kde(data_col)
            ax2 = ax.twinx()
            ax2.plot(xr, kde(xr), color='#334155', lw=1.5)
            ax2.set_yticks([])
        except ImportError:
            pass
        ax.set_title(col, fontsize=8.5, fontweight='bold', color='#0D2545')
        ax.tick_params(labelsize=7)
        ax.spines[['top','right']].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight='bold', color='#0D2545', y=1.01)
    plt.tight_layout()
    return fig
