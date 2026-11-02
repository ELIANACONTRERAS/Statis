"""
m7_cluster.py — Análisis de clúster sobre coordenadas del compromiso STATIS.

Compatible con modo longitudinal y modo bloques/dimensiones.
Implementación manual sin scipy:
  - K-means (Lloyd, iteración manual con numpy)
  - Jerárquico aglomerativo (Ward, distancias euclidianas manuales)

Las coordenadas de entrada son siempre F_rows (n × n_dims) del compromiso.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Resultados
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterResult:
    method:       str   = ""          # "kmeans" | "ward"
    n_clusters:   int   = 0
    labels:       Optional[np.ndarray] = None   # (n,) enteros 0..k-1
    centroids:    Optional[np.ndarray] = None   # (k × n_dims)
    inertia:      float = 0.0         # K-means: SSW total
    linkage_mat:  Optional[np.ndarray] = None   # Ward: (n-1 × 4)
    silhouette:   Optional[float] = None
    calinski:     Optional[float] = None
    davies:       Optional[float] = None
    coords:       Optional[np.ndarray] = None   # copia de F_rows usadas
    labels_str:   Optional[list] = None         # etiquetas de individuos


# ══════════════════════════════════════════════════════════════════════════════
# Índices de validación (implementación manual)
# ══════════════════════════════════════════════════════════════════════════════

def _eucl(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette medio — implementación vectorizada con numpy.
    O(n² * p) con operaciones de matriz, ~50x más rápido que el bucle Python.
    """
    n = len(X)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0

    # Matriz de distancias euclidianas completa (n × n) — una sola operación
    # ||xi - xj||² = ||xi||² + ||xj||² - 2·xi·xj'
    sq = np.sum(X**2, axis=1)
    D  = np.sqrt(np.maximum(
        sq[:, None] + sq[None, :] - 2.0 * (X @ X.T),
        0.0
    ))  # (n × n)

    s = np.zeros(n)
    for i in range(n):
        cls_i = labels[i]
        # a(i): distancia media a los demás puntos del mismo clúster
        same_mask = (labels == cls_i)
        same_mask[i] = False
        n_same = same_mask.sum()
        a_i = D[i, same_mask].mean() if n_same > 0 else 0.0

        # b(i): distancia media mínima a cualquier otro clúster
        b_i = np.inf
        for c in uniq:
            if c == cls_i: continue
            other_mask = (labels == c)
            b_c = D[i, other_mask].mean()
            if b_c < b_i:
                b_i = b_c

        denom = max(a_i, b_i)
        s[i]  = (b_i - a_i) / denom if denom > 1e-14 else 0.0

    return float(s.mean())


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabász (implementación manual)."""
    n, p = X.shape
    k = len(np.unique(labels))
    if k < 2: return 0.0
    grand_mean = X.mean(axis=0)
    SSB = sum(
        np.sum(labels == c) * np.sum((X[labels == c].mean(axis=0) - grand_mean)**2)
        for c in np.unique(labels)
    )
    SSW = sum(
        np.sum((X[labels == c] - X[labels == c].mean(axis=0))**2)
        for c in np.unique(labels)
    )
    if SSW < 1e-14: return 0.0
    return float((SSB / (k - 1)) / (SSW / (n - k)))


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin (implementación manual)."""
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2: return 0.0
    centroids = np.array([X[labels == c].mean(axis=0) for c in uniq])
    s = np.array([np.mean([_eucl(X[i], centroids[ci])
                            for i in np.where(labels == uniq[ci])[0]])
                  for ci in range(k)])
    db = []
    for i in range(k):
        vals = [(s[i] + s[j]) / max(_eucl(centroids[i], centroids[j]), 1e-14)
                for j in range(k) if j != i]
        db.append(max(vals))
    return float(np.mean(db))


# ══════════════════════════════════════════════════════════════════════════════
# K-means manual
# ══════════════════════════════════════════════════════════════════════════════

def kmeans(X: np.ndarray, k: int, n_init: int = 10,
           max_iter: int = 300, random_state: int = 42) -> ClusterResult:
    """K-means con múltiples inicializaciones, implementación manual."""
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_inertia = np.inf
    best_centroids = None

    for _ in range(n_init):
        # Inicialización K-means++
        idx = [rng.integers(0, len(X))]
        for _ in range(k - 1):
            d2 = np.array([min(np.sum((X[i] - X[j])**2) for j in idx)
                           for i in range(len(X))])
            probs = d2 / d2.sum()
            idx.append(rng.choice(len(X), p=probs))
        centroids = X[idx].copy().astype(float)

        labels = np.zeros(len(X), dtype=int)
        for it in range(max_iter):
            # Asignar
            dists = np.array([[np.sum((x - c)**2) for c in centroids] for x in X])
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels) and it > 0:
                break
            labels = new_labels
            # Actualizar centroides
            for ci in range(k):
                pts = X[labels == ci]
                if len(pts) > 0:
                    centroids[ci] = pts.mean(axis=0)

        inertia = float(sum(np.sum((X[labels == ci] - centroids[ci])**2)
                            for ci in range(k)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()
            best_centroids = centroids.copy()

    res = ClusterResult(
        method="kmeans", n_clusters=k,
        labels=best_labels, centroids=best_centroids,
        inertia=best_inertia, coords=X.copy()
    )
    res.silhouette = silhouette_score(X, best_labels)
    res.calinski   = calinski_harabasz(X, best_labels)
    res.davies     = davies_bouldin(X, best_labels)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Jerárquico Ward manual
# ══════════════════════════════════════════════════════════════════════════════

def ward_linkage(X: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de enlace Ward usando la fórmula de Lance-Williams
    con actualización vectorizada (O(n²) por paso → O(n²·logn) total).
    Para n > 200 es ~100x más rápido que la versión O(n³).
    """
    n = len(X)
    # Distancia Ward inicial entre todos los pares: D[i,j] = ni*nj/(ni+nj) * ||xi-xj||²
    # Con ni=nj=1 al inicio: D[i,j] = 0.5 * ||xi-xj||²
    # Usar broadcasting vectorizado
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (n × n × p)
    D = 0.5 * np.sum(diff**2, axis=2)                  # (n × n)
    np.fill_diagonal(D, np.inf)

    sizes  = np.ones(n, dtype=float)
    active = list(range(n))
    linkage = []
    next_id = n

    # Mapa de índice original → posición en la matriz D
    # (la matriz D permanece fija en n×n; usamos máscaras de activos)
    id_to_row = {i: i for i in range(n)}   # id clúster → fila en D
    row_to_id = {i: i for i in range(n)}   # fila → id clúster

    for step in range(n - 1):
        # Encontrar el par de mínima distancia entre clústeres activos
        active_rows = [id_to_row[a] for a in active]
        sub = D[np.ix_(active_rows, active_rows)]
        flat_idx = np.argmin(sub)
        ii, jj = divmod(int(flat_idx), len(active_rows))
        ri, rj = active_rows[ii], active_rows[jj]
        ci, cj = row_to_id[ri], row_to_id[rj]
        ni, nj = sizes[ri], sizes[rj]
        d_merge = float(D[ri, rj])
        linkage.append([ci, cj, np.sqrt(d_merge), ni + nj])

        # Actualizar distancias usando la fórmula de Lance-Williams para Ward:
        # D(k, i∪j) = (nk+ni)/(nk+ni+nj)*D(k,i) + (nk+nj)/(nk+ni+nj)*D(k,j)
        #             - nk/(nk+ni+nj)*D(i,j)
        n_new = ni + nj
        for rk in active_rows:
            if rk == ri or rk == rj:
                continue
            nk = sizes[rk]
            dki = D[rk, ri]
            dkj = D[rk, rj]
            dij = D[ri, rj]
            total = nk + n_new
            d_new = ((nk + ni) / total * dki +
                     (nk + nj) / total * dkj -
                     nk       / total * dij)
            D[ri, rk] = D[rk, ri] = max(d_new, 0.0)

        # El nuevo clúster ocupa la fila ri; desactivar ci y cj, activar next_id
        sizes[ri] = n_new
        active.remove(ci)
        active.remove(cj)
        active.append(next_id)
        # Actualizar mapas: next_id → fila ri
        id_to_row[next_id] = ri
        row_to_id[ri] = next_id
        del id_to_row[ci]; del id_to_row[cj]
        D[rj, :] = np.inf; D[:, rj] = np.inf
        next_id += 1

    return np.array(linkage, dtype=float)


def cut_tree(linkage_mat: np.ndarray, k: int, n: int) -> np.ndarray:
    """Corta el dendrograma para obtener k clústeres."""
    # Reconstruir fusiones
    clusters = {i: [i] for i in range(n)}
    next_id  = n
    for row in linkage_mat:
        ci, cj = int(row[0]), int(row[1])
        clusters[next_id] = clusters[ci] + clusters[cj]
        del clusters[ci]; del clusters[cj]
        next_id += 1

    # Los últimos k-1 pasos definen k clústeres
    # Deshacer las últimas k-1 fusiones
    clusters2 = {i: [i] for i in range(n)}
    next_id2  = n
    n_fus = len(linkage_mat)
    for step, row in enumerate(linkage_mat):
        if step >= n_fus - (k - 1):
            break
        ci, cj = int(row[0]), int(row[1])
        clusters2[next_id2] = clusters2[ci] + clusters2[cj]
        del clusters2[ci]; del clusters2[cj]
        next_id2 += 1

    labels = np.zeros(n, dtype=int)
    for grp_idx, (_, pts) in enumerate(clusters2.items()):
        for pt in pts:
            labels[pt] = grp_idx
    return labels


def ward_cluster(X: np.ndarray, k: int) -> ClusterResult:
    """Clúster jerárquico Ward con corte en k clústeres."""
    lmat    = ward_linkage(X)
    labels  = cut_tree(lmat, k, len(X))
    centroids = np.array([X[labels == ci].mean(axis=0)
                           for ci in range(k)])
    res = ClusterResult(
        method="ward", n_clusters=k,
        labels=labels, centroids=centroids,
        linkage_mat=lmat, coords=X.copy()
    )
    res.silhouette = silhouette_score(X, labels)
    res.calinski   = calinski_harabasz(X, labels)
    res.davies     = davies_bouldin(X, labels)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Búsqueda automática de k óptimo
# ══════════════════════════════════════════════════════════════════════════════

def optimal_k(X: np.ndarray, k_max: int = 8,
              method: str = "kmeans") -> pd.DataFrame:
    """Calcula índices de validación para k=2..k_max."""
    rows = []
    for k in range(2, min(k_max + 1, len(X))):
        if method == "kmeans":
            res = kmeans(X, k)
        else:
            res = ward_cluster(X, k)
        rows.append({
            "k": k,
            "Inercia": round(res.inertia, 4),
            "Silhouette": round(res.silhouette, 4),
            "Calinski-H": round(res.calinski, 2),
            "Davies-B":   round(res.davies, 4),
        })
    return pd.DataFrame(rows).set_index("k")


# ══════════════════════════════════════════════════════════════════════════════
# Gráficos de clúster
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_PALETTE = [
    "#E63946","#1A7A8A","#E9C46A","#2A9D8F","#457B9D",
    "#F4A261","#6A4C93","#2DC653","#FF6B6B","#4ECDC4"
]


def plot_cluster_scatter(res: ClusterResult,
                         dim_x: int = 0, dim_y: int = 1,
                         var_exp: Optional[list] = None,
                         labels_str: Optional[list] = None,
                         var_names: Optional[list] = None,
                         loadings: Optional[np.ndarray] = None,
                         title: str = "Clúster — Espacio del compromiso",
                         show_labels: bool = False) -> plt.Figure:
    """
    Scatter estático: puntos sin etiquetas + vectores de variables.
    Título compacto en una línea, leyenda pequeña fuera del área de datos,
    vectores bien proporcionados.
    """
    X = res.coords
    k = res.n_clusters
    lbs = res.labels

    # Figura más ancha para dar espacio a la leyenda exterior
    # Figura grande para renderizar bien en el ScrollArea (880×700px)
    fig = plt.figure(figsize=(12, 8.5), dpi=96)
    # Área del gráfico ocupa ~76% del ancho; leyenda en el 24% restante
    ax = fig.add_axes([0.08, 0.09, 0.74, 0.84])
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#E8E8E8", lw=0.5, zorder=0)
    ax.axhline(0, color="#CCCCCC", lw=0.8, ls="--")
    ax.axvline(0, color="#CCCCCC", lw=0.8, ls="--")

    lim_pts = max(np.abs(X[:, dim_x]).max(), np.abs(X[:, dim_y]).max()) * 1.12
    lim_pts = max(lim_pts, 0.5)

    for ci in range(k):
        mask = lbs == ci
        col  = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
        ax.scatter(X[mask, dim_x], X[mask, dim_y],
                   s=32, color=col, alpha=0.75,
                   edgecolors="white", linewidths=0.5,
                   zorder=3, label=f"Clúster {ci+1}  (n={mask.sum()})")
        if show_labels and labels_str is not None:
            for idx_pt in np.where(mask)[0]:
                ax.text(X[idx_pt, dim_x], X[idx_pt, dim_y] + 0.02*lim_pts,
                        str(labels_str[idx_pt]), fontsize=5, color=col,
                        ha='center', va='bottom', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.06',
                                  facecolor='white', alpha=0.55, edgecolor='none'))

    # Centroides
    if res.centroids is not None:
        for ci, cen in enumerate(res.centroids):
            col = CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
            ax.scatter(cen[dim_x], cen[dim_y],
                       s=180, color=col, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=6)
            ax.text(cen[dim_x] + 0.025*lim_pts, cen[dim_y] + 0.025*lim_pts,
                    f"C{ci+1}", fontsize=8, fontweight="bold",
                    color=col, ha="left", zorder=7)

    # Vectores de variables — flechas finas, etiquetas sin solapamiento
    if loadings is not None and var_names is not None:
        nd = loadings.shape[1] if loadings.ndim > 1 else 1
        ix = min(dim_x, nd-1); iy = min(dim_y, nd-1)
        L2 = np.column_stack([loadings[:, ix], loadings[:, iy]])
        lmax_L = np.sqrt((L2**2).sum(axis=1)).max()
        fac_L  = (lim_pts * 0.50) / lmax_L if lmax_L > 1e-10 else 1.0

        # Pre-calcular posiciones escaladas
        scaled = []
        for j, vname in enumerate(var_names):
            gx = float(L2[j, 0]) * fac_L
            gy = float(L2[j, 1]) * fac_L
            if abs(gx) < 1e-9 and abs(gy) < 1e-9:
                continue
            scaled.append((j, str(vname).replace("_score",""), gx, gy))

        # Dibujar flechas (muy finas)
        for _, _, gx, gy in scaled:
            ax.annotate("", xy=(gx, gy), xytext=(0, 0),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="black",
                            lw=0.8,           # flecha muy fina
                            mutation_scale=7  # punta pequeña
                        ), zorder=8)

        # Posicionar etiquetas con separación anti-solapamiento
        # Ordenar por ángulo para distribuir radialmente
        import math
        scaled_ang = sorted(scaled,
                            key=lambda t: math.atan2(t[3], t[2]))
        placed = []   # (lx, ly) ya colocadas
        min_sep = 0.12 * lim_pts   # separación mínima entre etiquetas

        for _, nombre, gx, gy in scaled_ang:
            r = np.hypot(gx, gy)
            nx_ = gx/r if r > 1e-9 else 1.0
            ny_ = gy/r if r > 1e-9 else 0.0
            # Offset base
            base_off = 0.10 * r + 0.035 * lim_pts
            lx = gx + nx_ * base_off
            ly = gy + ny_ * base_off

            # Empujar si hay solapamiento con etiquetas ya colocadas
            for attempt in range(8):
                too_close = any(
                    np.hypot(lx - px, ly - py) < min_sep
                    for px, py in placed
                )
                if not too_close:
                    break
                # Aumentar offset radialmente
                extra = base_off + attempt * 0.08 * lim_pts
                lx = gx + nx_ * extra
                ly = gy + ny_ * extra

            placed.append((lx, ly))
            ax.text(lx, ly, nombre,
                    fontsize=6, color="black", fontweight="bold",
                    ha="left" if nx_ >= 0 else "right",
                    va="bottom" if ny_ >= 0 else "top",
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.07",
                              facecolor="white", alpha=0.90,
                              edgecolor="none", linewidth=0))

    xe = f" ({var_exp[dim_x]:.1f}%)" if var_exp else ""
    ye = f" ({var_exp[dim_y]:.1f}%)" if var_exp else ""
    ax.set_xlim(-lim_pts, lim_pts)
    ax.set_ylim(-lim_pts, lim_pts)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim {dim_x+1}{xe}", fontsize=10)
    ax.set_ylabel(f"Dim {dim_y+1}{ye}", fontsize=10)
    ax.tick_params(labelsize=8)

    # Título compacto: una línea con método e índices
    sil = f"{res.silhouette:.3f}" if res.silhouette else "—"
    ax.set_title(
        f"{res.method.upper()} k={k} | Silhouette={sil}  "
        f"CH={res.calinski:.1f}  DB={res.davies:.3f}",
        fontsize=8.5, color="#334155", pad=6, loc="left"
    )

    # Leyenda fuera del área, a la derecha — letra muy pequeña
    leg = ax.legend(
        fontsize=7, framealpha=0.95, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
        markerscale=0.9, handlelength=1.0,
        borderpad=0.6, labelspacing=0.35,
        edgecolor="#CCCCCC"
    )
    ax.spines[["top","right"]].set_visible(False)
    return fig


def plot_cluster_interactive(res: ClusterResult,
                              dim_x: int = 0, dim_y: int = 1,
                              var_exp: Optional[list] = None,
                              labels_str: Optional[list] = None,
                              var_names: Optional[list] = None,
                              loadings: Optional[np.ndarray] = None,
                              title: str = "Clúster interactivo") -> str:
    """
    Gráfico interactivo HTML con Plotly:
    - Hover muestra etiqueta, clúster y coordenadas
    - Vectores de variables como anotaciones
    - Colores por clúster, centroides con estrella
    Devuelve HTML como string.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<h3>Plotly no instalado. Ejecute: pip install plotly</h3>"

    X   = res.coords
    k   = res.n_clusters
    lbs = res.labels

    PLOTLY_PAL = ["#E63946","#1A7A8A","#E9C46A","#2A9D8F",
                  "#457B9D","#F4A261","#6A4C93","#2DC653"]

    fig = go.Figure()

    # Puntos por clúster
    for ci in range(k):
        mask = lbs == ci
        col  = PLOTLY_PAL[ci % len(PLOTLY_PAL)]
        pts_x = X[mask, dim_x].tolist()
        pts_y = X[mask, dim_y].tolist()
        if labels_str is not None:
            hover = [f"<b>{labels_str[i]}</b><br>Clúster {ci+1}<br>"
                     f"Dim{dim_x+1}: {X[i,dim_x]:.3f}<br>Dim{dim_y+1}: {X[i,dim_y]:.3f}"
                     for i in np.where(mask)[0]]
        else:
            hover = [f"Clúster {ci+1}<br>Dim{dim_x+1}: {x:.3f}<br>Dim{dim_y+1}: {y:.3f}"
                     for x, y in zip(pts_x, pts_y)]
        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers",
            name=f"Clúster {ci+1} (n={mask.sum()})",
            marker=dict(color=col, size=9, opacity=0.80,
                        line=dict(color="white", width=0.8)),
            text=hover, hoverinfo="text"
        ))

    # Centroides
    if res.centroids is not None:
        for ci, cen in enumerate(res.centroids):
            col = PLOTLY_PAL[ci % len(PLOTLY_PAL)]
            fig.add_trace(go.Scatter(
                x=[cen[dim_x]], y=[cen[dim_y]],
                mode="markers+text",
                name=f"Centroide C{ci+1}",
                marker=dict(symbol="star", color=col, size=20,
                            line=dict(color="black", width=1)),
                text=[f"C{ci+1}"], textposition="top center",
                textfont=dict(size=12, color=col, family="Arial Black"),
                hovertext=f"Centroide C{ci+1}<br>Dim{dim_x+1}: {cen[dim_x]:.3f}<br>"
                          f"Dim{dim_y+1}: {cen[dim_y]:.3f}",
                hoverinfo="text", showlegend=True
            ))

    # Vectores de variables — flechas + etiquetas como scatter text
    annotations = []
    if loadings is not None and var_names is not None:
        nd = loadings.shape[1] if loadings.ndim > 1 else 1
        ix = min(dim_x, nd-1); iy = min(dim_y, nd-1)
        L2 = np.column_stack([loadings[:, ix], loadings[:, iy]])
        lim_pts = max(np.abs(X[:, dim_x]).max(), np.abs(X[:, dim_y]).max()) * 1.15
        lmax_L  = np.sqrt((L2**2).sum(axis=1)).max()
        fac_L   = (lim_pts * 0.55) / lmax_L if lmax_L > 1e-10 else 1.0

        tx_vec, ty_vec, txt_vec = [], [], []
        for j, vname in enumerate(var_names):
            gx = float(L2[j, 0]) * fac_L
            gy = float(L2[j, 1]) * fac_L
            if abs(gx) < 1e-9 and abs(gy) < 1e-9: continue
            nombre = str(vname).replace("_score", "")

            # Flecha como anotación (arrowhead visible en Plotly)
            annotations.append(dict(
                x=gx, y=gy, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowwidth=2.0, arrowcolor="black",
                text="", showarrow=True
            ))

            # Etiqueta desplazada en la dirección del vector
            r = np.hypot(gx, gy)
            nx_ = gx/r if r > 1e-9 else 1.0
            ny_ = gy/r if r > 1e-9 else 0.0
            off = 0.10 * r + 0.04 * lim_pts
            tx_vec.append(gx + nx_*off)
            ty_vec.append(gy + ny_*off)
            txt_vec.append(f"<b>{nombre}</b>")

        # Etiquetas como un único scatter para que Plotly pueda gestionar
        if tx_vec:
            fig.add_trace(go.Scatter(
                x=tx_vec, y=ty_vec,
                mode="text",
                text=txt_vec,
                textfont=dict(size=10, color="black", family="Arial"),
                showlegend=False,
                hoverinfo="skip",
                name="Variables"
            ))

    xe = f" ({var_exp[dim_x]:.1f}%)" if var_exp else ""
    ye = f" ({var_exp[dim_y]:.1f}%)" if var_exp else ""
    sil = f"{res.silhouette:.3f}" if res.silhouette else "—"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br>"
                        f"<span style='font-size:12px'>"
                        f"{res.method.upper()} k={k} | "
                        f"Silhouette={sil}  CH={res.calinski:.1f}  DB={res.davies:.3f}"
                        f"</span>",
                   x=0.5, xanchor="center",
                   font=dict(size=14, color="#0D2545")),
        xaxis=dict(title=f"Dim {dim_x+1}{xe}", zeroline=True,
                   zerolinecolor="#CCCCCC", gridcolor="#EEEEEE"),
        yaxis=dict(title=f"Dim {dim_y+1}{ye}", zeroline=True,
                   zerolinecolor="#CCCCCC", gridcolor="#EEEEEE",
                   scaleanchor="x"),
        annotations=annotations,
        legend=dict(orientation="v", x=1.01, xanchor="left",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#CCCCCC", borderwidth=1,
                    font=dict(size=11),
                    itemsizing="constant",
                    tracegroupgap=4),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=60, r=200, t=100, b=60),
        hovermode="closest"
    )
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def plot_cluster_dendrogram(res: ClusterResult,
                             k: int = 3,
                             labels_str: Optional[list] = None,
                             title: str = "Dendrograma Ward") -> plt.Figure:
    """Dendrograma con línea de corte para k clústeres."""
    if res.linkage_mat is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dendrograma solo disponible con método Ward",
                ha="center", va="center")
        return fig

    lmat = res.linkage_mat
    n    = res.coords.shape[0]

    fig, ax = plt.subplots(figsize=(max(12, n*0.12), 6), dpi=110)

    # Dibujar dendrograma manualmente (estilo simplificado)
    # Posición horizontal de cada punto original
    pos = {i: i for i in range(n)}
    next_id = n
    y_max   = lmat[-1, 2]

    for row in lmat:
        ci, cj, h, _ = int(row[0]), int(row[1]), row[2], row[3]
        xi = pos.get(ci, ci)
        xj = pos.get(cj, cj)
        xm = (xi + xj) / 2
        ax.plot([xi, xi], [0, h], color="#457B9D", lw=1.0, alpha=0.7)
        ax.plot([xj, xj], [0, h], color="#457B9D", lw=1.0, alpha=0.7)
        ax.plot([xi, xj], [h, h], color="#457B9D", lw=1.0, alpha=0.7)
        pos[next_id] = xm
        next_id += 1

    # Línea de corte
    if len(lmat) >= k - 1:
        cut_h = lmat[-(k-1), 2] + 0.01
        ax.axhline(cut_h, color="#E63946", lw=1.8, ls="--",
                   label=f"Corte k={k}")
        ax.legend(fontsize=10)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#0D2545")
    ax.set_ylabel("Distancia Ward", fontsize=10)
    ax.set_xlabel("Individuos", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_validation_indices(df_valid: pd.DataFrame,
                             title: str = "Índices de validación") -> plt.Figure:
    """4 subplots con los índices para seleccionar k óptimo."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=110)
    axes = axes.flatten()
    ks   = df_valid.index.tolist()

    specs = [
        ("Silhouette",  "#1A7A8A", "↑ mayor es mejor"),
        ("Calinski-H",  "#E9C46A", "↑ mayor es mejor"),
        ("Davies-B",    "#E63946", "↓ menor es mejor"),
        ("Inercia",     "#457B9D", "Método del codo"),
    ]

    for ax, (col, color, note) in zip(axes, specs):
        if col not in df_valid.columns: continue
        ax.plot(ks, df_valid[col], "o-", color=color, lw=2, ms=7)
        ax.set_title(f"{col}  ({note})", fontsize=10, fontweight="bold")
        ax.set_xlabel("k", fontsize=9); ax.grid(True, alpha=0.4)
        ax.set_xticks(ks)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", color="#0D2545", y=1.01)
    plt.tight_layout()
    return fig
