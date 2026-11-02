"""
M7_outputs_interactive.py — Gráficos interactivos (Plotly)
===========================================================
Genera versiones interactivas de los gráficos clave:
  I01 · HJ-Biplot interactivo (hover, zoom, pan)
  I02 · Trayectorias interactivas (mapa de trayectorias con slider de dimensiones)
  I03 · Trayectoria destacada individual

Los gráficos se pueden:
  • Ver inline en la ventana PyQt6 (via QWebEngineView que carga HTML en memoria)
  • Descargar como HTML standalone offline (sin conexión a internet)

NOTA: plotly.io.to_html(include_plotlyjs="cdn") usa CDN.
      Para offline se usa include_plotlyjs=True (incluye JS ~3MB en el HTML).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from core.m0_config import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT,
    COLOR_BG, COLOR_TEXT, TRAJECTORY_PALETTE
)
from core.m4_interstructure import StatisDualResult, biplot_label, biplot_cita
from core.m5_intrastructure import IntrastructureResult


# ─── Paleta ──────────────────────────────────────────────────────────────────

def _layout_base(title: str, xlab: str, ylab: str) -> dict:
    """Layout base compartido para todos los gráficos."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=COLOR_PRIMARY, family="Arial"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=xlab, zeroline=True, zerolinecolor="#CBD5E0",
                   zerolinewidth=1.2, gridcolor="#E2E8F0",
                   tickfont=dict(size=11, color=COLOR_TEXT)),
        yaxis=dict(title=ylab, zeroline=True, zerolinecolor="#CBD5E0",
                   zerolinewidth=1.2, gridcolor="#E2E8F0",
                   tickfont=dict(size=11, color=COLOR_TEXT)),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(family="Arial", color=COLOR_TEXT),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CBD5E0",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="closest",
    )


# ─── I01 · HJ-Biplot interactivo ─────────────────────────────────────────────

PALETA_I = [
    '#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261',
    '#6A4C93','#1D3557','#E76F51','#264653','#A8DADC',
    '#F72585','#4361EE','#3A86FF','#8338EC','#FB5607',
    '#006400','#FF006E','#8AC926','#1982C4','#FFCA3A',
    '#6A994E','#BC6C25','#540B0E','#023E8A','#7209B7','#B5179E',
    '#0077B6','#D62828','#F77F00','#52B788',
]


def interactive_hjbiplot(statis: StatisDualResult,
                          dim_x: int = 0,
                          dim_y: int = 1) -> go.Figure:
    """
    HJ-Biplot interactivo — Compromiso STATIS Dual.

    FIX: se grafican los puntos INDIVIDUALES (tabla-fila, ej. país-año),
    no los centroides por tabla — el promedio de cada tabla es siempre
    (0,0) por construcción (cada tabla se centra por su propia media antes
    de proyectar), por lo que graficar centroides colapsaba todos los
    puntos en el origen. Ver diagnóstico completo en plot_hjbiplot
    (m6_outputs_static.py).

    - Un trace por tabla (país), con TODOS sus puntos individuales
      (uno por fila/año) — permite mostrar/ocultar cada país desde la
      leyenda de Plotly con un clic.
    - Vectores de variables en negro.
    - Hover: tabla, fila (año, si está disponible) y coordenadas exactas
      de cada punto.
    """
    ve       = statis.variance_explained
    xlab     = f"Dimensión {dim_x+1}  ({ve[dim_x]:.1f}% varianza)"
    ylab     = f"Dimensión {dim_y+1}  ({ve[dim_y]:.1f}% varianza)"
    etiqueta_biplot = biplot_label(statis)

    coord_df  = statis.coord_df
    dim1_col  = f"Dim{dim_x+1}"
    dim2_col  = f"Dim{dim_y+1}"
    G         = statis.F_vars
    var_names = statis.variable_names

    tablas_sorted = sorted(statis.tablas)
    color_t = {t: PALETA_I[i % len(PALETA_I)]
               for i, t in enumerate(tablas_sorted)}

    fig = go.Figure()

    # ── Puntos individuales (tabla-fila) — un trace por tabla ───────────────
    tiene_fila = "fila" in coord_df.columns
    for tabla in tablas_sorted:
        sub = coord_df[coord_df["tabla"] == tabla]
        col = color_t.get(tabla, "#4A90D9")

        if tiene_fila:
            hovertemplate = (
                f"<b>{tabla}</b><br>%{{customdata}}<br>"
                f"Dim{dim_x+1}: %{{x:.3f}}<br>"
                f"Dim{dim_y+1}: %{{y:.3f}}<extra></extra>"
            )
            customdata = sub["fila"].astype(str).tolist()
        else:
            hovertemplate = (
                f"<b>{tabla}</b><br>"
                f"Dim{dim_x+1}: %{{x:.3f}}<br>"
                f"Dim{dim_y+1}: %{{y:.3f}}<extra></extra>"
            )
            customdata = None

        fig.add_trace(go.Scatter(
            x=sub[dim1_col], y=sub[dim2_col],
            mode="markers",
            name=str(tabla),
            marker=dict(
                size=8,
                color=col,
                symbol="circle",
                line=dict(width=0.6, color="white"),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))

    # ── Vectores en negro ─────────────────────────────────────────────────────
    lmax  = float(np.sqrt((G**2).sum(axis=1)).max())
    all_r = max(abs(coord_df[dim1_col]).max(), abs(coord_df[dim2_col]).max())
    fac   = (all_r * 0.72) / lmax if lmax > 1e-10 else 1.0

    for j, vname in enumerate(var_names):
        gx = float(G[j, dim_x]) * fac
        gy = float(G[j, dim_y]) * fac
        fig.add_trace(go.Scatter(
            x=[0, gx], y=[0, gy], mode="lines",
            line=dict(color="black", width=2),
            showlegend=(j == 0),
            name="Variables" if j == 0 else vname,
            hoverinfo="skip", legendgroup="vars",
        ))
        fig.add_annotation(
            x=gx, y=gy, ax=0, ay=0, axref="x", ayref="y",
            arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor="black", showarrow=True, text="",
        )
        nombre = vname.replace("_score", "")
        r2 = np.hypot(gx, gy)
        nx2, ny2 = (gx/r2, gy/r2) if r2 > 1e-9 else (1.0, 0.0)
        fig.add_annotation(
            x=gx + nx2*abs(gx)*0.14,
            y=gy + ny2*abs(gy)*0.14,
            text=f"<b>{nombre}</b>", showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.80)",
        )

    lyt = _layout_base(
        title=f"{etiqueta_biplot} del Compromiso<br>"
              f"<sub>K={len(statis.tablas)} tablas · "
              f"Dim{dim_x+1} ({ve[dim_x]:.1f}%) × Dim{dim_y+1} ({ve[dim_y]:.1f}%)</sub>",
        xlab=xlab, ylab=ylab
    )
    lyt["yaxis"]["scaleanchor"] = "x"
    lyt["yaxis"]["scaleratio"]  = 1
    fig.update_layout(**lyt)
    return fig



def interactive_trajectories(statis, dim_x: int = 0, dim_y: int = 1) -> str:
    """
    Mapa de trayectorias interactivo — Canvas 2D (igual al modelo HJBiplot_STATIS_Dual_LATAM.html).
    Gráfico grande, proporcional al ancho del navegador, con scroll si es necesario.
    """
    import json as _json
    etiqueta_biplot = biplot_label(statis)
    cita_biplot     = biplot_cita(statis)

    ve       = statis.variance_explained
    d1_pct   = ve[dim_x]
    d2_pct   = ve[dim_y]
    coord_df = statis.coord_df
    dim1_col = f"Dim{dim_x+1}"
    dim2_col = f"Dim{dim_y+1}"
    G        = statis.F_vars
    var_names= statis.variable_names
    tablas   = sorted(statis.tablas)

    alpha_dict   = dict(zip(statis.tablas, statis.alpha_k))
    alpha_ranked = sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True)
    alpha_rank   = {t: r+1 for r, (t,_) in enumerate(alpha_ranked)}
    anios        = sorted(coord_df["fila"].unique())

    centroides = (coord_df.groupby("tabla")[[dim1_col, dim2_col]]
                  .mean()
                  .rename(columns={dim1_col:"C1", dim2_col:"C2"}))

    # Paleta viva por tabla
    PALETA_VIVA = [
        '#E63946','#1D3557','#2A9D8F','#E9C46A','#6A4C93',
        '#F4A261','#264653','#457B9D','#C1121F','#2D6A4F',
        '#8338EC','#FB5607','#3A86FF','#FFBE0B','#06D6A0',
        '#118AB2','#073B4C','#EF233C','#4CC9F0','#F72585',
        '#7209B7','#560BAD','#480CA8','#3A0CA3','#3F37C9',
    ]
    color_t = {t: PALETA_VIVA[i % len(PALETA_VIVA)] for i, t in enumerate(tablas)}

    # VS: escala de vectores — igual al modelo
    # Calcular VS para que los vectores sean proporcionales a los datos
    max_data = max(abs(coord_df[dim1_col]).max(), abs(coord_df[dim2_col]).max())
    max_load = float(np.sqrt((G**2).sum(axis=1)).max()) if G.size > 0 else 1.0
    VS = round(float(max_data * 0.72 / max_load), 4) if max_load > 1e-10 else 1.0

    # CTRY: {tabla: [{yr, d1, d2}, ...]}
    CTRY = {}
    for tabla in tablas:
        temp = (coord_df[coord_df["tabla"]==tabla]
                .sort_values("fila").reset_index(drop=True))
        CTRY[str(tabla)] = [
            {"yr": int(row["fila"]) if str(row["fila"]).isdigit() else str(row["fila"]),
             "d1": round(float(row[dim1_col]),4),
             "d2": round(float(row[dim2_col]),4)}
            for _, row in temp.iterrows()
        ]

    # LOAD: [{name, d1, d2}]
    LOAD = [{"name": vn.replace("_score",""),
             "d1": round(float(G[j, dim_x]),4),
             "d2": round(float(G[j, dim_y]),4)}
            for j, vn in enumerate(var_names)]

    # AKV, RKV, COL por tabla
    AKV = {str(t): f"{alpha_dict.get(t,0):.5f}" for t in tablas}
    RKV = {str(t): f"{alpha_rank.get(t,'—')}/{len(tablas)}" for t in tablas}
    COL = {str(t): color_t[t] for t in tablas}
    ANIOS_JS = [int(a) if str(a).isdigit() else str(a) for a in anios]

    ctry_js  = _json.dumps(CTRY,  ensure_ascii=False)
    load_js  = _json.dumps(LOAD,  ensure_ascii=False)
    akv_js   = _json.dumps(AKV,   ensure_ascii=False)
    rkv_js   = _json.dumps(RKV,   ensure_ascii=False)
    col_js   = _json.dumps(COL,   ensure_ascii=False)
    anios_js = _json.dumps(ANIOS_JS)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trayectorias STATIS Dual</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,sans-serif;background:#f4f4f2;color:#1a1a1a}}
.wrap{{max-width:1200px;margin:0 auto;padding:14px}}
h1{{font-size:14px;font-weight:700;text-align:center;margin-bottom:2px}}
.sub{{font-size:11px;color:#777;text-align:center;margin-bottom:10px}}
.topbar{{background:#fff;border:1px solid #ddd;border-radius:8px;
         padding:8px 12px;margin-bottom:8px;display:flex;gap:8px;
         flex-wrap:wrap;align-items:center}}
.ctrl{{font-size:11px;color:#555;font-weight:600;white-space:nowrap}}
.btns{{display:flex;gap:4px;flex-wrap:wrap;flex:1}}
.pb{{font-size:10px;padding:2px 8px;border:1px solid #ccc;border-radius:20px;
     cursor:pointer;background:#fff;color:#555;white-space:nowrap}}
.pb:hover{{background:#f0f0ee}}
.pb.on{{color:#fff;border-color:transparent}}
.main{{display:flex;gap:10px}}
.cvw{{flex:1;background:#fff;border:1px solid #ddd;border-radius:8px;
      padding:8px;min-width:0}}
canvas{{display:block;width:100%}}
.dimx{{font-size:10px;color:#888;text-align:center;margin-top:3px}}
.acts{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center}}
.ab{{font-size:11px;padding:4px 10px;border:1px solid #ccc;border-radius:6px;
     cursor:pointer;background:#fff;color:#333}}
.ab:hover{{background:#f0f0ee}}
.ab.blue{{background:#185FA5;color:#fff;border-color:#185FA5}}
.ab.red{{background:#E24B4A;color:#fff;border-color:#E24B4A}}
.yrbts{{display:flex;gap:3px;flex-wrap:wrap;align-items:center}}
.yrlbl{{font-size:10px;color:#555;font-weight:600}}
.yb{{font-size:10px;padding:2px 6px;border:1px solid #ccc;border-radius:12px;
     cursor:pointer;background:#fff;color:#555}}
.yb.on{{background:#333;color:#fff;border-color:#333}}
.note{{font-size:10px;color:#aaa;margin-top:6px;line-height:1.4}}
.side{{width:220px;flex-shrink:0;display:flex;flex-direction:column;gap:8px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}}
.ct{{font-size:10px;font-weight:700;color:#aaa;letter-spacing:.06em;
     text-transform:uppercase;margin-bottom:7px}}
.sn{{font-size:15px;font-weight:700;margin-bottom:3px}}
.kv{{display:flex;justify-content:space-between;font-size:11px;padding:2px 0}}
.kk{{color:#888}}.vv{{font-weight:700}}
.vr{{display:flex;justify-content:space-between;padding:2px 0;
     border-bottom:1px solid #f2f2f2;font-size:11px}}
</style>
</head>
<body>
<div class="wrap">
<h1>{etiqueta_biplot} STATIS Dual &mdash; Trayectorias longitudinales</h1>
<p class="sub">Lavit et al. (1994) &middot; {cita_biplot} &middot;
Dim1={round(float(d1_pct),2)}% &middot; Dim2={round(float(d2_pct),2)}%</p>

<div class="topbar">
  <span class="ctrl">Pa&iacute;s:</span>
  <div class="btns" id="pbg"></div>
</div>

<div class="main">
  <div class="cvw">
    <canvas id="bp" role="img"></canvas>
    <div class="dimx" id="dimx_lbl">Dim {dim_x+1} &mdash; ({round(float(d1_pct),2)}% varianza) &rarr;</div>
    <div class="acts">
      <button class="ab blue" onclick="doSel(null)">Todos</button>
      <button class="ab" id="vbtn" onclick="togV()">Vectores: ON</button>
      <button class="ab" id="ybtn" onclick="togY()">A&ntilde;os: ON</button>
      <div class="yrbts">
        <span class="yrlbl">A&ntilde;os:</span>
        <div id="yrbts"></div>
      </div>
    </div>
    <p class="note">Clic en pa&iacute;s (barra superior o punto) para destacar.
    Teclas &larr;&rarr; para navegar. Esc para limpiar.</p>
  </div>

  <div class="side">
    <div class="card">
      <div class="ct">Pa&iacute;s seleccionado</div>
      <div class="sn" id="sn">Todos</div>
      <div class="kv"><span class="kk">&alpha;k</span><span class="vv" id="sak">&mdash;</span></div>
      <div class="kv"><span class="kk">Rango</span><span class="vv" id="srk">&mdash;</span></div>
      <div class="kv"><span class="kk">Centroide Dim1</span><span class="vv" id="sd1">&mdash;</span></div>
      <div class="kv"><span class="kk">Centroide Dim2</span><span class="vv" id="sd2">&mdash;</span></div>
      <div class="kv"><span class="kk">Ediciones</span><span class="vv" id="sn2">&mdash;</span></div>
    </div>
    <div class="card">
      <div class="ct">Cargas vectoriales (Dim1 / Dim2)</div>
      <div id="vtbl"></div>
    </div>
  </div>
</div>
</div>

<script>
const LOAD={load_js};
const CTRY={ctry_js};
const AKV={akv_js};
const RKV={rkv_js};
const COL={col_js};
const VS={VS};
const YEARS={anios_js};
const D1_PCT="{round(float(d1_pct),2)}";
const D2_PCT="{round(float(d2_pct),2)}";

let curSel=null,showV=true,showY=true;
let actYrs=new Set(YEARS.map(String));

// Chips de países
const pbg=document.getElementById('pbg');
Object.keys(CTRY).forEach(name=>{{
  const b=document.createElement('button');
  b.className='pb';b.textContent=name;b.dataset.n=name;
  b.onclick=()=>doSel(curSel===name?null:name);
  pbg.appendChild(b);
}});

// Botones de años
const yrbts=document.getElementById('yrbts');
YEARS.forEach(a=>{{
  const b=document.createElement('button');
  b.className='yb on';b.textContent=a;b.dataset.y=String(a);
  b.onclick=()=>{{
    const k=String(a);
    actYrs.has(k)?actYrs.delete(k):actYrs.add(k);
    b.classList.toggle('on',actYrs.has(k));draw();
  }};
  yrbts.appendChild(b);
}});

// Tabla de cargas vectoriales
const vtbl=document.getElementById('vtbl');
const hdr=document.createElement('div');hdr.className='vr';
hdr.innerHTML='<span style="font-weight:700;width:36px">Ind.</span><span style="color:#185FA5;font-weight:700">Dim1</span><span style="color:#534AB7;font-weight:700">Dim2</span>';
vtbl.appendChild(hdr);
LOAD.forEach(l=>{{
  const r=document.createElement('div');r.className='vr';
  r.innerHTML='<span style="font-weight:700;width:36px">'+l.name+'</span><span style="color:#185FA5">'+(l.d1>=0?'+':'')+l.d1.toFixed(4)+'</span><span style="color:#534AB7">'+(l.d2>=0?'+':'')+l.d2.toFixed(4)+'</span>';
  vtbl.appendChild(r);
}});

// Canvas
const cv=document.getElementById('bp');
const ctx=cv.getContext('2d');
const PAD={{l:50,r:20,t:20,b:36}};

function allR(){{
  let xs=[],ys=[];
  Object.values(CTRY).forEach(a=>a.forEach(p=>{{xs.push(p.d1);ys.push(p.d2)}}));
  LOAD.forEach(l=>{{xs.push(l.d1*VS);ys.push(l.d2*VS)}});
  return{{xs,ys}};
}}
const {{xs,ys}}=allR();
const PF=0.7;
const xMn=Math.min(...xs)-PF,xMx=Math.max(...xs)+PF;
const yMn=Math.min(...ys)-PF,yMx=Math.max(...ys)+PF;

function gx(v){{return PAD.l+(v-xMn)/(xMx-xMn)*(cv.width-PAD.l-PAD.r)}}
function gy(v){{return cv.height-PAD.b-(v-yMn)/(yMx-yMn)*(cv.height-PAD.t-PAD.b)}}
function resize(){{
  const W=cv.parentElement.clientWidth-20,H=Math.round(W*0.70);
  cv.width=W;cv.height=H;cv.style.height=H+'px';draw();
}}

function draw(){{
  const W=cv.width,H=cv.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  // Grid
  ctx.strokeStyle='rgba(0,0,0,0.055)';ctx.lineWidth=0.5;
  for(let x=Math.ceil(xMn);x<=Math.floor(xMx);x++){{ctx.beginPath();ctx.moveTo(gx(x),PAD.t);ctx.lineTo(gx(x),H-PAD.b);ctx.stroke()}}
  for(let y=Math.ceil(yMn);y<=Math.floor(yMx);y++){{ctx.beginPath();ctx.moveTo(PAD.l,gy(y));ctx.lineTo(W-PAD.r,gy(y));ctx.stroke()}}
  // Ejes
  const ox=gx(0),oy=gy(0);
  ctx.strokeStyle='rgba(0,0,0,0.3)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(PAD.l,oy);ctx.lineTo(W-PAD.r,oy);ctx.stroke();
  ctx.beginPath();ctx.moveTo(ox,PAD.t);ctx.lineTo(ox,H-PAD.b);ctx.stroke();
  // Ticks
  ctx.fillStyle='#999';ctx.font='10px Arial';ctx.textAlign='center';ctx.textBaseline='top';
  for(let x=Math.ceil(xMn);x<=Math.floor(xMx);x++){{if(x===0)continue;ctx.fillText(x,gx(x),oy+3)}}
  ctx.textAlign='right';ctx.textBaseline='middle';
  for(let y=Math.ceil(yMn);y<=Math.floor(yMx);y++){{if(y===0)continue;ctx.fillText(y,ox-4,gy(y))}}
  // Etiqueta Dim2
  ctx.save();ctx.fillStyle='#666';ctx.font='bold 10px Arial';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.translate(13,H/2);ctx.rotate(-Math.PI/2);
  ctx.fillText('Dim {dim_y+1} ('+D2_PCT+'%)',0,0);ctx.restore();
  // Etiqueta Dim1
  ctx.fillStyle='#666';ctx.font='bold 10px Arial';ctx.textAlign='center';ctx.textBaseline='top';
  ctx.fillText('Dim {dim_x+1} ('+D1_PCT+'%)',W/2,H-PAD.b+18);

  // Vectores
  if(showV){{
    LOAD.forEach(l=>{{
      const vx=gx(l.d1*VS),vy=gy(l.d2*VS);
      ctx.strokeStyle='#111';ctx.lineWidth=1.8;ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(vx,vy);ctx.stroke();
      const ang=Math.atan2(vy-oy,vx-ox),al=9;
      ctx.fillStyle='#111';ctx.beginPath();ctx.moveTo(vx,vy);
      ctx.lineTo(vx-al*Math.cos(ang-0.38),vy-al*Math.sin(ang-0.38));
      ctx.lineTo(vx-al*Math.cos(ang+0.38),vy-al*Math.sin(ang+0.38));
      ctx.closePath();ctx.fill();
      const lx=gx(l.d1*VS*1.08),ly=gy(l.d2*VS*1.08);
      ctx.fillStyle='#111';ctx.font='bold 12px Arial';
      ctx.textAlign=l.d1>=0?'left':'right';ctx.textBaseline='middle';
      ctx.fillText(l.name,lx+(l.d1>=0?3:-3),ly);
    }});
  }}

  // Trayectorias
  Object.keys(CTRY).forEach(name=>{{
    const pts=CTRY[name].filter(p=>actYrs.has(String(p.yr)));
    if(!pts.length)return;
    const col=COL[name]||'#888';
    const isSel=curSel===name,anyS=curSel!==null,dim=anyS&&!isSel;
    ctx.globalAlpha=dim?0.08:1;
    if(pts.length>1){{
      ctx.strokeStyle=col;ctx.lineWidth=isSel?2.8:1.1;
      ctx.setLineDash(isSel?[]:[3,2.5]);
      ctx.beginPath();pts.forEach((p,i)=>{{i===0?ctx.moveTo(gx(p.d1),gy(p.d2)):ctx.lineTo(gx(p.d1),gy(p.d2))}});ctx.stroke();
      ctx.setLineDash([]);
      if(isSel){{
        const lp=pts[pts.length-1],pp=pts[pts.length-2];
        const ax=gx(lp.d1),ay=gy(lp.d2),bx=gx(pp.d1),by=gy(pp.d2);
        const ang=Math.atan2(ay-by,ax-bx),al=10;
        ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(ax,ay);
        ctx.lineTo(ax-al*Math.cos(ang-0.38),ay-al*Math.sin(ang-0.38));
        ctx.lineTo(ax-al*Math.cos(ang+0.38),ay-al*Math.sin(ang+0.38));
        ctx.closePath();ctx.fill();
      }}
    }}
    pts.forEach(p=>{{
      const ppx=gx(p.d1),ppy=gy(p.d2),r=isSel?5.5:3.5;
      ctx.beginPath();ctx.arc(ppx,ppy,r,0,Math.PI*2);ctx.fillStyle=col;ctx.fill();
      if(isSel){{ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke()}}
      if(showY&&isSel){{
        ctx.fillStyle='#111';ctx.font='bold 10px Arial';
        ctx.textAlign='left';ctx.textBaseline='bottom';
        ctx.fillText(p.yr,ppx+7,ppy-1);
      }}
    }});
    if(!dim){{
      const ap=pts;
      const cd1=ap.reduce((a,p)=>a+p.d1,0)/ap.length;
      const cd2=ap.reduce((a,p)=>a+p.d2,0)/ap.length;
      ctx.font=(isSel?'bold 13px':'10px')+' Arial';
      ctx.fillStyle=col;ctx.textBaseline='middle';
      ctx.textAlign=cd1>=0?'left':'right';
      ctx.fillText(name,cd1>=0?gx(cd1)+6:gx(cd1)-6,gy(cd2));
    }}
    ctx.globalAlpha=1;
  }});
  ctx.textBaseline='alphabetic';
}}

function doSel(name){{
  curSel=name;
  document.querySelectorAll('.pb').forEach(b=>{{
    if(b.dataset.n===curSel){{b.classList.add('on');b.style.background=COL[b.dataset.n]||'#888'}}
    else{{b.classList.remove('on');b.style.background=''}}
  }});
  if(!curSel){{
    document.getElementById('sn').textContent='Todos';
    ['sak','srk','sd1','sd2','sn2'].forEach(id=>document.getElementById(id).textContent='—');
  }}else{{
    const pts=CTRY[curSel];
    const cd1=(pts.reduce((a,p)=>a+p.d1,0)/pts.length).toFixed(3);
    const cd2=(pts.reduce((a,p)=>a+p.d2,0)/pts.length).toFixed(3);
    document.getElementById('sn').textContent=curSel;
    document.getElementById('sn').style.color=COL[curSel]||'#1a1a1a';
    document.getElementById('sak').textContent=AKV[curSel]||'—';
    document.getElementById('srk').textContent=RKV[curSel]||'—';
    document.getElementById('sd1').textContent=cd1;
    document.getElementById('sd2').textContent=cd2;
    document.getElementById('sn2').textContent=pts.length;
  }}
  draw();
}}

function togV(){{showV=!showV;document.getElementById('vbtn').textContent='Vectores: '+(showV?'ON':'OFF');draw()}}
function togY(){{showY=!showY;document.getElementById('ybtn').textContent='Años: '+(showY?'ON':'OFF');draw()}}

cv.addEventListener('click',e=>{{
  const rect=cv.getBoundingClientRect(),scX=cv.width/rect.width,scY=cv.height/rect.height;
  const mx=(e.clientX-rect.left)*scX,my=(e.clientY-rect.top)*scY;
  let best=null,bestD=1e9;
  Object.keys(CTRY).forEach(name=>{{
    CTRY[name].filter(p=>actYrs.has(String(p.yr))).forEach(p=>{{
      const dx=gx(p.d1)-mx,dy=gy(p.d2)-my,d=dx*dx+dy*dy;
      if(d<bestD){{bestD=d;best=name}}
    }});
  }});
  if(Math.sqrt(bestD)<22)doSel(curSel===best?null:best);
}});

const cnames=Object.keys(CTRY);
document.addEventListener('keydown',e=>{{
  if(e.key==='Escape'){{doSel(null);return}}
  if(!curSel){{if(e.key==='ArrowRight')doSel(cnames[0]);return}}
  const i=cnames.indexOf(curSel);
  if(e.key==='ArrowRight')doSel(cnames[(i+1)%cnames.length]);
  if(e.key==='ArrowLeft')doSel(cnames[(i-1+cnames.length)%cnames.length]);
}});

resize();
window.addEventListener('resize',()=>{{clearTimeout(window._rt);window._rt=setTimeout(resize,150)}});
</script>
</body>
</html>"""
    return html


def interactive_single_trajectory(statis: StatisDualResult,
                                   entity_name: str,
                                   dim_x: int = 0,
                                   dim_y: int = 1) -> go.Figure:
    """
    Trayectoria detallada de una sola entidad con etiquetas de tabla.
    """
    if entity_name not in statis.row_names:
        raise ValueError(f"Entidad '{entity_name}' no encontrada.")

    i = statis.row_names.index(entity_name)
    tablas = statis.tablas
    ve = statis.variance_explained

    xs = [statis.F_k[b][i, dim_x] for b in tablas]
    ys = [statis.F_k[b][i, dim_y] for b in tablas]

    # Color por tiempo (frío → cálido)
    cmap = [f"rgb({int(255*(1-t))},{int(100+100*t)},{int(255*t)})"
            for t in np.linspace(0, 1, len(tablas))]

    fig = go.Figure()

    # Segmentos
    for t in range(len(tablas) - 1):
        fig.add_trace(go.Scatter(
            x=[xs[t], xs[t+1]], y=[ys[t], ys[t+1]],
            mode="lines",
            line=dict(color=cmap[t], width=3),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Puntos con etiquetas de tabla
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=[str(b) for b in tablas],
        textposition="top right",
        textfont=dict(size=11, color=COLOR_PRIMARY),
        marker=dict(size=12, color=list(range(len(tablas))),
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Tiempo", tickvals=list(range(len(tablas))),
                                  ticktext=[str(b) for b in tablas])),
        name=entity_name,
        hovertemplate=(
            f"<b>{entity_name}</b><br>"
            "Tabla: %{text}<br>"
            f"Dim{dim_x+1}: %{{x:.4f}}<br>"
            f"Dim{dim_y+1}: %{{y:.4f}}<extra></extra>"
        ),
    ))

    # Flecha dirección
    if len(xs) >= 2:
        fig.add_annotation(
            x=xs[-1], y=ys[-1],
            ax=xs[-2], ay=ys[-2],
            axref="x", ayref="y",
            arrowhead=4, arrowsize=1.5, arrowwidth=3,
            arrowcolor=COLOR_SECONDARY,
            showarrow=True, text="",
        )

    fig.update_layout(**_layout_base(
        title=f"Trayectoria — <b>{entity_name}</b><br>"
              f"<sub>Dim{dim_x+1} ({ve[dim_x]:.1f}%) × Dim{dim_y+1} ({ve[dim_y]:.1f}%)</sub>",
        xlab=f"Dim{dim_x+1} ({ve[dim_x]:.1f}%)",
        ylab=f"Dim{dim_y+1} ({ve[dim_y]:.1f}%)",
    ))

    return fig


# ─── Exportación a HTML ───────────────────────────────────────────────────────

def fig_to_html(fig: go.Figure, offline: bool = True) -> str:
    """
    Convierte figura Plotly a HTML.
    offline=True incluye el JS de Plotly (~3MB) en el HTML — sin internet.
    offline=False usa CDN (más liviano pero requiere conexión).
    """
    include_plotlyjs = True if offline else "cdn"
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "statis_dual_graph",
                "height": 700,
                "width": 1000,
                "scale": 2,
            },
        }
    )


def save_html(fig: go.Figure, filepath: str, offline: bool = True):
    """Guarda la figura como HTML en disco."""
    html = fig_to_html(fig, offline=offline)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
