"""
M0_config.py — Configuración global STATIS Dual
================================================
Parámetros editables sin tocar el resto del código.
Todos los módulos importan desde aquí.
"""

# ─── Versión del programa ────────────────────────────────────────────────────
APP_NAME    = "STATIS"
APP_VERSION = "1.0.0"
APP_AUTHOR  = "Eliana Contreras"

# ─── Criterios mínimos STATIS Dual (Lavit et al. 1994) ──────────────────────
MIN_TABLAS   = 3      # K mínimo de tablas (tablas)
MIN_FILAS     = 3      # mínimo de filas por tabla
MIN_VARIABLES = 2      # mínimo de variables numéricas

# ─── Solver numérico manual ──────────────────────────────────────────────────
QR_MAX_ITER   = 10_000   # iteraciones máximas algoritmo QR
QR_TOLERANCE  = 1e-12    # umbral de convergencia
SVD_MAX_ITER  = 10_000
SVD_TOLERANCE = 1e-12

# ─── Dimensiones a retener por defecto ──────────────────────────────────────
N_DIMS_DEFAULT = 2        # el usuario puede cambiar esto en la UI

# ─── Escalado del biplot ─────────────────────────────────────────────────────
VECTOR_SCALE  = 0.55      # 55% del espacio para vectores de variables

# ─── Paleta visual del programa ──────────────────────────────────────────────
# Navy / Gold / Teal — coherente con tus figuras validadas
COLOR_PRIMARY    = "#0D2545"   # navy oscuro
COLOR_SECONDARY  = "#C9A84C"   # gold
COLOR_ACCENT     = "#1A7A8A"   # teal
COLOR_BG         = "#F5F7FA"   # fondo gris muy claro
COLOR_SURFACE    = "#FFFFFF"
COLOR_TEXT       = "#1C2B3A"
COLOR_TEXT_LIGHT = "#6B7C93"
COLOR_ERROR      = "#C0392B"
COLOR_SUCCESS    = "#1E8449"
COLOR_WARNING    = "#D4AC0D"

# Paleta para trayectorias (hasta 30 entidades)
TRAJECTORY_PALETTE = [
    "#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6",
    "#1ABC9C","#E67E22","#34495E","#C0392B","#2980B9",
    "#27AE60","#F1C40F","#8E44AD","#16A085","#D35400",
    "#BDC3C7","#7F8C8D","#6C3483","#1A5276","#0E6655",
    "#784212","#4A235A","#1B2631","#7D6608","#0B5345",
    "#4D5656","#2C3E50","#A93226","#117A65","#1F618D",
]

# ─── Paleta semántica del script validado (notebook 2__STATIS_DUAL_...) ─────
# Idéntica a la usada en las figuras SEN_*.png y en el análisis de sensibilidad.
# Uso: colorear país/trayectorias/año en gráficos analíticos (m6_outputs_static,
# m7_outputs_interactive). NO reemplaza COLOR_PRIMARY/SECONDARY/ACCENT, que
# son exclusivos del look-and-feel de la interfaz PyQt6 (QSS_GLOBAL).
COLOR_ECUADOR   = "#E63946"   # rojo — siempre Ecuador
COLOR_CERCANOS  = "#F4A261"   # naranja — países más cercanos a Ecuador
COLOR_LATAM     = "#4A90D9"   # azul — resto de LATAM
COLOR_VARS      = "#2C6E49"   # verde — vectores de variables/indicadores QS
COLOR_GRAY_SEN  = "#CCCCCC"   # gris — referencia neutra (líneas 0,0 etc.)

# 7 colores por año (2020–2026), idéntico al script
COLORES_AÑOS = [
    "#1D3557","#457B9D","#2A9D8F",
    "#E9C46A","#F4A261","#E76F51","#6A4C93",
]

# Colores por criterio de limpieza (Panel A/B/C), idéntico al script
COLORES_CRITERIO = {"A": "#2B6CB0", "B": "#C1121F", "C": "#2D6A4F"}

# ─── Paleta de magnitud (terciles) ───────────────────────────────────────────
# Para gráficos de ranking/pesos donde el color debe indicar el NIVEL del
# valor (alto/medio/bajo), no la identidad ni cercanía geográfica del país.
# Ecuador se resalta aparte en COLOR_ECUADOR, sin importar su tercil.
COLOR_TIER_ALTO  = "#1D3557"   # azul oscuro — tercio superior
COLOR_TIER_MEDIO = "#4A90D9"   # azul medio — tercio intermedio
COLOR_TIER_BAJO  = "#A9CCE3"   # azul claro — tercio inferior

# ─── Paleta para clúster (K-means/Ward) ──────────────────────────────────────
# El clúster agrupa por grupo estadístico (1,2,3...), no por país, así que no
# hay mapeo semántico directo con Ecuador/cercanos/LATAM. Por decisión del
# usuario: se reutilizan 4 colores de la paleta del script (azul/naranja/
# rojo/verde) y, si se necesitan más de 4 clústeres, se completa con
# COLORES_AÑOS para mantener la misma familia visual.
CLUSTER_PALETTE = [
    COLOR_LATAM, COLOR_CERCANOS, COLOR_ECUADOR, COLOR_VARS,
] + COLORES_AÑOS

# ─── Exportación de gráficos ─────────────────────────────────────────────────
EXPORT_DPI        = 600
EXPORT_FORMAT_PNG = "png"
EXPORT_FORMAT_TIF = "tiff"
FIG_WIDTH_SINGLE  = 7.0    # pulgadas, columna doble publicación
FIG_WIDTH_DOUBLE  = 14.0

# ─── Fuentes (con fallback automático) ──────────────────────────────────────
FONT_PRIORITY = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]

# ─── Estilo PyQt6 — stylesheet global ───────────────────────────────────────
QSS_GLOBAL = f"""
    QMainWindow, QDialog, QWidget {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
        font-family: Arial, sans-serif;
        font-size: 13px;
    }}
    QGroupBox {{
        border: 1.5px solid #CBD5E0;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 8px;
        font-weight: 600;
        color: {COLOR_PRIMARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        left: 12px;
    }}
    QPushButton {{
        background-color: {COLOR_PRIMARY};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: #1A3A6B;
    }}
    QPushButton:pressed {{
        background-color: {COLOR_ACCENT};
    }}
    QPushButton:disabled {{
        background-color: #A0AEC0;
        color: #E2E8F0;
    }}
    QPushButton#btn_secondary {{
        background-color: transparent;
        color: {COLOR_PRIMARY};
        border: 1.5px solid {COLOR_PRIMARY};
    }}
    QPushButton#btn_secondary:hover {{
        background-color: {COLOR_PRIMARY};
        color: white;
    }}
    QPushButton#btn_danger {{
        background-color: {COLOR_ERROR};
    }}
    QLineEdit, QComboBox, QSpinBox {{
        border: 1.5px solid #CBD5E0;
        border-radius: 5px;
        padding: 6px 10px;
        background: white;
        color: {COLOR_TEXT};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border-color: {COLOR_ACCENT};
    }}
    QTableWidget {{
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        gridline-color: #EDF2F7;
        background: white;
        alternate-background-color: #F7FAFC;
    }}
    QTableWidget::item:selected {{
        background-color: {COLOR_ACCENT};
        color: white;
    }}
    QHeaderView::section {{
        background-color: {COLOR_PRIMARY};
        color: white;
        padding: 6px;
        font-weight: 600;
        border: none;
    }}
    QCheckBox {{
        spacing: 8px;
        color: {COLOR_TEXT};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1.5px solid #A0AEC0;
        border-radius: 3px;
        background: white;
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLOR_ACCENT};
        border-color: {COLOR_ACCENT};
        image: none;
    }}
    QLabel#label_title {{
        color: {COLOR_PRIMARY};
        font-size: 22px;
        font-weight: 700;
    }}
    QLabel#label_subtitle {{
        color: {COLOR_TEXT_LIGHT};
        font-size: 13px;
    }}
    QLabel#label_section {{
        color: {COLOR_PRIMARY};
        font-size: 14px;
        font-weight: 600;
    }}
    QLabel#label_error {{
        color: {COLOR_ERROR};
        font-size: 12px;
    }}
    QLabel#label_ok {{
        color: {COLOR_SUCCESS};
        font-size: 12px;
    }}
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: #E2E8F0;
        height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {COLOR_ACCENT};
        border-radius: 4px;
    }}
    QScrollBar:vertical {{
        border: none;
        background: #F7FAFC;
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: #CBD5E0;
        border-radius: 4px;
        min-height: 20px;
    }}
    QTabWidget::pane {{
        border: 1px solid #CBD5E0;
        border-radius: 6px;
        background: white;
    }}
    QTabBar::tab {{
        background: #EDF2F7;
        color: {COLOR_TEXT_LIGHT};
        padding: 8px 18px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        margin-right: 2px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        background: {COLOR_PRIMARY};
        color: white;
        font-weight: 600;
    }}
    QSplitter::handle {{
        background: #E2E8F0;
        width: 2px;
    }}
    QStatusBar {{
        background: {COLOR_PRIMARY};
        color: white;
        font-size: 12px;
    }}
"""
