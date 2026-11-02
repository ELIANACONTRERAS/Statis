# -*- mode: python ; coding: utf-8 -*-
# Configuración de PyInstaller para NekaStat (Windows y macOS).
# Uso, desde la carpeta raíz de NekaStat:
#     pyinstaller instalador/NekaStat.spec --noconfirm
import sys, os
RAIZ = os.path.abspath(os.path.join(SPECPATH, '..'))
ICONO = os.path.join(RAIZ, 'assets', 'NekaStat.icns' if sys.platform == 'darwin' else 'NekaStat.ico')

a = Analysis(
    [os.path.join(RAIZ, 'main.py')],
    pathex=[RAIZ],
    datas=[(os.path.join(RAIZ, 'assets'), 'assets')],
    hiddenimports=['core', 'ui_main', 'openpyxl', 'seaborn', 'adjustText',
                   'plotly', 'matplotlib.backends.backend_qtagg'],
    excludes=['tkinter', 'gi'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='NekaStat',
          console=False, icon=ICONO)
coll = COLLECT(exe, a.binaries, a.datas, name='NekaStat')

if sys.platform == 'darwin':
    app = BUNDLE(coll, name='NekaStat.app', icon=ICONO,
                 bundle_identifier='ec.nekastat.app',
                 info_plist={'CFBundleShortVersionString': '3.8',
                             'CFBundleVersion': '3.8',
                             'NSHighResolutionCapable': True,
                             'CFBundleDisplayName': 'NekaStat'})
