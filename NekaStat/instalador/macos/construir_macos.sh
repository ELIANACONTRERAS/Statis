#!/bin/bash
# ============================================================
#  NekaStat v3.8 - Construye NekaStat.app, el instalador .pkg y la imagen .dmg
#  Requisitos: macOS con Python 3.10+ (python.org o Homebrew)
#  Uso:  bash instalador/macos/construir_macos.sh   (desde la carpeta raíz de NekaStat)
# ============================================================
set -e
cd "$(dirname "$0")/../.."
echo "[1/5] Entorno virtual..."
python3 -m venv .venv_build
source .venv_build/bin/activate
python -m pip install --upgrade pip
echo "[2/5] Dependencias..."
pip install -r requirements.txt pyinstaller
echo "[3/5] Compilando NekaStat.app..."
pyinstaller instalador/NekaStat.spec --noconfirm
mkdir -p instalador/macos/salida
echo "[4/5] Instalador .pkg (instala en Aplicaciones y crea acceso directo en el Escritorio)..."
chmod +x instalador/macos/scripts/postinstall
pkgbuild --component dist/NekaStat.app \
         --install-location /Applications \
         --scripts instalador/macos/scripts \
         --identifier ec.nekastat.app --version 3.8 \
         instalador/macos/salida/NekaStat_v3.8_Instalador_Mac.pkg
echo "[5/5] Imagen .dmg (arrastrar a Aplicaciones)..."
rm -rf dmg_tmp && mkdir dmg_tmp
cp -R dist/NekaStat.app dmg_tmp/
ln -s /Applications dmg_tmp/Aplicaciones
hdiutil create -volname "NekaStat 3.8" -srcfolder dmg_tmp -ov -format UDZO \
        instalador/macos/salida/NekaStat_v3.8_Mac.dmg
rm -rf dmg_tmp
echo "LISTO: instalador/macos/salida/"
