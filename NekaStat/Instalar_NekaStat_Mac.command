#!/bin/bash
# ============================================================
#  NekaStat v3.8 - Instalación en macOS
#  - Si no hay Python 3.10+, lo descarga e instala (Python 3.12.10 de python.org;
#    macOS pedirá la contraseña del equipo)
#  - Prepara NekaStat y crea el acceso directo en el ESCRITORIO
# ============================================================
cd "$(dirname "$0")" || exit 1
RAIZ="$(pwd)"
PYURL="https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"

version_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' >/dev/null 2>&1; }

buscar_python() {
  PY=""
  for v in 3.14 3.13 3.12 3.11 3.10; do
    c="/Library/Frameworks/Python.framework/Versions/$v/bin/python3"
    [ -x "$c" ] && version_ok "$c" && { PY="$c"; return; }
  done
  for c in /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    [ -x "$c" ] && version_ok "$c" && { PY="$c"; return; }
  done
  # /usr/bin/python3 solo si están las herramientas de Apple (si no, abre un aviso de instalación)
  if xcode-select -p >/dev/null 2>&1 && [ -x /usr/bin/python3 ] && version_ok /usr/bin/python3; then PY=/usr/bin/python3; fi
}

echo "[1/4] Buscando Python..."
buscar_python
if [ -z "$PY" ]; then
  echo "      No se encontró Python 3.10 o superior. Descargando Python 3.12.10 de python.org..."
  curl -fL -o /tmp/NekaStat_python.pkg "$PYURL" || { echo "No se pudo descargar Python. Revise su conexión."; read -r; exit 1; }
  echo "      Instalando Python (macOS le pedirá su contraseña)..."
  osascript -e 'do shell script "installer -pkg /tmp/NekaStat_python.pkg -target /" with administrator privileges' \
    || { echo "No se pudo instalar Python."; read -r; exit 1; }
  rm -f /tmp/NekaStat_python.pkg
  buscar_python
  [ -z "$PY" ] && { echo "Python no quedó disponible. Instálelo desde python.org y vuelva a ejecutar."; read -r; exit 1; }
fi
echo "      Python: $PY"
echo "[2/4] Creando entorno de NekaStat..."
"$PY" -m venv .venv || { echo "Error al crear el entorno."; read -r; exit 1; }
echo "[3/4] Instalando librerías (puede tardar varios minutos la primera vez)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt || { echo "Error al instalar librerías."; read -r; exit 1; }
echo "[4/4] Creando acceso directo en el Escritorio..."
LANZADOR="$HOME/Desktop/NekaStat.command"
cat > "$LANZADOR" <<EOF2
#!/bin/bash
cd "$RAIZ"
"$RAIZ/.venv/bin/python" main.py
EOF2
chmod +x "$LANZADOR"
echo "LISTO. Abra NekaStat con el ícono 'NekaStat' del Escritorio."
read -r -p "Pulse Enter para cerrar..."
