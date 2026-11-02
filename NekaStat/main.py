"""
main.py — Punto de entrada de NekaStat (STATIS Dual)
====================================================
Ejecutar:
    python main.py
"""

import sys
import os
import traceback
import subprocess
import importlib

# Agregar la carpeta raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Auto-instalación de dependencias faltantes ─────────────────────────────
# Antes de abrir la app, revisa que todas las librerías necesarias estén
# instaladas. Si falta alguna (por ejemplo al agregar una nueva más adelante,
# como adjustText), la instala sola con pip en vez de que el usuario tenga
# que abrir una terminal y hacerlo a mano.
REQUIRED_PACKAGES = {
    # nombre_para_import: nombre_del_paquete_en_pip
    "PyQt6":       "PyQt6",
    "numpy":       "numpy",
    "pandas":      "pandas",
    "matplotlib":  "matplotlib",
    "seaborn":     "seaborn",
    "openpyxl":    "openpyxl",
    "adjustText":  "adjustText",
    "plotly":      "plotly",
}

def _ensure_dependencies():
    # Si la app corre como ejecutable compilado (PyInstaller la marca con
    # sys.frozen=True), todas las dependencias ya están empaquetadas dentro
    # del .exe — no hay un Python real ni pip accesible ahí, así que intentar
    # instalar algo en ese contexto solo produce errores o cuelgues.
    if getattr(sys, "frozen", False):
        return

    faltantes = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            faltantes.append(pip_name)

    if not faltantes:
        return

    print(f"Instalando dependencias faltantes: {', '.join(faltantes)} ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *faltantes]
        )
        print("Dependencias instaladas correctamente.")
    except Exception as e:
        print(f"No se pudieron instalar automáticamente: {e}")
        print("Instálalas manualmente con:")
        print(f"    pip install {' '.join(faltantes)}")
        input("Presiona Enter para salir...")
        sys.exit(1)

_ensure_dependencies()


def handle_exception(exc_type, exc_value, exc_tb):
    """Captura cualquier excepción no manejada y la muestra en un diálogo."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    
    # Intentar mostrar en QMessageBox si Qt está disponible
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setWindowTitle("Error crítico — STATIS Dual")
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("El programa encontró un error inesperado:")
        msg.setDetailedText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    except Exception:
        # Si Qt también falla, escribir a archivo
        # En el ejecutable instalado la carpeta del programa puede no tener
        # permiso de escritura: el registro va a la carpeta personal.
        if getattr(sys, "frozen", False):
            log_path = os.path.join(os.path.expanduser("~"), "NekaStat_error_log.txt")
        else:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(error_msg)
        print(f"Error guardado en: {log_path}")
    
    sys.exit(1)

# Instalar el manejador global ANTES de importar nada más
sys.excepthook = handle_exception

from ui_main import run

if __name__ == "__main__":
    run()
