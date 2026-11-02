@echo off
REM ============================================================
REM  NekaStat v3.8 - Construye el instalador para Windows
REM  Requisitos: Python 3.10+ e Inno Setup 6 (jrsoftware.org/isinfo.php)
REM  Ejecutar con doble clic desde la carpeta instalador\windows
REM ============================================================
cd /d "%~dp0\..\.."
echo [1/4] Creando entorno virtual...
python -m venv .venv_build || goto :error
call .venv_build\Scripts\activate.bat
echo [2/4] Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller || goto :error
echo [3/4] Compilando NekaStat.exe...
pyinstaller instalador\NekaStat.spec --noconfirm || goto :error
echo [4/4] Creando el instalador...
set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
%ISCC% instalador\windows\NekaStat_instalador.iss || goto :error
echo.
echo LISTO: instalador en  instalador\windows\salida\NekaStat_v3.8_Instalador_Windows.exe
pause
exit /b 0
:error
echo Ocurrio un error. Revise los mensajes de arriba.
pause
exit /b 1
