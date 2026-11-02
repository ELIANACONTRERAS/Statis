@echo off
setlocal EnableExtensions
REM ============================================================
REM  NekaStat v3.8 - Instalacion en Windows (doble clic)
REM  - Si no hay Python 3.10+, lo descarga e instala (Python 3.12.10, solo para este usuario)
REM  - Prepara NekaStat y crea el acceso directo en el ESCRITORIO
REM ============================================================
cd /d "%~dp0"
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYLOCAL=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

echo [1/4] Buscando Python...
call :buscar_python
if defined PY goto :tengo_python

echo       No se encontro Python 3.10 o superior. Instalando Python 3.12.10...
where winget >nul 2>&1
if %errorlevel%==0 (
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    call :buscar_python
)
if defined PY goto :tengo_python

echo       Descargando Python desde python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -Uri '%PYURL%' -OutFile '%TEMP%\NekaStat_python.exe' -UseBasicParsing"
if not exist "%TEMP%\NekaStat_python.exe" goto :sinpython
start /wait "" "%TEMP%\NekaStat_python.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
del "%TEMP%\NekaStat_python.exe" >nul 2>&1
call :buscar_python
if not defined PY goto :sinpython

:tengo_python
echo       Python: %PY%
echo [2/4] Creando entorno de NekaStat...
"%PY%" -m venv .venv || goto :error
echo [3/4] Instalando librerias (puede tardar varios minutos la primera vez)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo [4/4] Creando acceso directo en el escritorio...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalador\windows\acceso_directo.ps1" || goto :error
echo.
echo LISTO. Abra NekaStat con el icono del escritorio.
pause
exit /b 0

:buscar_python
set "PY="
if exist "%PYLOCAL%" (
    "%PYLOCAL%" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PY=%PYLOCAL%" && exit /b 0
)
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable if sys.version_info>=(3,10) else '')" 2^>nul') do set "PY=%%i"
if defined PY exit /b 0
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable if sys.version_info>=(3,10) else '')" 2^>nul') do set "PY=%%i"
exit /b 0

:sinpython
echo.
echo No fue posible instalar Python automaticamente (revise su conexion a internet).
echo Instalelo manualmente desde python.org marcando "Add python.exe to PATH"
echo y vuelva a ejecutar este archivo.
pause
exit /b 1

:error
echo.
echo Ocurrio un error. Revise los mensajes de arriba.
pause
exit /b 1
