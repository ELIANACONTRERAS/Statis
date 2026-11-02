# Crea el acceso directo de NekaStat en el escritorio (incluye escritorios en OneDrive)
$raiz   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pyw    = Join-Path $raiz '.venv\Scripts\pythonw.exe'
$main   = Join-Path $raiz 'main.py'
$icono  = Join-Path $raiz 'assets\NekaStat.ico'
$destino = Join-Path ([Environment]::GetFolderPath('Desktop')) 'NekaStat.lnk'
$ws = New-Object -ComObject WScript.Shell
$s  = $ws.CreateShortcut($destino)
$s.TargetPath       = $pyw
$s.Arguments        = '"' + $main + '"'
$s.WorkingDirectory = $raiz
$s.IconLocation     = $icono
$s.Description      = 'NekaStat v3.8 - N-way Exploratory K-table Analysis'
$s.Save()
Write-Host "Acceso directo creado en: $destino"
