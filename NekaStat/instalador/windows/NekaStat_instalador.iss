; Instalador de NekaStat v3.8 para Windows (Inno Setup 6)
; Instala la app y crea el acceso directo en el ESCRITORIO y en el menú Inicio.
#define AppName "NekaStat"
#define AppVersion "3.8"
#define AppPublisher "Eliana Contreras Jordán"

[Setup]
AppId={{80A87B72-FD13-4BA1-BA49-1857E2C2ECB5}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=salida
OutputBaseFilename=NekaStat_v3.8_Instalador_Windows
SetupIconFile=..\..\assets\NekaStat.ico
UninstallDisplayIcon={app}\NekaStat.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
; Marcada por defecto: el acceso directo se crea en el escritorio salvo que el usuario la desmarque
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

[Files]
Source: "..\..\dist\NekaStat\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\NekaStat.exe"
Name: "{group}\Desinstalar {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\NekaStat.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NekaStat.exe"; Description: "Abrir NekaStat ahora"; Flags: nowait postinstall skipifsilent
