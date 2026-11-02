# NekaStat v3.8 — Instalación

## Instalación rápida (recomendada): acceso directo en el escritorio
**No hace falta tener Python:** si la computadora no lo tiene (o tiene una versión anterior a 3.10),
el instalador descarga e instala Python 3.12.10 de python.org automáticamente.
En Mac, el sistema pedirá la contraseña del equipo para instalar Python. Se necesita internet.

**Windows:** descomprima la carpeta `NekaStat` (por ejemplo en Documentos) y haga doble clic en
`Instalar_NekaStat_Windows.bat`.

**Mac:** descomprima la carpeta `NekaStat` (por ejemplo en Documentos), clic derecho sobre
`Instalar_NekaStat_Mac.command` → **Abrir** → **Abrir**.

Tarda unos minutos la primera vez (descarga las librerías). Al terminar aparece el ícono
**NekaStat** en el escritorio. No mueva ni borre la carpeta `NekaStat` después de instalar:
el acceso directo apunta a ella.

---

# Instaladores compilados (sin necesidad de Python)

Al instalar NekaStat con estos instaladores queda un **acceso directo en el escritorio**
y no hace falta tener Python.

## Opción 1 (recomendada): construir los instaladores en GitHub, sin instalar nada
(En el repositorio Statis, el flujo es `.github/workflows/nekastat_instaladores.yml`, en la raíz.)
1. Cree un repositorio en GitHub y suba **toda** esta carpeta (incluida `.github/`).
2. En el repositorio, abra **Actions → "NekaStat - construir instaladores" → Run workflow**.
3. Al terminar (unos 15 minutos), descargue en **Artifacts**:
   - `NekaStat-Windows` → `NekaStat_v3.8_Instalador_Windows.exe`
   - `NekaStat-Mac-AppleSilicon` → `.pkg` y `.dmg` para Mac con chip M1, M2, M3 o M4
   - `NekaStat-Mac-Intel` → `.pkg` y `.dmg` para Mac con procesador Intel

## Opción 2: construir en cada computadora
### Windows
1. Instale Python 3.10+ (python.org; marque "Add Python to PATH") e Inno Setup 6 (jrsoftware.org).
2. Doble clic en `instalador\windows\construir_windows.bat`.
3. Resultado: `instalador\windows\salida\NekaStat_v3.8_Instalador_Windows.exe`.

### macOS
1. Instale Python 3.10+ (python.org).
2. En Terminal, desde esta carpeta: `bash instalador/macos/construir_macos.sh`
3. Resultado en `instalador/macos/salida/`: `.pkg` (instalador) y `.dmg`.

## Qué hace cada instalador
| | Windows (.exe) | macOS (.pkg) | macOS (.dmg) |
|---|---|---|---|
| Dónde instala | Archivos de programa del usuario | Aplicaciones | El usuario arrastra a Aplicaciones |
| Acceso directo en el escritorio | Sí (casilla marcada por defecto) | Sí, automático | No (formato habitual de Mac) |
| Menú Inicio / Launchpad | Sí | Sí | Sí |
| Desinstalar | Panel de control → Aplicaciones | Arrastrar NekaStat de Aplicaciones a la papelera | Igual |

## Primera apertura (aplicación sin firma digital)
- **Windows:** puede aparecer "Windows protegió su PC". Pulse **Más información → Ejecutar de todas formas**.
- **macOS:** si dice que no se puede abrir porque el desarrollador no está identificado, vaya a
  **Ajustes del Sistema → Privacidad y seguridad → Abrir de todos modos**, o clic derecho sobre la app → **Abrir**.
- Para que no aparezcan estos avisos hace falta un certificado de firma de código (Windows) y la
  membresía del Apple Developer Program con notarización (macOS); ambos son de pago.

## Si algo falla
El registro de errores se guarda en la carpeta personal: `NekaStat_error_log.txt`.
