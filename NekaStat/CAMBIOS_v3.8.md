# NekaStat v3.8 (antes STATIS Dual App v3.7d) — cambios (2026-10-25)

Archivos modificados:
- Carpeta raíz: ui_main.py
- Carpeta core/: m1_loader.py, m3_blocks.py, m4_interstructure.py, m6_outputs_static.py, m7_outputs_interactive.py
- No hace falta modificar core/m0_config.py: nombre y versión se definen en ui_main.py

## 1. Selector de biplot (m4, F11)
- biplot_type = "HJ" (por defecto): HJ-Biplot del compromiso (Galindo-Villardón, 1986)
  Fk = sqrt(alpha_k / (T_k * ||W_k||_F)) * Z_k U   -> filas de Zc U, con Zc'Zc = W*, J'J = Lambda
- biplot_type = "GH": Fk = Z_k U Lambda^(-1/2) (Gabriel, 1971). Reproduce las coordenadas de la v3.7d.
- "symmetric" se acepta por compatibilidad y se calcula como "GH".
- Títulos de figuras (G01, trayectorias, trayectoria individual, HTML) muestran el biplot calculado.

## 2. cos² corregido (m4, F12)
- Antes: cos² con coordenadas divididas entre sqrt(lambda) -> proporción de distancia de Mahalanobis.
- Ahora: cos²_ij = (z_i'u_j)² / ||z_i||² (coordenadas principales; Lebart et al., 1995).
- Igual con HJ y con GH. Los valores difieren de la v3.7d.
- Título del gráfico de COS² y encabezado de la hoja COS2 con la fórmula correcta (ya no mencionan 1/√λ).

## 3. Escala de los gráficos (m6)
- Trayectoria individual: límites según los puntos (antes mínimo fijo ±1); vectores con un factor común c.
- G01 y trayectorias conjuntas: nota al pie con los factores k (puntos) y c (vectores).

## 4. Citas corregidas
- Operador W_k: Lavit et al. (1994, p. 107, Ejemplo 1) (antes "ec. 2", inexistente).
- Intraestructura: círculo de correlaciones del ACP (Lebart et al., 1995) (antes "Lavit §3.3", que trata de trayectorias).
- Coordenadas GH: Gabriel (1971) (antes atribuidas a Lavit §3.2).

## 5. Trayectorias de las variables (Lavit et al., 1994, p. 107) — NUEVO
- m4: Gk = Wk*·U·Λ^(-1/2) para cada tabla y variable; se cumple Σ αk·Gk = U·√λ (verificado, dif. 3e-16).
- m6: gráfico plot_variable_trajectories (todas las variables o una) y hoja Excel "Tray_Variables_Lavit"
  con coordenadas y distancia al compromiso.
- Interpretación: si las tablas son periodos, es una trayectoria temporal; si son entidades
  (países, universidades), muestra cuánto se aparta cada entidad del compromiso en cada variable.

## 6. Interfaz (ui_main.py)
- La casilla "HJ-Biplot (desact. = Biplot simétrico)" se reemplaza por un menú:
  "HJ-Biplot del compromiso (Galindo, 1986)" -> "HJ"  |  "GH-Biplot (Gabriel, 1971) — escala v3.7d" -> "GH".
- Nueva pestaña "Tray. Variables" (solo modo longitudinal) con selector de variable y exportación PNG/TIFF.
- Casilla "Tray. variables (Lavit)" en "Gráficos a generar".
- Textos de menú y "Acerca de" con las referencias de HJ y GH.

## 7. Nombre de la aplicación
- NekaStat — N-way Exploratory K-table Analysis for Statistics, v3.8.
- Pantalla de inicio, título de la ventana, menú, panel izquierdo, "Acerca de" y nombre por defecto del Excel.
- El subtítulo deja de decir "Longitudinal": la app sirve para datos longitudinales y no longitudinales.
- Excel: "Versión del cálculo" = "NekaStat 3.8 (2026-10-25)".

## 8. Logo e ícono
- Carpeta nueva assets/ (junto a ui_main.py): NekaStat.ico y NekaStat_icono_256.png.
- Ícono en la ventana, la barra de tareas y la pantalla de inicio. Si falta assets/, se usa una copia incrustada.
- Si compila con PyInstaller, agregue: --add-data "assets;assets" --icon assets/NekaStat.ico

## 9. Trazabilidad (m6, Excel)
- Resumen_Global: "Tipo de biplot", "Versión del cálculo" y fórmula del cos².
- Coords: encabezado con la fórmula aplicada.

## Sin cambios
alpha_k, RV, valores propios, inercia, cargas U·sqrt(lambda), intraestructura, cos² respecto del consenso, modo bloques.

## Verificación
- Interfaz probada sin pantalla con el panel Ecuador: el menú envía HJ/GH, los títulos cambian, la pestaña nueva se genera.
LATAM (179 obs.) y Ecuador (163 obs.): GH = hoja Coords de la v3.7d y HJ = hoja Verif_Biplot_Coords,
diferencia máxima 5e-7; cos² = definición clásica en todas las observaciones; todas las figuras,
Excel y HTML generados sin errores con ambos tipos.
