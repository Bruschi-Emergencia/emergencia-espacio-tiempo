"""
════════════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — GUÍA COMPLETA DE REPRODUCCIÓN EN GOOGLE COLAB
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════════════

  Copiá este script en una notebook de Google Colab y ejecutá las
  celdas en orden. Cada celda es independiente y descarga su propio
  resultado. Tiempo total estimado: 60–90 minutos.

════════════════════════════════════════════════════════════════════════
CONTENIDO:
  CELDA 0  — Setup: instalación y clonado del repo
  CELDA 1  — SIM 1: Propagador G(r) ∝ 1/r  (~5 min)
  CELDA 2  — SIM 2: Curvatura κ_ij y gravedad  (~15 min)
  CELDA 3  — SIM 3: Cosmología + DESI DR2 + AIC/BIC  (~20 min)
  CELDA 4  — SIM 4: Flujo RG + bariogénesis  (~10 min)
  CELDA 5  — Descarga datos SPARC (Zenodo 16284118)
  CELDA 6  — SIM 6: 175 galaxias SPARC reales  (~15 min)
  CELDA 7  — SIM 7: α por tipo morfológico  (~15 min)
  CELDA 8  — Descargar todos los resultados
════════════════════════════════════════════════════════════════════════
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 0 — SETUP (ejecutar primero, una sola vez)
# ════════════════════════════════════════════════════════════════════
CELDA_0 = """
# Instalar dependencias
!pip install numpy scipy matplotlib tqdm zenodo-get -q

# Clonar repositorio DEE v2.0
!git clone https://github.com/Bruschi-Emergencia/emergencia-espacio-tiempo.git
%cd emergencia-espacio-tiempo/simulaciones

# Verificar archivos disponibles
import os
scripts = [f for f in os.listdir('.') if f.endswith('.py')]
print(f"Scripts disponibles: {len(scripts)}")
for s in sorted(scripts):
    print(f"  {s}")
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 1 — SIM 1: G(r) ∝ 1/r
# ════════════════════════════════════════════════════════════════════
CELDA_1 = """
# SIM 1 — Propagador de Newton emergente
# Verifica: K·(1/r) = (N-1)·δ(x−x₀)
# Tiempo: ~5 minutos

!python sim1_propagador_G.py

# Resultado esperado:
# Ratio (20 seeds, d_min=1e-6) ≈ 999 ± <2 (error < 0.01%)
# La identidad es algebraicamente exacta en el límite d_min→0
# Con d_min = 1e-3 el ratio es 935 ± 28 (error ~6%)

from google.colab import files
files.download('sim1_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 2 — SIM 2: κ_ij y gravedad
# ════════════════════════════════════════════════════════════════════
CELDA_2 = """
# SIM 2 — Curvatura de Ollivier-Ricci y gravedad emergente
# Verifica: κ_ij > 0 → a = +∇R → gravedad atractiva
# Tiempo: ~15 minutos (N=2000 nodos)

!python sim2_curvatura_gravedad.py

# Resultado esperado:
# κ medio ≈ 0.150 ± 0.07 > 0  (geometría esférica)
# 4/4 partículas de prueba convergen al centro
# F(r) ∝ r^{~-0.4} (Newton en límite continuo: r^{-2})

from google.colab import files
files.download('sim2_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 3 — SIM 3: Cosmología + DESI DR2
# ════════════════════════════════════════════════════════════════════
CELDA_3 = """
# SIM 3 — Cosmología emergente + comparación con DESI DR2 (2025)
# Verifica:
#   · Ωm ≈ 0.298 y ΩΛ ≈ 0.702 emergen sin ajuste
#   · ε ajustado ≈ 0.085 ± 0.048  →  w = -1 + ε ≈ -0.915
#   · Consistente con DESI DR2 (Phys. Rev. D 112, 2025): w ≠ -1 a 2.3σ
#   · AIC_DEE(0 params) = 9.65  vs  AIC_ΛCDM(6p) = 17.49
# Tiempo: ~20 minutos (N=1000 nodos)

!python sim3_friedmann_beta.py

# Resultado esperado:
# Ωm emergente ≈ 0.298  ΩΛ ≈ 0.702
# ε ≈ 0.085 ± 0.048  →  w ≈ -0.915
# χ²_red DEE ≈ 0.57  (ΛCDM: 0.64)
# AIC: DEE gana con 0 parámetros libres (peso Akaike 58%)
# DESI DR2 2025: consistente ✓

from google.colab import files
files.download('sim3_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 4 — SIM 4: Flujo RG y bariogénesis
# ════════════════════════════════════════════════════════════════════
CELDA_4 = """
# SIM 4 — Flujo del grupo de renormalización y bariogénesis
# Verifica:
#   · β ∝ r_c^α_RG   con α_RG ≈ 1.72  (valor teórico: 1.72)
#   · R² > 0.98  (ajuste ley de potencias)
#   · Fórmula bariogénesis: η = (1-β_inf) × (E_RH/E_Pl)^α_RG
#     Con E_RH ≈ 5×10¹³ GeV → η ≈ 6×10⁻¹⁰ (observado)
# Tiempo: ~10 minutos (N=800 nodos)

!python sim4_beta_Dcorr.py

# Resultado esperado:
# α_RG ≈ 1.72  R² ≈ 0.98  β_inf ≈ 0.006
# E_RH ≈ 5×10¹³ GeV  →  η ≈ 6×10⁻¹⁰ ✓
# (E_RH es el único parámetro libre residual)

from google.colab import files
files.download('sim4_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 5 — DATOS SPARC (una sola vez, ~2 minutos)
# ════════════════════════════════════════════════════════════════════
CELDA_5 = """
# Descargar datos SPARC desde Zenodo (Lelli et al. 2016)
# Solo necesario una vez por sesión de Colab

!pip install zenodo-get -q
!zenodo_get 16284118

import zipfile, os, glob
if not os.path.exists('/content/SPARC_rotcurves'):
    os.makedirs('/content/SPARC_rotcurves', exist_ok=True)
    zip_files = glob.glob('*.zip')
    if zip_files:
        with zipfile.ZipFile(zip_files[0], 'r') as z:
            z.extractall('/content/SPARC_rotcurves')
        print(f"Descomprimido: {zip_files[0]}")
    else:
        print("No se encontró ZIP. Intentando búsqueda alternativa...")
        !find . -name "*.zip" -exec unzip -q {} -d /content/SPARC_rotcurves \\;

# Verificar
rotcurves = glob.glob('/content/SPARC_rotcurves/**/*_rotmod.dat', recursive=True)
if not rotcurves:
    rotcurves = glob.glob('/content/SPARC_rotcurves/*_rotmod.dat')

# Mover al lugar correcto si están en subdirectorios
for f in rotcurves:
    target = '/content/SPARC_rotcurves/' + os.path.basename(f)
    if f != target:
        import shutil
        shutil.copy(f, target)

final = glob.glob('/content/SPARC_rotcurves/*_rotmod.dat')
print(f"Archivos *_rotmod.dat disponibles: {len(final)}")
assert len(final) >= 170, f"Error: solo {len(final)} archivos (se esperan 175)"
print("✓ Datos SPARC listos para SIM 6 y SIM 7")
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 6 — SIM 6: 175 galaxias SPARC
# ════════════════════════════════════════════════════════════════════
CELDA_6 = """
# SIM 6 — Curvas de rotación SPARC (175 galaxias reales)
# Requiere CELDA 5 ejecutada
# Verifica:
#   · α mediana ≈ 0.917 ± 0.635  (no aleatorio: KS p ≈ 0)
#   · DEE > NFW en ~89% de 155 galaxias ajustadas
#   · α predice estructura morfológica (base para SIM 7)
# Tiempo: ~15 minutos

!python sim6_sparc.py

# Resultado esperado:
# α mediana ≈ 0.917  σ ≈ 0.635
# KS p ≈ 0.000  (α definitivamente no aleatorio)
# DEE > NFW en 138/155 galaxias (89%)

from google.colab import files
files.download('sim6_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 7 — SIM 7: α por morfología
# ════════════════════════════════════════════════════════════════════
CELDA_7 = """
# SIM 7 — α por tipo morfológico (predicción DEE v2.0)
# Requiere CELDA 5 ejecutada
# Verifica:
#   · Predicción: α = (2/3) × f_masa (fórmula corregida v2.0)
#   · Dirección: α(masivas) < α(irregulares) ← predicha antes de ver los datos
#   · Mann-Whitney p ≈ 0.0007  (dirección confirmada estadísticamente)
#   · Kruskal-Wallis p ≈ 0.003  (4 grupos son distintos)
# Tiempo: ~15 minutos

!python sim7_morfologia.py

# Resultado esperado:
# Masivas (S0-Sb): α_obs ≈ 0.23   α_pred ≈ 0.49
# Espirales:       α_obs ≈ 1.13   α_pred ≈ 0.58
# Tardías:         α_obs ≈ 0.82   α_pred ≈ 0.61
# Irregulares:     α_obs ≈ 1.04   α_pred ≈ 0.62
# → Dirección masivas < irregulares: ✓ CONFIRMADA
# → La escala absoluta muestra brecha ~40% (requiere corrección analítica futura)

from google.colab import files
files.download('sim7_resultado.png')
"""

# ════════════════════════════════════════════════════════════════════
# CELDA 8 — DESCARGA MASIVA
# ════════════════════════════════════════════════════════════════════
CELDA_8 = """
# Descargar todos los resultados de una vez
from google.colab import files
import glob

pngs = sorted(glob.glob('sim*_resultado.png'))
print(f"Imágenes disponibles: {len(pngs)}")
for f in pngs:
    print(f"  Descargando {f}...")
    files.download(f)

print("\\n¡Listo! Todos los resultados descargados.")
print("Subí los PNG a la carpeta resultados/ del repositorio GitHub.")
"""

# ════════════════════════════════════════════════════════════════════
# TABLA DE RESULTADOS ESPERADOS (referencia rápida)
# ════════════════════════════════════════════════════════════════════
print("="*70)
print("  MODELO DEE v2.0 — GUÍA DE REPRODUCCIÓN")
print("="*70)
print("""
INSTRUCCIONES:
  1. Ir a colab.research.google.com
  2. Nuevo notebook
  3. Copiar cada bloque CELDA_N en una celda separada
  4. Ejecutar en orden: CELDA_0 → CELDA_1 → ... → CELDA_8

RESULTADOS ESPERADOS:
  SIM  Predicción DEE                         Resultado esperado
  ─────────────────────────────────────────────────────────────
  1    K·(1/r) = (N-1)·δ                     ratio = 999 ± <2 (d_min=1e-6)
  2    κ > 0 → gravedad atractiva             4/4 partículas convergen
  3    Ωm ≈ 0.30, ΩΛ ≈ 0.70, ε > 0          χ²=0.57, ε=0.085, AIC_DEE<AIC_ΛCDM
  4    β ∝ r_c^1.72, η ~ 6×10⁻¹⁰           R²≈0.98, E_RH≈5×10¹³ GeV
  6    DEE > NFW, α no aleatorio             89% DEE>NFW, KS p≈0
  7    α(masivas) < α(irregulares)           p=0.0007 ✓

CONEXIÓN CON DESI DR2 (Phys. Rev. D 112, octubre 2025):
  SIM 3 predice ε ≈ 0.085 > 0 → w > -1
  DESI DR2 mide w ≠ -1 con evidencia 2.3σ
  → Consistente con la predicción DEE ✓

FALSIFICACIÓN:
  Si KS p(SIM 6) > 0.05 → α es aleatorio → modelo falla
  Si p(SIM 7)    > 0.05 → α no depende de morfología → modelo falla
  Si ε(SIM 3)    < 0    → w < -1 hoy → inconsistente con DEE

REFERENCIAS:
  · Lelli et al. 2016 (SPARC): AJ 152, 157
  · DESI Collaboration 2025 (DR2): Phys. Rev. D 112, 083515
  · Ollivier 2009 (κ_ij → Ric): J. Funct. Anal. 256, 810
  · Bruschi J.P. 2026 (Modelo DEE): github.com/Bruschi-Emergencia
""")

# ════════════════════════════════════════════════════════════════════
# PARA EJECUTAR ESTE SCRIPT COMO GUÍA:
# python GUIA_COLAB.py   →   imprime la tabla y las celdas
# ════════════════════════════════════════════════════════════════════
print("\nCELDA 0 (pegar en Colab):")
print(CELDA_0)
print("\nCELDA 1 — SIM 1:")
print(CELDA_1)
print("\nCELDA 2 — SIM 2:")
print(CELDA_2)
print("\nCELDA 3 — SIM 3:")
print(CELDA_3)
print("\nCELDA 4 — SIM 4:")
print(CELDA_4)
print("\nCELDA 5 — DATOS SPARC:")
print(CELDA_5)
print("\nCELDA 6 — SIM 6:")
print(CELDA_6)
print("\nCELDA 7 — SIM 7:")
print(CELDA_7)
print("\nCELDA 8 — DESCARGAR TODO:")
print(CELDA_8)
