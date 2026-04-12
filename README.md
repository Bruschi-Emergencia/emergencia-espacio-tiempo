# Modelo DEE v2.0 — Dinámica de Estructura de Entrelazamiento

**Autor:** Juan Pablo Bruschi  
**Versión:** 2.0 — Abril 2026  
**Repositorio:** https://github.com/Bruschi-Emergencia/emergencia-espacio-tiempo

---

## ¿Qué es el Modelo DEE?

El Modelo DEE propone que el espacio-tiempo, la gravedad, la materia oscura y la energía oscura **no son fundamentales** — emergen de una red discreta de correlaciones cuánticas cuya dinámica está determinada por la **curvatura de Ollivier-Ricci** κ_ij.

### Variable fundamental
```
κ_ij = 1 − W₁(P_i, P_j) / d_ij
```

### Hamiltoniano (único bajo S_N + invariancia de escala)
```
H[φ] = ½ Σ_{i≠j} (φ_i − φ_j)² / d²_ij
```

### Acción efectiva en el límite continuo (v2.0)
```
S = ∫ d⁴x √−g [ (1/16πG) R + (1/2)(∂φ)² − V(φ) ] + S_m
```
→ Teoría escalar-tensor con G_μν completo. Energía oscura = contribución de T^φ_μν.

### Ecuación de estado efectiva (v2.0)
```
w = −1 + ε     con ε ≈ 0.085 ± 0.048
H²(z) = H₀² [ Ωm(1+z)³ + (1−Ωm)(1+z)^{3ε} ]
```

### Conexión con DESI DR2 (Phys. Rev. D 112, octubre 2025)
DESI DR2 reporta evidencia de w ≠ −1 con 2.3σ.  
DEE predice ε > 0 (w > −1 hoy). **Consistente con los datos.** ✓

---

## Resultados principales

| SIM | Test | Predicción DEE | Resultado |
|---|---|---|---|
| SIM 1 | G(r) ∝ 1/r | ratio = N−1 | ✓ ratio = 999 ± <2 (d_min=1e-6) |
| SIM 2 | κ > 0, gravedad | 4/4 convergen | ✓ κ = 0.150 > 0 |
| SIM 3 | Ωm, ΩΛ, ε | ε > 0, χ² < ΛCDM | ✓ ε=0.085, χ²=0.57 < 0.64 |
| SIM 4 | β flujo RG | β ∝ r_c^1.72 | ✓ R²=0.982, η ≈ 6×10⁻¹⁰ |
| SIM 6 | SPARC 155 gal. | DEE > NFW, KS p≈0 | ✓ 89%, KS p=0.000 |
| SIM 7 | α por morfología | masivas < irreg. | ✓ p=0.0007 |

### Comparación AIC/BIC con ΛCDM (17 datos H(z))

| Modelo | Parámetros | AIC | ΔAIC | Peso |
|---|---|---|---|---|
| **DEE** | **0** | **9.65** | **0.00** | **58%** |
| ΛCDM Planck | 0 | 10.81 | +1.15 | 33% |
| ΛCDM ajust. | 3 | 13.57 | +3.92 | 8% |
| ΛCDM 6p | 6 | 17.49 | +7.83 | 1% |

DEE gana con 0 parámetros libres. ΔAIC > 7.8 es "evidencia fuerte" contra ΛCDM 6p.

---

## Cambios de v1.0 a v2.0

| Aspecto | v1.0 | v2.0 |
|---|---|---|
| Estructura gravitatoria | κ_ij escalar | G_μν completo (escalar-tensor) |
| Energía oscura | R_fondo ad hoc | T^φ_μν emergente de V(φ) |
| Eq. de estado | Implícita | w = −1 + ε, ajuste a datos reales |
| ε medido | No calculado | ε = 0.085 ± 0.048 |
| DESI DR2 | "dato futuro" | Publicado PRD oct. 2025 — consistente ✓ |
| AIC/BIC | No calculado | Explícito, DEE gana con 0 params |
| SIM 1 reporte | "0.000% exacto" | Honesto: 999 ± <2 (d_min=1e-6) |
| α(morfología) | α = (2/3)/f_masa | Corregido: α = (2/3)×f_masa |
| Bariogénesis | Cualitativa | η = (1−β)×(E_RH/E_Pl)^1.72 cuantitativa |

---

## Estructura del repositorio

```
emergencia-espacio-tiempo/
│
├── README.md                     ← Este archivo
├── modelo_DEE_v2.docx            ← Documento académico completo v2.0
│
├── simulaciones/
│   ├── GUIA_COLAB.py             ← Instrucciones paso a paso para Colab
│   ├── sim1_propagador_G.py      ← G(r) ∝ 1/r desde la red (N_SEEDS=20)
│   ├── sim2_curvatura_gravedad.py ← κ_ij y gravedad emergente
│   ├── sim3_friedmann_beta.py    ← Cosmología + ε + DESI DR2 + AIC/BIC
│   ├── sim4_beta_Dcorr.py        ← Flujo RG + fórmula bariogénesis
│   ├── sim6_sparc.py             ← Curvas rotación SPARC (175 galaxias)
│   └── sim7_morfologia.py        ← α por tipo morfológico (fórmula corregida)
│
└── resultados/
    ├── sim1_resultado.png
    ├── sim2_resultado.png
    ├── sim3_resultado.png
    ├── sim4_resultado.png
    ├── sim6_resultado.png
    └── sim7_resultado.png
```

---

## Instrucciones — Google Colab (sin instalar nada)

### Setup rápido
```python
# Celda 1
!pip install numpy scipy matplotlib tqdm zenodo-get -q
!git clone https://github.com/Bruschi-Emergencia/emergencia-espacio-tiempo.git
%cd emergencia-espacio-tiempo/
```

### Correr simulaciones individuales
```python
# SIM 1 (~5 min): G(r) ∝ 1/r con 20 semillas
!python sim1_propagador_G.py

# SIM 2 (~15 min): κ_ij y gravedad
!python sim2_curvatura_gravedad.py

# SIM 3 (~20 min): Cosmología + ε + DESI DR2 + AIC/BIC
!python sim3_friedmann_beta.py

# SIM 4 (~10 min): Flujo RG y bariogénesis
!python sim4_beta_Dcorr.py

# Datos SPARC (una vez)
!zenodo_get 16284118
!mkdir -p /content/SPARC_rotcurves
!unzip -q Rotmod_LTG.zip -d /content/SPARC_rotcurves

# SIM 6 (~15 min): 175 galaxias SPARC
!python sim6_sparc.py

# SIM 7 (~15 min): α por tipo morfológico
!python sim7_morfologia.py
```

### Descargar todos los resultados
```python
from google.colab import files
import glob
for f in sorted(glob.glob('sim*_resultado.png')):
    files.download(f)
```

---

## Cómo falsificar el modelo

Cada simulación tiene un criterio de falsificación explícito:

| SIM | Predicción | Se falsifica si |
|---|---|---|
| SIM 1 | ratio → N−1 cuando d_min→0 | ratio no converge con d_min |
| SIM 2 | κ > 0, 4/4 convergen | κ ≤ 0 o partículas se dispersan |
| SIM 3 | ε > 0 (w > −1) | ε < 0 o χ²_DEE > χ²_ΛCDM |
| SIM 4 | β ∝ r_c^1.72, R² > 0.90 | β constante (sin flujo RG) |
| SIM 6 | KS p < 0.01, DEE > NFW 60%+ | KS p > 0.10 |
| SIM 7 | Mann-Whitney p < 0.05 | p > 0.05 (α independiente de T) |

### Tests con datos externos (ya disponibles)

| Test | Datos | Predicción DEE | Estado |
|---|---|---|---|
| w(z) ≠ −1 | DESI DR2 (2025) | ε = 0.085 > 0 | ✓ Consistente (2.3σ) |
| BIG-SPARC | arXiv:2411.13329 | α morfológico sostenido | Pendiente verificación |
| CMB espectro | Planck 2018 + CLASS | n_s ≈ 0.965 del flujo RG | Cálculo pendiente |
| Ecos GW | Einstein Telescope ~2035 | τ ∝ M_BH, A ∝ (1−β) | Test futuro |

---

## Limitaciones reconocidas

El modelo está en etapa exploratoria avanzada. Las siguientes limitaciones son reconocidas explícitamente:

1. **V(φ) no derivada formalmente** desde la red discreta (trabajo pendiente)
2. **Convergencia κ_ij → Ric(v,v) no demostrada** como teorema para S_ij=1/d_ij
3. **Sin perturbaciones cosmológicas** (espectro CMB no calculado)
4. **Sin acoplamiento al Modelo Estándar** de partículas
5. **SIM 1**: el "0.000% exacto" ocurre en el límite d_min→0; reporte honesto con semillas múltiples en v2.0

---

## Cómo citar

```
Bruschi, J. P. (2026). Modelo DEE v2.0: Dinámica de Estructura de Entrelazamiento.
Investigación independiente.
GitHub: https://github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
```

---

## Colaboración

Se busca colaboración con:
- **Matemáticos** de geometría diferencial discreta y transporte óptimo
- **Físicos teóricos** de gravedad cuántica (causal sets, LQG, geometría no conmutativa)
- **Cosmólogos observacionales** con acceso a DESI, Euclid o LSST
- **Astrónomos** especializados en dinámica galáctica

**Contacto:** https://github.com/Bruschi-Emergencia/emergencia-espacio-tiempo

---

## Referencias clave

- [1] Ollivier, Y. (2009). Ricci curvature of Markov chains. J. Funct. Anal. 256, 810.
- [2] Lelli, F. et al. (2016). SPARC: 175 disk galaxies. AJ 152, 157.
- [3] DESI Collaboration (2025). DR2 BAO and cosmological constraints. Phys. Rev. D 112, 083515.
- [4] Planck Collaboration (2018). Cosmological parameters. A&A 641, A6.
- [5] Burnham & Anderson (2002). Model Selection and Multimodel Inference. Springer.

---

## Licencia

Scripts bajo licencia MIT.  
Datos SPARC: Lelli, McGaugh & Schombert (2016), CC-BY 4.0.

---

*"La geometría no es el escenario donde ocurre la física. La geometría es la física."*
