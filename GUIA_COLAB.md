# Guía Colab — DEE v2.0
## Cómo reproducir todos los resultados desde cero

---

## Requisitos

```python
!pip install classy numpy scipy matplotlib pandas astropy
```

---

## Parte 1 — Simulaciones principales (no requieren CLASS)

Cada sim es independiente. Correr en orden:

### SIM 1 — Propagador gravitacional
```python
!python sims/sim1_propagador_G.py
```
**Resultado esperado:** `ratio = 998.9 ± 0.0` sobre 20 semillas. Tiempo: ~2 min.

### SIM 2 — Curvatura y gravedad emergente
```python
!python sims/sim2_curvatura_gravedad.py
```
**Resultado esperado:** F∝r⁻², 4/4 partículas convergen. Tiempo: ~3 min.

### SIM 3 — FLRW emergente
```python
!python sims/sim3_friedmann_beta.py
```
**Resultado esperado:** Ω_m=0.277, Ω_Λ=0.723. Tiempo: ~1 min.

### SIM 4 — Flujo RG
```python
!python sims/sim4_beta_Dcorr.py
```
**Resultado esperado:** β∝r_c^{1.83}, R²=0.993. Tiempo: ~2 min.

### SIM 6 — Galaxias SPARC
```python
# Requiere descargar SPARC dataset primero
!wget http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
!python sims/sim6_sparc.py
```
**Resultado esperado:** DEE>NFW en 89% de 155 galaxias. Tiempo: ~5 min.

### SIM 7 — Dependencia morfológica
```python
!python sims/sim7_morfologia.py
```
**Resultado esperado:** KS p=0.0007. Tiempo: ~3 min.

### SIM 8 — κ mesoscópico
```python
!python sims/sim8_kappa_mesocopico.py
```
**Resultado esperado:** C_{1/d}=(5/6)×C_u. Tiempo: ~5 min.

### SIM 9 — Potencial V(φ)
```python
!python sims/sim9_potencial_efectivo.py
!python sims/sim9b_lambda4.py
!python sims/sim9c_tabla_sistematica.py
```
**Resultado esperado:** m_φ²=29000–41000, |V₄/V₂|∝N⁻¹·⁵⁴. Tiempo: ~10 min.

---

## Parte 2 — Benchmarks CLASS (requieren classy)

```python
!pip install classy
```

### Benchmark completo: P(k) + fσ₈ + CMB TT
```python
!python benchmarks/dee_class_benchmark.py
```
**Resultado esperado:** 3 paneles — P(k) suprimido ~1.5%, fσ₈ levemente bajo, CMB TT idéntico a ΛCDM hasta ℓ=2500. Tiempo: ~5 min.

### Benchmark 1 — G_eff constante (B1)
```python
!python benchmarks/benchmark1_dee_classy.py
```

### Benchmark 2 — G_eff temporal (B2)
```python
!python benchmarks/benchmark2_dee_classy.py
```

---

## Parte 3 — Tests de gravedad, Fase 2

### Test temporal G_t y F_t con CLASS real
```python
from google.colab import files
files.upload()  # subir dee_temporal_runner_v2.py
!python dee_temporal_runner_v2.py
```
**Resultado esperado:**
```
B2: G_t=0.47σ  F_t=2.43σ  → degeneración parcial
```
Descargar resultados:
```python
import shutil
shutil.make_archive('temporal_outputs','zip','temporal_outputs')
files.download('temporal_outputs.zip')
```

### Scan F_t binado (σ=5% conservador)
```python
!python phase2_gravity/Ft_scan_class_real.py
```
**Resultado esperado:** F_t < 1.1σ para todo μ₁ ≤ 0.30 con σ=5%.

### Observable O_kz escala-tiempo
```python
# Requiere tablas P(k,z) — generadas por benchmarks
!python phase2_gravity/kz_full_analysis.py
```
**Resultado esperado:** max 0.58σ (indetectable con σ=3%).

### Slip gravitacional η(z)
```python
!python phase2_gravity/slip_test_phase2.py
```
**Resultado esperado:**
```
η₁=0.07 → ~2σ (umbral detección)
η₁=0.10 → 2.92σ ✓
η₁=0.20 → 5.84σ ✓✓
```

### Degeneración B1-B2 Euclid
```python
!python phase2_gravity/dee_euclid_degeneracy.py
```
**Resultado esperado:** 0.16σ en fσ₈ — degeneración confirmada.

---

## Resumen de resultados esperados

| Test | Script | Resultado clave |
|---|---|---|
| Newton | sim1 | ratio=998.9 |
| Gravedad | sim2 | F∝r⁻², 4/4 convergen |
| Friedmann | sim3 | Ω_m=0.277, Ω_Λ=0.723 |
| RG | sim4 | β∝r_c^{1.83} |
| SPARC | sim6 | DEE>NFW 89% |
| Morfología | sim7 | p=0.0007 |
| CMB+P(k)+fσ₈ | dee_class_benchmark | Indistinguible de ΛCDM en CMB |
| Temporal | dee_temporal_runner_v2 | F_t=2.43σ (σ=2%) |
| Ft binado | Ft_scan_class_real | <1.1σ (σ=5%) |
| O_kz | kz_full_analysis | 0.58σ |
| Slip | slip_test_phase2 | umbral η₁≥0.07 |

---

## Nota sobre el patch CLASS

Algunos tests (O_kz modo por modo, slip desde sustrato) requieren el patch DEE compilado en CLASS (C). Eso modifica las ecuaciones de perturbaciones en clase para resolver δ_m(k,z) con G_eff(k,a) modo por modo. Sin el patch, estos tests usan aproximaciones de post-proceso.

El patch está en desarrollo — ver §11 del documento principal para detalles.

---

## Contacto

Para colaboración en:
- Implementación CLASS/CAMB
- Tests observacionales Euclid/DESI
- Demostración rigurosa κ→Ricci

Ver §11 del documento `modelo_DEE_v2.docx`.
