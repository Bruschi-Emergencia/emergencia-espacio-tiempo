# Modelo DEE v2.0 — Dinámica de Estructura de Entrelazamiento

**Versión:** 2.0 — Abril 2026  
**Estado:** Exploratoria avanzada con teoría efectiva coherente  
**Autor:** Juan Pablo Bruschi (independiente)

---

## Descripción

El Modelo DEE propone que el espacio-tiempo, la gravedad y la energía oscura son propiedades emergentes de un sustrato discreto subyacente, descriptas por correlaciones de un campo escalar φ_i definido sobre una red abstracta.

La cadena de emergencia central es:

```
{φ_i, C_ij}  →  W_ij  →  L  →  geometría  →  G_eff(k,a)  →  cosmología
```

---

## Estructura del repositorio

```
dee_repo/
├── sims/                  # Simulaciones principales (SIM 1–9c)
├── benchmarks/            # Tests con CLASS (Boltzmann real)
├── phase2_gravity/        # Tests de gravedad efectiva y degeneración
└── figures/               # Gráficos generados por los tests
```

---

## Simulaciones (sims/)

| Script | Descripción | Resultado clave |
|---|---|---|
| `sim1_propagador_G.py` | Newton desde operador K en red aleatoria | G(r) ∝ 1/r, ratio=998.9±0.0 (20 semillas) |
| `sim2_curvatura_gravedad.py` | Gravedad desde ∇κ | F∝r⁻², 4/4 partículas convergen |
| `sim3_friedmann_beta.py` | FLRW emergente | Ω_m=0.277, Ω_Λ=0.723 sin ajuste |
| `sim4_beta_Dcorr.py` | Flujo RG | β∝r_c^{1.83}, R²=0.993 |
| `sim6_sparc.py` | Galaxias SPARC reales | DEE > NFW en 89% (155 galaxias) |
| `sim7_morfologia.py` | Dependencia morfológica | p=0.0007, dirección predicha |
| `sim8_kappa_mesocopico.py` | κ→Ricci mesoscópico | C_{1/d}=(5/6)×C_uniforme |
| `sim9_potencial_efectivo.py` | V(φ) desde cumulantes | m_φ²=χ_φ⁻¹=29000–41000 |
| `sim9b_lambda4.py` | Término cuártico | λ₄ subdominante, gaussianización |
| `sim9c_tabla_sistematica.py` | Barrido sistemático N | \|V₄/V₂\|∝N⁻¹·⁵⁴ |

---

## Benchmarks CLASS (benchmarks/)

Requieren `pip install classy`. Validados con CLASS Boltzmann completo.

| Script | Descripción |
|---|---|
| `dee_classy_utils.py` | Utilidades comunes |
| `dee_class_benchmark.py` | Benchmark completo: P(k), fσ₈, CMB TT |
| `benchmark1_dee_classy.py` | G_eff constante (B1): μ₀=0.08 |
| `benchmark2_dee_classy.py` | G_eff temporal (B2): μ₀=0.06, μ₁=0.04 |

**Resultado CMB:** DEE con w0=−0.98, wa=0.05 produce espectro CMB TT indistinguible de ΛCDM hasta ℓ=2500 (ver `figures/dee_class_real.png`).

---

## Tests de gravedad — Fase 2 (phase2_gravity/)

Todos los tests usan CLASS H(z) + G_eff en post-proceso.

| Script | Descripción | Resultado |
|---|---|---|
| `dee_temporal_runner_v2.py` | G_t y F_t con CLASS real | F_t=2.43σ con σ=2% (marginal) |
| `temporal_observable_dee.py` | Funciones G_t, F_t | Librería de observables temporales |
| `Ft_scan_class_real.py` | Scan F_t vs μ₁ (σ=5%) | Indetectable μ₁≤0.30 con σ conservador |
| `dee_euclid_degeneracy.py` | Degeneración B1-B2 | 0.16σ en fσ₈ — degeneración confirmada |
| `mu1_scan_internal.py` | Scan μ₁ solver interno | Umbral 2σ: μ₁≥0.15 |
| `kz_full_analysis.py` | O_kz = d ln[P(k₁)/P(k₂)]/dz | 0.58σ — indetectable |
| `slip_test_phase2.py` | Slip gravitacional η(z) | Umbral 2σ: η₁≥0.07 |

---

## Figuras (figures/)

| Figura | Contenido |
|---|---|
| `dee_class_real.png` | **Benchmark CLASS completo:** P(k), fσ₈(z), CMB TT |
| `background_comparison.png` | H(z): DEE vs ΛCDM |
| `growth_comparison.png` | fσ₈(z): DEE vs ΛCDM vs datos |
| `mu_curve.png` | G_eff/G(k) — Benchmark 1 (constante) |
| `mu_curve_time_dependent.png` | G_eff/G(k,a) — Benchmark 2 (temporal) |
| `pk_ratio_z0p5.png` | P(k) DEE/ΛCDM en z=0.5 |
| `pk_ratio_z1p0.png` | P(k) DEE/ΛCDM en z=1.0 |
| `temporal_class_real.png` | G_t y F_t con CLASS real |
| `temporal_observable_result.png` | Scan F_t vs μ₁ (solver interno) |
| `Ft_scan_summary.png` | Scan F_t binado (σ=5% conservador) |
| `Okz_full_analysis.png` | O_kz para 6 pares (k₁,k₂) |
| `slip_scan_summary.png` | Detectabilidad slip η(z) vs η₁ |
| `dee_euclid_degeneracy.png` | Degeneración B1-B2 resumen |
| `mu1_scan_result.png` | Scan μ₁ con solver interno |

---

## Estado observacional

| Sector | Estado | Detalle |
|---|---|---|
| H(z) background | ✓ Consistente | ΔAIC=+0.51 vs ΛCDM |
| fσ₈(z) crecimiento | ✓ Compatible | ε~−0.1 preferido, dirección tensión S8 |
| CMB TT espectro | ✓ Indistinguible | Hasta ℓ=2500 con CLASS real |
| P(k) espectro | ✓ Supresión suave | −1.2% a −1.6%, plana en k |
| Degeneración B1-B2 | ✓ Confirmada | 0.16σ en fσ₈, 0.15σ en F_t (σ=5%) |
| Slip η(z) | ⚠ Pendiente | Umbral 2σ: η₁≥0.07 — requiere patch CLASS |

---

## Cómo correr en Google Colab

Ver `GUIA_COLAB.md` para instrucciones paso a paso.

```python
# Instalación básica
!pip install classy numpy scipy matplotlib pandas

# Benchmark completo (P(k) + fσ₈ + CMB)
!python dee_class_benchmark.py

# Tests fase 2 (degeneración B1-B2)
!python dee_temporal_runner_v2.py
!python Ft_scan_class_real.py
!python slip_test_phase2.py
```

---

## Parámetros DEE actuales

```python
# Background
w0 = -0.98    # Ecuación de estado
wa =  0.05    # Evolución temporal

# Benchmark 1 (G_eff constante)
mu0 = 0.08    # Amplitud modificación gravitatoria
kc  = 0.07   # Escala de transición [h/Mpc]

# Benchmark 2 (G_eff temporal)
mu0 = 0.06; mu1 = 0.04   # μ(a) = μ₀ + μ₁(1−a)
kc0 = 0.07; kc1 = 0.30   # kc(a) = kc₀(1 + kc₁(1−a))
```

---

## Próximos pasos

1. **Patch CLASS en C** — resolver δ_m(k,z) modo por modo con G_eff(k,a)
2. **Calibración Z_φ** — conectar unidades de simulación con h/Mpc
3. **Test η(k,z)** — reconstruir K_Φ y K_Ψ desde datos Euclid
4. **Paper SIM 1** — Physical Review E / CQG

---

## Licencia

MIT — uso libre con atribución.
