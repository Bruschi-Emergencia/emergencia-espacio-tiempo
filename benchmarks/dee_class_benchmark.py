"""
════════════════════════════════════════════════════════════════
  DEE + CLASS: Benchmark Boltzmann simplificado
  Modelo DEE v2.0 — Test cosmológico completo
  Autor: Juan Pablo Bruschi (2026)

  INSTRUCCIONES PARA COLAB:
  !pip install classy
  !python dee_class_benchmark.py --use-class

  Sin classy: corre el solver interno (modo independiente, default).
════════════════════════════════════════════════════════════════

IMPLEMENTA (según dee_class_patch_esqueleto.txt):
  Etapa 1 — Background DEE: H²(a) con w(a) = w0 + wa(1-a)
  Etapa 2 — G_eff(k,a): mu(k,a) = 1 - mu0*k²/(k²+kc²)
            + versión refinada: mu(k,a) = 1 + α₀(1-a)^{2s}/[(k²/a²)(k²/a²+m_φ²)]
  Etapa 3 — fσ₈(z): integración de la ecuación de Riccati
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ── Parámetros del modelo (del .ini file) ─────────────────
PARAMS = {
    # Cosmología base
    'h':          0.67,
    'Omega_b':    0.0224 / 0.67**2,
    'Omega_cdm':  0.12   / 0.67**2,
    'Omega_r':    9.1e-5,
    # DEE background
    'dee_w0':    -0.98,
    'dee_wa':     0.05,
    # DEE perturbaciones — vieja
    'dee_mu0':    0.08,
    'dee_kc':     0.07,   # h/Mpc
    # DEE perturbaciones — refinada (SIM 9 anclada)
    'dee_alpha0': 2e-4,
    'dee_s':      1.0,
    'dee_mphi2':  3e4,    # unidades sim → necesita calibración
    # σ₈ fiducial
    'sigma8_0':   0.80,
}

# Densidad de materia hoy
H0 = PARAMS['h'] * 100.  # km/s/Mpc
Omega_m0 = PARAMS['Omega_b'] + PARAMS['Omega_cdm']
Omega_r0 = PARAMS['Omega_r']
Omega_Q0 = 1.0 - Omega_m0 - Omega_r0  # energía oscura hoy

print("="*65)
print("  DEE + CLASS Benchmark — Solver interno")
print(f"  h={PARAMS['h']}  Ω_m={Omega_m0:.4f}  Ω_Q={Omega_Q0:.4f}")
print(f"  w0={PARAMS['dee_w0']}  wa={PARAMS['dee_wa']}")
print("="*65)

# ── Funciones del background ──────────────────────────────
def w_dee(a):
    """Ecuación de estado DEE: w(a) = w0 + wa(1-a)"""
    return PARAMS['dee_w0'] + PARAMS['dee_wa'] * (1.0 - a)

def rho_dee_integrand(lna):
    a = np.exp(lna)
    return 3.0 * (1.0 + w_dee(a))

# Omega_Q(a) por integración directa (usando scipy)
_lna_arr = np.linspace(-10, 0, 1000)
_integrand = np.array([rho_dee_integrand(x) for x in _lna_arr])
_cumint = np.cumsum(_integrand) * (_lna_arr[1] - _lna_arr[0])
_cumint -= _cumint[-1]  # normalizar en a=1
_rho_Q_ratio = np.exp(-_cumint)
_rho_Q_interp = interp1d(_lna_arr, _rho_Q_ratio, bounds_error=False, fill_value='extrapolate')

def Omega_Q(a):
    """Ω_Q(a) normalizado a Omega_Q0 en a=1"""
    return Omega_Q0 * _rho_Q_interp(np.log(a))

def E2(a):
    """H²(a)/H0² = E²(a)"""
    return Omega_m0*a**-3 + Omega_r0*a**-4 + Omega_Q(a)

def E(a):
    return np.sqrt(np.maximum(E2(a), 1e-20))

def H_of_a(a):
    return H0 * E(a)

def Omega_m_of_a(a):
    return Omega_m0 * a**-3 / E2(a)

def dlnH_dlna(a):
    """d ln H / d ln a — para la ecuación de Riccati"""
    da = a * 1e-4
    return a / (2*E2(a)) * (E2(a+da) - E2(a-da)) / (2*da)

# ── G_eff implementations ─────────────────────────────────
def mu_old(k_phys, a):
    """Parametrización vieja: 1 - mu0 k²/(k²+kc²)"""
    kc = PARAMS['dee_kc']
    return 1.0 - PARAMS['dee_mu0'] * k_phys**2 / (k_phys**2 + kc**2)

def mu_refined(k_phys, a, s=None):
    """
    Parametrización refinada (anclada al vacío de SIM 9):
    G_eff/G = 1 + α₀(1-a)^{2s} / [(k²/a²)(k²/a²+m_φ²)]
    
    Motivación física: V,φ→0 cerca del mínimo del potencial efectivo
    → corrección se apaga en a→1 consistentemente con SIM 9/9b/9c
    """
    if s is None: s = PARAMS['dee_s']
    alpha0 = PARAMS['dee_alpha0']
    mphi2  = PARAMS['dee_mphi2']
    k_com  = k_phys / a  # k comoving → k/a = k físico
    denom  = k_com**2 * (k_com**2 + mphi2)
    if denom < 1e-30: return 1.0
    return 1.0 + alpha0 * (1.0 - a)**(2*s) / denom

def mu_background_only(k_phys, a):
    """Sin corrección al crecimiento"""
    return 1.0

# ── Solver de crecimiento (ecuación de Riccati) ───────────
def solve_growth(mu_func, k_phys=0.15, a_ini=1e-3, a_fin=1.0):
    """
    Resuelve δ_m'' + (2 + H'/H) δ_m' - 3/2 Ω_m(a) G_eff/G δ_m = 0
    en variable N = ln a, donde ' = d/dN.
    
    Estado: y = [δ_m, δ_m' = f × δ_m]
    """
    def rhs(lna, y):
        a = np.exp(lna)
        delta, f_delta = y
        
        Hprime_over_H = dlnH_dlna(a)
        Omm = Omega_m_of_a(a)
        mu  = mu_func(k_phys, a)
        
        # δ_m'' = (3/2 Ω_m μ - f² - (2 + H'/H) f) × δ_m
        # d(f×δ_m)/dN = δ_m'' + f' × δ_m  ... resolución directa en (δ, δ')
        coeff = 2.0 + Hprime_over_H
        source = 1.5 * Omm * mu
        
        d_delta   = f_delta
        d_fdelta  = (source * delta - coeff * f_delta)
        return [d_delta, d_fdelta]
    
    # Condiciones iniciales MD: δ ∝ a → δ'=δ, f=1
    y0 = [a_ini, a_ini]
    lna_span = (np.log(a_ini), np.log(a_fin))
    lna_eval = np.linspace(*lna_span, 500)
    
    sol = solve_ivp(rhs, lna_span, y0, t_eval=lna_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)
    return sol

# ── Calcular fσ₈ para cada modelo ────────────────────────
print("\n[1/3] Resolviendo ecuación de crecimiento para 4 modelos...")

k_modes = [0.10, 0.15, 0.20]  # h/Mpc — diferentes escalas
models = [
    ('DEE background-only', mu_background_only, '#2980b9', '-'),
    ('DEE + G_eff (old)',    mu_old,             '#e74c3c', '--'),
    ('DEE + G_eff (ref s=1)',lambda k,a: mu_refined(k,a,1.0), '#f1c40f', '-.'),
    ('DEE + G_eff (ref s=2)',lambda k,a: mu_refined(k,a,2.0), '#27ae60', ':'),
]

z_arr = np.linspace(0, 2.0, 200)
a_arr = 1.0 / (1.0 + z_arr)

solutions = {}
for name, mu_func, color, ls in models:
    sol = solve_growth(mu_func, k_phys=0.15)
    a_sol = np.exp(sol.t)
    delta_sol = sol.y[0]
    fdelta_sol = sol.y[1]
    
    # f(a) = δ'/δ
    f_sol = fdelta_sol / np.maximum(delta_sol, 1e-30)
    
    # σ₈(a) = σ₈,0 × δ(a)/δ(1)
    delta_interp = interp1d(a_sol, delta_sol, bounds_error=False, fill_value='extrapolate')
    f_interp     = interp1d(a_sol, f_sol,     bounds_error=False, fill_value='extrapolate')
    
    delta_today = delta_interp(1.0)
    delta_a = delta_interp(a_arr)
    f_a     = f_interp(a_arr)
    sigma8_a = PARAMS['sigma8_0'] * delta_a / delta_today
    fsigma8 = f_a * sigma8_a
    
    solutions[name] = {
        'f': f_a, 'sigma8': sigma8_a, 'fsigma8': fsigma8,
        'delta': delta_a, 'color': color, 'ls': ls, 'mu_func': mu_func
    }

# ── Datos observacionales fσ₈ (compilación pública) ──────
print("[2/3] Cargando datos observacionales de fσ₈...")

# Selección de mediciones publicadas (compilación de 63 puntos — subset representativo)
# Fuentes: 2dFGRS, BOSS, eBOSS, VIPERS, FastSound, SDSS
fsig8_data = np.array([
    # z,    fσ₈,  σ_fσ₈,  Survey
    [0.02,  0.428, 0.0465, 1],  # 2dFGRS
    [0.067, 0.423, 0.055,  1],  # 6dFGRS
    [0.10,  0.37,  0.13,   1],  # SDSS
    [0.15,  0.490, 0.145,  1],  # SDSS MGS
    [0.17,  0.510, 0.060,  1],  # 2dFGRS
    [0.22,  0.420, 0.070,  1],  # WiggleZ
    [0.25,  0.3512,0.0583, 2],  # SDSS DR7
    [0.30,  0.407, 0.055,  2],  # SDSS
    [0.32,  0.427, 0.056,  2],  # BOSS LOWZ
    [0.35,  0.440, 0.050,  2],  # SDSS DR7
    [0.38,  0.497, 0.045,  2],  # BOSS
    [0.40,  0.419, 0.041,  2],  # WiggleZ
    [0.44,  0.413, 0.080,  2],  # WiggleZ
    [0.50,  0.427, 0.043,  2],  # BOSS CMASS
    [0.51,  0.458, 0.038,  2],  # BOSS
    [0.57,  0.426, 0.029,  3],  # BOSS CMASS
    [0.60,  0.390, 0.063,  3],  # WiggleZ
    [0.61,  0.436, 0.034,  3],  # BOSS
    [0.73,  0.437, 0.072,  3],  # WiggleZ
    [0.77,  0.490, 0.18,   3],  # VIPERS
    [0.80,  0.470, 0.080,  3],  # VIPERS
    [0.85,  0.450, 0.110,  3],  # VIPERS
    [1.40,  0.482, 0.116,  4],  # FastSound
])

# ── χ² ────────────────────────────────────────────────────
print("[3/3] Calculando χ² para cada modelo...")
chi2_results = {}
for name, data in solutions.items():
    fsig8_pred = data['fsigma8']
    fsig8_interp = interp1d(z_arr, fsig8_pred, bounds_error=False, fill_value='extrapolate')
    
    z_d, f_d, sf_d = fsig8_data[:,0], fsig8_data[:,1], fsig8_data[:,2]
    f_pred = fsig8_interp(z_d)
    chi2 = np.sum(((f_d - f_pred)/sf_d)**2)
    chi2_results[name] = chi2
    print(f"  {name:35s}  χ²={chi2:.2f}  χ²/N={chi2/len(z_d):.4f}")

# ── GRÁFICO ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117'; CW='#ecf0f1'; CGR='#7f8c8d'

def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR, labelsize=9)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True, alpha=0.15)

# Panel 1: fσ₈(z)
ax1 = axes[0]; style(ax1)
ax1.errorbar(fsig8_data[:,0], fsig8_data[:,1], yerr=fsig8_data[:,2],
             fmt='o', color='white', ms=5, alpha=0.7, capsize=3,
             label='Datos fσ₈ (compilación)', zorder=5)
for name, data in solutions.items():
    ax1.plot(z_arr, data['fsigma8'], lw=2.5,
             color=data['color'], ls=data['ls'],
             label=f"{name.split('(')[0].strip()} χ²={chi2_results[name]:.1f}")
ax1.set_xlabel('z', fontsize=10, color=CW)
ax1.set_ylabel('fσ₈(z)', fontsize=10, color=CW)
ax1.set_xlim(0, 2); ax1.set_ylim(0.2, 0.7)
ax1.set_title('fσ₈(z): DEE vs datos\nBenchmark exploratorio',
              fontsize=10, fontweight='bold', color='#f1c40f')
ax1.legend(fontsize=7, facecolor=BG, labelcolor=CW)

# Panel 2: G_eff(k) a z=0, 0.5, 1
ax2 = axes[1]; style(ax2)
k_arr = np.logspace(-2, 0.5, 200)
for z_plot, lw in [(0.0, 3), (0.5, 2), (1.0, 1.5)]:
    a_plot = 1/(1+z_plot)
    for name, data in list(solutions.items())[1:]:  # skip background-only
        mu_vals = np.array([data['mu_func'](k, a_plot) for k in k_arr])
        ax2.semilogx(k_arr, mu_vals, color=data['color'], ls=data['ls'], lw=lw,
                     label=f"z={z_plot} {name.split('(')[0].strip()}" if z_plot==0 else "")
ax2.axhline(1.0, color='white', lw=1, ls=':', alpha=0.5, label='GR estándar')
ax2.set_xlabel('k (h/Mpc)', fontsize=10, color=CW)
ax2.set_ylabel('G_eff(k,a)/G', fontsize=10, color=CW)
ax2.set_title('Acoplamiento efectivo G_eff(k,a)\nz=0 (grueso), 0.5, 1.0',
              fontsize=10, fontweight='bold', color=CW)
ax2.legend(fontsize=7, facecolor=BG, labelcolor=CW)

# Panel 3: H(z) background DEE vs ΛCDM
ax3 = axes[2]; style(ax3)
z_bg = np.linspace(0, 3, 300)
a_bg = 1/(1+z_bg)
H_dee  = np.array([H_of_a(a) for a in a_bg])
# ΛCDM reference (ε=0)
H_lcdm = H0 * np.sqrt(Omega_m0*a_bg**-3 + Omega_r0*a_bg**-4 + Omega_Q0)
ax3.plot(z_bg, H_dee,  '-',  color='#f1c40f', lw=2.5, label='DEE (w0,wa)')
ax3.plot(z_bg, H_lcdm, '--', color='#2980b9', lw=2,   label='ΛCDM (Λ puro)')
ax3.set_xlabel('z', fontsize=10, color=CW)
ax3.set_ylabel('H(z) [km/s/Mpc]', fontsize=10, color=CW)
ax3.set_title('Background: H(z) DEE vs ΛCDM',
              fontsize=10, fontweight='bold', color='#27ae60')
ax3.legend(fontsize=9, facecolor=BG, labelcolor=CW)

fig.suptitle(
    'DEE Benchmark Boltzmann (solver interno) — Para validar con CLASS\n'
    f'Ω_m={Omega_m0:.3f}  w0={PARAMS["dee_w0"]}  wa={PARAMS["dee_wa"]}  '
    f'σ₈={PARAMS["sigma8_0"]}  k_eff=0.15 h/Mpc',
    fontsize=11, fontweight='bold', color=CW)
plt.tight_layout()
plt.savefig('dee_class_benchmark.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
print("\n[OK] dee_class_benchmark.png guardado")

# ── Tabla resumen ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  TABLA: Benchmark de crecimiento fσ₈ ({len(fsig8_data)} puntos)")
print(f"{'='*65}")
print(f"  {'Modelo':35s}  {'χ²':>7}  {'N':>5}  {'χ²/N':>7}")
print(f"  {'-'*60}")
for name, chi2 in chi2_results.items():
    N = len(fsig8_data)
    print(f"  {name:35s}  {chi2:>7.2f}  {N:>5}  {chi2/N:>7.4f}")

print(f"\n  Parámetros usados:")
print(f"  {'Ω_m0':20s} = {Omega_m0:.4f}")
print(f"  {'w0':20s} = {PARAMS['dee_w0']}")
print(f"  {'wa':20s} = {PARAMS['dee_wa']}")
print(f"  {'σ₈,0':20s} = {PARAMS['sigma8_0']}")
print(f"  {'k_eff (h/Mpc)':20s} = 0.15")
print(f"  {'m_φ² (unid. sim)':20s} = {PARAMS['dee_mphi2']:.0e}")
print(f"  {'α₀':20s} = {PARAMS['dee_alpha0']:.0e}")

print(f"\n  NOTA: Comparación exploratoria — sin matriz de covarianza")
print(f"  Para validación con CLASS real: !pip install classy en Colab")
print(f"{'='*65}")

# ── Instrucciones para CLASS real ─────────────────────────
print("""
PRÓXIMO PASO — CLASS REAL EN COLAB:
  1. !pip install classy
  2. from classy import Class
  3. Agregar los parámetros DEE de dee_minimal_test.ini
  4. Comparar P(k) y C_ℓ con este benchmark interno
  5. Los χ² de este script son el baseline de comparación
""")

# ════════════════════════════════════════════════════════════════
# RAMA CLASS REAL — se activa automáticamente si classy está instalado
# ════════════════════════════════════════════════════════════════

def run_with_class():
    """
    Corre el benchmark completo usando CLASS real (classy).
    Compara P(k) y fσ₈ DEE vs ΛCDM a nivel Boltzmann completo.
    """
    from classy import Class

    print("\n" + "="*65)
    print("  RAMA CLASS REAL — solver Boltzmann completo")
    print("="*65)

    # ── Parámetros base compartidos ───────────────────────
    base = {
        'h':             PARAMS['h'],
        'omega_b':       0.0224,
        'omega_cdm':     0.12,
        'A_s':           2.1e-9,
        'n_s':           0.965,
        'tau_reio':      0.054,
        'output':        'mPk,tCl,lCl',
        'lensing':       'yes',
        'P_k_max_h/Mpc': 10.0,
        'z_pk':          '0.0,0.5,1.0',
        'l_max_scalars': 2500,
    }

    # ── Caso 1: ΛCDM puro (referencia) ───────────────────
    print("\n  [1/4] ΛCDM puro (referencia)...")
    cosmo_lcdm = Class()
    # ΛCDM: no fluid, Omega_Lambda inferred from flatness
    cosmo_lcdm.set({**base})
    cosmo_lcdm.compute()

    # ── Caso 2: DEE background-only (w0wa) ───────────────
    print("  [2/4] DEE background-only (w0,wa)...")
    cosmo_dee_bg = Class()
    # DEE bg: set w0/wa fluid, let CLASS compute Omega_fld from flatness
    cosmo_dee_bg.set({
        **base,
        'w0_fld': PARAMS['dee_w0'],
        'wa_fld': PARAMS['dee_wa'],
        'Omega_Lambda': 0,  # =0 forces CLASS to use Omega_fld
    })
    cosmo_dee_bg.compute()

    # ── Caso 3: DEE + G_eff refinado (s=1) ───────────────
    # CLASS estándar no tiene G_eff nativo.
    # Aproximación: usar mu_of_k como modificación efectiva
    # de sigma_8 para el P(k) integrado.
    # Para G_eff completo se necesita el patch de dee_class_patch_esqueleto.txt
    print("  [3/4] DEE + G_eff refinado s=1 (aproximación efectiva)...")
    # Aproximamos G_eff promedio en k=0.15 como rescalado de σ₈
    k_eff = 0.15; a_today = 1.0
    mu_eff = mu_refined(k_eff, a_today, s=1.0)
    sigma8_eff = PARAMS['sigma8_0'] * mu_eff**0.5

    cosmo_dee_geff = Class()
    # DEE+G_eff: same background, A_s rescalado como proxy de G_eff
    cosmo_dee_geff.set({
        **base,
        'w0_fld':      PARAMS['dee_w0'],
        'wa_fld':      PARAMS['dee_wa'],
        'Omega_Lambda': 0,
        'A_s':          2.1e-9 * mu_eff,  # proxy G_eff
    })
    cosmo_dee_geff.compute()

    # ── Extraer P(k) ──────────────────────────────────────
    print("  [4/4] Extrayendo P(k), fσ₈, Cl...")
    k_arr_pk = np.logspace(-3, 0.7, 200)  # h/Mpc

    Pk_lcdm = np.array([cosmo_lcdm.pk(k * PARAMS['h'],    0.0) * PARAMS['h']**3
                         for k in k_arr_pk])
    Pk_dee   = np.array([cosmo_dee_bg.pk(k * PARAMS['h'],  0.0) * PARAMS['h']**3
                         for k in k_arr_pk])
    Pk_geff  = np.array([cosmo_dee_geff.pk(k * PARAMS['h'],0.0) * PARAMS['h']**3
                         for k in k_arr_pk])

    # ── fσ₈ desde CLASS ───────────────────────────────────
    z_fsig = np.linspace(0.01, 2.0, 100)
    fsig_lcdm = []
    fsig_dee  = []
    for z in z_fsig:
        try:
            s8_lcdm = cosmo_lcdm.sigma(8.0/PARAMS['h'], z)
            s8_dee  = cosmo_dee_bg.sigma(8.0/PARAMS['h'], z)
            # f ≈ Ω_m(z)^0.55 como aproximación
            Om_z_lcdm = Omega_m0*(1+z)**3 / (Omega_m0*(1+z)**3 + Omega_Q0)
            Om_z_dee  = Omega_m_of_a(1/(1+z))
            fsig_lcdm.append(Om_z_lcdm**0.55 * s8_lcdm)
            fsig_dee.append(Om_z_dee**0.55  * s8_dee)
        except Exception:
            fsig_lcdm.append(np.nan)
            fsig_dee.append(np.nan)

    # ── Cl del CMB ────────────────────────────────────────
    cl_lcdm = cosmo_lcdm.lensed_cl(2500)
    cl_dee  = cosmo_dee_bg.lensed_cl(2500)
    ell = cl_lcdm['ell'][2:]

    # ── GRÁFICO CLASS ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a1a')
    def style(ax):
        ax.set_facecolor('#0d1117'); ax.tick_params(colors='#7f8c8d')
        for s in ax.spines.values(): s.set_color('#2c3e50')
        ax.grid(True, alpha=0.15)

    # P(k)
    ax1 = axes[0]; style(ax1)
    ax1.loglog(k_arr_pk, Pk_lcdm, '-',  color='#2980b9', lw=2.5, label='ΛCDM')
    ax1.loglog(k_arr_pk, Pk_dee,  '--', color='#f1c40f', lw=2,   label='DEE (w0,wa)')
    ax1.loglog(k_arr_pk, Pk_geff, ':',  color='#27ae60', lw=2,   label='DEE+G_eff (ref s=1)')
    ax1.set_xlabel('k (h/Mpc)', fontsize=10, color='#ecf0f1')
    ax1.set_ylabel('P(k) [(Mpc/h)³]', fontsize=10, color='#ecf0f1')
    ax1.set_title('Espectro de potencias P(k)\nCLASS real', fontsize=10,
                  fontweight='bold', color='#f1c40f')
    ax1.legend(fontsize=9, facecolor='#0d1117', labelcolor='#ecf0f1')

    # fσ₈
    ax2 = axes[1]; style(ax2)
    ax2.errorbar(fsig8_data[:,0], fsig8_data[:,1], yerr=fsig8_data[:,2],
                 fmt='o', color='white', ms=5, alpha=0.7, capsize=3)
    ax2.plot(z_fsig, fsig_lcdm, '-',  color='#2980b9', lw=2.5, label='ΛCDM')
    ax2.plot(z_fsig, fsig_dee,  '--', color='#f1c40f', lw=2,   label='DEE bg-only')
    # Agregar la curva del solver interno para comparar
    ax2.plot(z_arr, solutions['DEE background-only']['fsigma8'],
             ':', color='#e74c3c', lw=1.5, label='Solver interno (ref)')
    ax2.set_xlabel('z', fontsize=10, color='#ecf0f1')
    ax2.set_ylabel('fσ₈(z)', fontsize=10, color='#ecf0f1')
    ax2.set_title('fσ₈: CLASS vs solver interno\nValidación cruzada',
                  fontsize=10, fontweight='bold', color='#27ae60')
    ax2.legend(fontsize=9, facecolor='#0d1117', labelcolor='#ecf0f1')
    ax2.set_xlim(0, 2); ax2.set_ylim(0.2, 0.7)

    # CMB TT
    ax3 = axes[2]; style(ax3)
    norm = ell*(ell+1)/(2*np.pi) * 1e12
    ax3.plot(ell, norm * cl_lcdm['tt'][2:], '-',  color='#2980b9', lw=2.5, label='ΛCDM')
    ax3.plot(ell, norm * cl_dee['tt'][2:],  '--', color='#f1c40f', lw=2,   label='DEE (w0,wa)')
    ax3.set_xlabel('ℓ', fontsize=10, color='#ecf0f1')
    ax3.set_ylabel('ℓ(ℓ+1)Cℓ/2π  [μK²]', fontsize=10, color='#ecf0f1')
    ax3.set_title('Espectro CMB TT\nDEE vs ΛCDM', fontsize=10,
                  fontweight='bold', color='#e74c3c')
    ax3.legend(fontsize=9, facecolor='#0d1117', labelcolor='#ecf0f1')
    ax3.set_xlim(2, 2500)

    fig.suptitle('DEE + CLASS REAL — Benchmark Boltzmann completo\n'
                 f'w0={PARAMS["dee_w0"]}  wa={PARAMS["dee_wa"]}  '
                 f'μ₀={PARAMS["dee_mu0"]}  kc={PARAMS["dee_kc"]} h/Mpc',
                 fontsize=11, fontweight='bold', color='#ecf0f1')
    plt.tight_layout()
    plt.savefig('dee_class_real.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a')
    print("\n[OK] dee_class_real.png guardado")

    # Limpiar objetos CLASS
    for c in [cosmo_lcdm, cosmo_dee_bg, cosmo_dee_geff]:
        c.struct_cleanup()
        c.empty()

    return Pk_lcdm, Pk_dee, z_fsig, fsig_lcdm, fsig_dee


# ── Entry point: detectar classy automáticamente ──────────
try:
    import classy
    print("\n[INFO] classy detectado — activando rama CLASS real...")
    run_with_class()
except ImportError:
    print("\n[INFO] classy no disponible — usando solver interno (modo standalone).")
    print("       Para activar CLASS real en Colab:")
    print("         !pip install classy")
    print("         !python dee_class_benchmark.py")
