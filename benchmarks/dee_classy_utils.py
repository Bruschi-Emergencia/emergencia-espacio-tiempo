"""
dee_classy_utils.py
════════════════════════════════════════════════════════════════
Módulo compartido para los benchmarks DEE + CLASS.
Funciona en DOS modos automáticamente:

  MODO A — classy instalado (Colab con !pip install classy):
    Usa CLASS real para background, P(k), CMB.

  MODO B — sin classy (cualquier entorno):
    Usa solver interno equivalente (mismo pipeline,
    sin bariones/radiación completa, pero válido z<1).

Importado por benchmark1_dee_classy.py y benchmark2_dee_classy.py.
════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ── Detectar classy ──────────────────────────────────────
try:
    from classy import Class as _Class
    HAS_CLASSY = True
except ImportError:
    HAS_CLASSY = False

BG = '#0d1117'; CW = '#ecf0f1'; CY = '#f1c40f'
CB = '#2980b9'; CG = '#27ae60'; CR = '#e74c3c'; CGR = '#7f8c8d'

# ════════════════════════════════════════════════════════
# PARÁMETROS BASE
# ════════════════════════════════════════════════════════
def lcdm_params() -> dict:
    """Parámetros ΛCDM de referencia (Planck 2018 + sin DEE)."""
    return {
        'h':          0.6736,
        'omega_b':    0.02237,
        'omega_cdm':  0.1200,
        'A_s':        2.1e-9,
        'n_s':        0.9649,
        'tau_reio':   0.0544,
        'sigma8_0':   0.8111,
    }

def make_output_dir(name: str) -> Path:
    p = Path(name); p.mkdir(exist_ok=True); return p

# ════════════════════════════════════════════════════════
# FUNCIONES DE FÍSICA DEE
# ════════════════════════════════════════════════════════
def mu_benchmark1(k_h: float, a: float,
                  mu0: float, kc: float) -> float:
    """G_eff/G = 1 - mu0 * k² / (k² + kc²)   (Benchmark 1)"""
    return 1.0 - mu0 * k_h**2 / (k_h**2 + kc**2)

def mu_benchmark2(k_h: float, a: float,
                  mu0: float, mu1: float,
                  kc0: float, kc1: float) -> float:
    """
    G_eff/G con evolución temporal (Benchmark 2):
      mu(a) = mu0 + mu1*(1-a)
      kc(a) = kc0*(1 + kc1*(1-a))
    """
    mu_a  = mu0  + mu1  * (1.0 - a)
    kc_a  = kc0  * (1.0 + kc1 * (1.0 - a))
    return 1.0 - mu_a * k_h**2 / (k_h**2 + kc_a**2)

def w_dee(a: float, w0: float, wa: float) -> float:
    return w0 + wa * (1.0 - a)

# ════════════════════════════════════════════════════════
# SOLVER INTERNO (fallback sin classy)
# ════════════════════════════════════════════════════════
class _InternalSolver:
    """Solver cosmológico interno — válido en z < 2."""

    def __init__(self, p: dict):
        h = p['h']; H0 = h * 100.0
        self.H0 = H0
        self.Om = (p['omega_b'] + p['omega_cdm']) / h**2
        self.Or = 9.1e-5
        self.w0 = p.get('dee_w0', -1.0)
        self.wa = p.get('dee_wa',  0.0)
        self.s8 = p.get('sigma8_0', p.get('sigma8', 0.8111))
        self.Oq = 1.0 - self.Om - self.Or
        self._build_rho_q()

    def _build_rho_q(self):
        lna_arr = np.linspace(-12, 0, 2000)
        w_arr   = np.array([w_dee(np.exp(x), self.w0, self.wa) for x in lna_arr])
        expon   = -3.0 * np.cumsum(1.0 + w_arr) * (lna_arr[1] - lna_arr[0])
        expon  -= expon[-1]
        rho_ratio = np.exp(expon)
        self._rhoq = interp1d(lna_arr, rho_ratio,
                              bounds_error=False, fill_value='extrapolate')

    def rhoq(self, a): return self.Oq * self._rhoq(np.log(a))

    def E2(self, a):
        return self.Om*a**-3 + self.Or*a**-4 + self.rhoq(a)

    def H(self, a): return self.H0 * np.sqrt(max(self.E2(a), 1e-20))

    def Om_a(self, a): return self.Om * a**-3 / self.E2(a)

    def dlogH_dlna(self, a):
        da = a * 1e-5
        return a / (2*self.E2(a)) * (self.E2(a+da) - self.E2(a-da)) / (2*da)

    def solve_growth(self, mu_func=None, k_h=0.15, a_ini=1e-4):
        if mu_func is None: mu_func = lambda k,a: 1.0

        def rhs(lna, y):
            a  = np.exp(lna)
            d, dp = y
            hh = 2.0 + self.dlogH_dlna(a)
            src = 1.5 * self.Om_a(a) * mu_func(k_h, a)
            return [dp, src*d - hh*dp]

        y0 = [a_ini, a_ini]
        lna_span = (np.log(a_ini), 0.0)
        lna_ev   = np.linspace(*lna_span, 600)
        sol = solve_ivp(rhs, lna_span, y0, t_eval=lna_ev,
                        method='DOP853', rtol=1e-9, atol=1e-12)
        return np.exp(sol.t), sol.y[0], sol.y[1]

    def fsigma8(self, z_arr, mu_func=None, k_h=0.15):
        a_sol, d_sol, dp_sol = self.solve_growth(mu_func, k_h)
        d_itp  = interp1d(a_sol, d_sol,  bounds_error=False, fill_value='extrapolate')
        dp_itp = interp1d(a_sol, dp_sol, bounds_error=False, fill_value='extrapolate')
        a_arr  = 1.0/(1.0 + z_arr)
        d_a    = d_itp(a_arr)
        d_1    = d_itp(1.0)
        f_a    = dp_itp(a_arr) / np.maximum(d_a, 1e-30)
        s8_a   = self.s8 * d_a / d_1
        return f_a * s8_a

    def Pk_approx(self, k_h_arr, z=0.0, mu_func=None):
        """P(k) aproximado (Eisenstein-Hu 1998 + crecimiento)."""
        h  = self.H0 / 100.0
        Om = self.Om; Ob = 0.02237/h**2; ns = 0.9649
        T   = 2.725; zeq = 2.5e4 * Om * h**2 * (T/2.7)**-4
        keq = 7.46e-2 * Om * h**2 * (T/2.7)**-2
        b1  = 0.313*(Om*h**2)**-0.419*(1+0.607*(Om*h**2)**0.674)
        b2  = 0.238*(Om*h**2)**0.223
        zd  = 1291*(Om*h**2)**0.251/(1+0.659*(Om*h**2)**0.828)*(1+b1*(Ob*h**2)**b2)
        Req = 31.5e3*Ob*h**2*(T/2.7)**-4/(zeq)
        Rd  = 31.5e3*Ob*h**2*(T/2.7)**-4/zd
        s   = 2/(3*keq)*np.sqrt(6/Req)*np.log(
            (np.sqrt(1+Rd)+np.sqrt(Rd+Req))/(1+np.sqrt(Req)))
        ksilk = 1.6*(Ob*h**2)**0.52*(Om*h**2)**0.01*zd**(-0.52)
        q_arr = k_h_arr/(13.41*keq)
        C0    = 14.2 + 731/(1+62.5*q_arr)
        T0    = np.log(np.e+1.8*q_arr)/(np.log(np.e+1.8*q_arr)+C0*q_arr**2)
        f_arr = 1/(1+(k_h_arr*s/5.4)**4)
        Tb    = (T0/(1+(k_h_arr*s/6)**3) +
                 f_arr*T0*np.exp(-(k_h_arr/ksilk)**1.4))
        Tc    = T0
        T_arr = Ob/Om*Tb + (Om-Ob)/Om*Tc
        Pk0   = 2*np.pi**2 * 2e-9 / (k_h_arr**3) * (k_h_arr*2998)**ns * T_arr**2
        a_z   = 1/(1+z)
        a_sol, d_sol, _ = self.solve_growth(mu_func)
        d_itp = interp1d(a_sol, d_sol, bounds_error=False, fill_value='extrapolate')
        D     = d_itp(a_z)/d_itp(1.0)
        return Pk0 * D**2

# ════════════════════════════════════════════════════════
# run_class_model — núcleo del pipeline
# ════════════════════════════════════════════════════════
def _mu_func_from_params(p: dict):
    """Construye la función mu(k,a) según los parámetros DEE."""
    if p.get('dee_model', 'no') != 'yes':
        return lambda k, a: 1.0
    if 'dee_mu1' in p:  # Benchmark 2
        return lambda k, a: mu_benchmark2(
            k, a,
            float(p['dee_mu0']), float(p['dee_mu1']),
            float(p['dee_kc']),  float(p['dee_kc1']))
    else:  # Benchmark 1
        return lambda k, a: mu_benchmark1(
            k, a, float(p['dee_mu0']), float(p['dee_kc']))

def run_class_model(params: dict, name: str,
                    z_grid: np.ndarray,
                    k_h_grid: np.ndarray) -> dict:
    """
    Corre el modelo (CLASS real o solver interno).
    Devuelve dict con background, growth, Pk.
    """
    print(f"  Corriendo: {name} ({'CLASS' if HAS_CLASSY else 'solver interno'})...")
    mu_func = _mu_func_from_params(params)
    h = params.get('h', 0.6736)

    if HAS_CLASSY:
        return _run_with_class(params, name, z_grid, k_h_grid, mu_func, h)
    else:
        return _run_internal(params, name, z_grid, k_h_grid, mu_func, h)


def _run_with_class(params, name, z_grid, k_h_grid, mu_func, h):
    c_params = {
        'h':            params['h'],
        'omega_b':      params['omega_b'],
        'omega_cdm':    params['omega_cdm'],
        'A_s':          params.get('A_s', 2.1e-9),
        'n_s':          params.get('n_s', 0.9649),
        'tau_reio':     params.get('tau_reio', 0.0544),
        'output':        'mPk,tCl,lCl',
        'P_k_max_h/Mpc':  k_h_grid.max() * 1.5,
        'z_pk':           ','.join([str(round(z,2)) for z in [0.0, 0.5, 1.0]]),
        'lensing':        'yes',
        'l_max_scalars':  2500,
        'modes':          's',
    }
    if params.get('dee_model') == 'yes':
        c_params['w0_fld'] = float(params.get('dee_w0', -1.0))
        c_params['wa_fld'] = float(params.get('dee_wa',  0.0))
        c_params['Omega_Lambda'] = 0

    cosmo = _Class()
    cosmo.set(c_params)
    cosmo.compute()

    # Background
    bg = cosmo.get_background()
    z_bg = bg['z']
    H_bg = bg['H [1/Mpc]'] * 299792.458  # → km/s/Mpc
    chi_bg = bg['comov. dist.']
    H_itp  = interp1d(z_bg, H_bg,  bounds_error=False, fill_value='extrapolate')
    chi_itp= interp1d(z_bg, chi_bg, bounds_error=False, fill_value='extrapolate')

    # Growth: f*sigma8 via sigma(R,z)
    s8 = params.get('sigma8_0', 0.8111)
    fsig8 = []
    for z in z_grid:
        if z < 0.01: z = 0.01
        try:
            Om_z = (params['omega_b']+params['omega_cdm'])/h**2*(1+z)**3
            Om_z /= (H_itp(z)/H_itp(0))**2
            s8z   = cosmo.sigma(8.0/h, z)
            fsig8.append(Om_z**0.55 * s8z)
        except Exception:
            fsig8.append(np.nan)

    # P(k) at z=0 and z=0.5
    Pk_z0  = np.array([cosmo.pk(k*h, 0.0) * h**3 for k in k_h_grid])
    Pk_z05 = np.array([cosmo.pk(k*h, 0.5) * h**3 for k in k_h_grid])
    Pk_z1  = np.array([cosmo.pk(k*h, 1.0) * h**3 for k in k_h_grid])

    cosmo.struct_cleanup(); cosmo.empty()

    return {
        'name': name, 'params': params, 'h': h,
        'z_bg': z_grid, 'H': H_itp(z_grid), 'chi': chi_itp(z_grid),
        'z_growth': z_grid, 'fsigma8': np.array(fsig8),
        'k_h': k_h_grid, 'Pk_z0': Pk_z0, 'Pk_z05': Pk_z05, 'Pk_z1': Pk_z1,
        'mode': 'CLASS',
    }


def _run_internal(params, name, z_grid, k_h_grid, mu_func, h):
    sol = _InternalSolver(params)
    H_arr   = np.array([sol.H(1/(1+z)) for z in z_grid])
    chi_arr = np.array([_comoving_dist(sol, z) for z in z_grid])
    fs8     = sol.fsigma8(z_grid, mu_func)
    Pk_z0   = sol.Pk_approx(k_h_grid, z=0.0, mu_func=mu_func)
    Pk_z05  = sol.Pk_approx(k_h_grid, z=0.5, mu_func=mu_func)
    Pk_z1   = sol.Pk_approx(k_h_grid, z=1.0, mu_func=mu_func)
    return {
        'name': name, 'params': params, 'h': h,
        'z_bg': z_grid, 'H': H_arr, 'chi': chi_arr,
        'z_growth': z_grid, 'fsigma8': fs8,
        'k_h': k_h_grid, 'Pk_z0': Pk_z0, 'Pk_z05': Pk_z05, 'Pk_z1': Pk_z1,
        'mode': 'internal',
    }

def _comoving_dist(sol, z_max, npts=400):
    if z_max <= 0: return 0.0
    z_int = np.linspace(0, z_max, npts)
    c_km  = 299792.458
    integrand = c_km / np.array([sol.H(1/(1+z)) for z in z_int])
    return np.trapezoid(integrand, z_int)

# ════════════════════════════════════════════════════════
# GUARDAR TABLAS
# ════════════════════════════════════════════════════════
def save_tables(res: dict, prefix: Path):
    prefix = Path(prefix); prefix.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'z': res['z_bg'], 'H': res['H'], 'chi': res['chi']}
                 ).to_csv(prefix / 'background.csv', index=False)
    pd.DataFrame({'z': res['z_growth'], 'fsigma8': res['fsigma8']}
                 ).to_csv(prefix / 'growth.csv', index=False)
    pd.DataFrame({'k_h': res['k_h'], 'Pk_z0': res['Pk_z0'],
                  'Pk_z05': res['Pk_z05'], 'Pk_z1': res['Pk_z1']}
                 ).to_csv(prefix / 'pk.csv', index=False)
    print(f"    Tablas guardadas en {prefix}/")

def save_summary_table(results: list, path: Path):
    rows = []
    for r in results:
        w0  = r['params'].get('dee_w0', -1.0)
        wa  = r['params'].get('dee_wa',  0.0)
        mu0 = r['params'].get('dee_mu0', 0.0)
        kc  = r['params'].get('dee_kc',  0.0)
        fs8_z0 = float(np.interp(0.0,  r['z_growth'], r['fsigma8']))
        fs8_z1 = float(np.interp(1.0,  r['z_growth'], r['fsigma8']))
        rows.append({'model': r['name'], 'mode': r['mode'],
                     'w0': w0, 'wa': wa, 'mu0': mu0, 'kc': kc,
                     'fsigma8_z0': round(fs8_z0,4),
                     'fsigma8_z1': round(fs8_z1,4)})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    Tabla resumen → {path}")
    for r in rows:
        print(f"      {r['model']:30s} fσ₈(z=0)={r['fsigma8_z0']}  fσ₈(z=1)={r['fsigma8_z1']}")

# ════════════════════════════════════════════════════════
# GRÁFICOS
# ════════════════════════════════════════════════════════
COLORS = [CB, CY, CG, CR, '#e67e22', '#9b59b6']
LINES  = ['-', '--', '-.', ':', (0,(3,1,1,1))]

def _ax_style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR, labelsize=9)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True, alpha=0.15)

def _fig():
    f = plt.figure(facecolor='#0a0a1a'); return f

def plot_background(results: list, out_file: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a1a')
    for ax in (ax1, ax2): _ax_style(ax)
    for i, r in enumerate(results):
        col, ls = COLORS[i], LINES[i]
        ax1.plot(r['z_bg'], r['H'],   color=col, ls=ls, lw=2.5, label=r['name'])
        ax2.plot(r['z_bg'], r['chi'], color=col, ls=ls, lw=2.5, label=r['name'])
    ax1.set_xlabel('z', fontsize=10, color=CW); ax1.set_ylabel('H(z) [km/s/Mpc]', fontsize=10, color=CW)
    ax1.set_title('Historia de expansión H(z)', fontsize=10, fontweight='bold', color=CY)
    ax1.legend(fontsize=8, facecolor=BG, labelcolor=CW)
    ax2.set_xlabel('z', fontsize=10, color=CW); ax2.set_ylabel('χ(z) [Mpc]', fontsize=10, color=CW)
    ax2.set_title('Distancia comóvil χ(z)', fontsize=10, fontweight='bold', color=CY)
    ax2.legend(fontsize=8, facecolor=BG, labelcolor=CW)
    fig.suptitle('Background cosmológico: DEE vs ΛCDM', fontsize=11, fontweight='bold', color=CW)
    plt.tight_layout(); plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(); print(f"    → {out_file}")

def plot_growth(results: list, out_file: Path):
    # Datos fσ₈ observacionales (subset representativo)
    obs = np.array([[0.02,0.428,0.047],[0.10,0.37,0.13],[0.15,0.490,0.145],
                    [0.25,0.351,0.058],[0.32,0.427,0.056],[0.38,0.497,0.045],
                    [0.50,0.427,0.043],[0.57,0.426,0.029],[0.61,0.436,0.034],
                    [0.73,0.437,0.072],[0.80,0.470,0.080],[1.40,0.482,0.116]])
    fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor('#0a0a1a'); _ax_style(ax)
    ax.errorbar(obs[:,0], obs[:,1], yerr=obs[:,2],
                fmt='o', color='white', ms=6, alpha=0.8, capsize=3, label='Datos fσ₈', zorder=5)
    for i, r in enumerate(results):
        ax.plot(r['z_growth'], r['fsigma8'], color=COLORS[i], ls=LINES[i],
                lw=2.5, label=f"{r['name']} ({r['mode']})")
    ax.set_xlabel('z', fontsize=10, color=CW); ax.set_ylabel('fσ₈(z)', fontsize=10, color=CW)
    ax.set_xlim(0, 2); ax.set_ylim(0.2, 0.65)
    ax.set_title('Tasa de crecimiento fσ₈(z)\nvs datos de redshift surveys', fontsize=10, fontweight='bold', color=CG)
    ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)
    plt.tight_layout(); plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(); print(f"    → {out_file}")

def plot_pk_ratio(res_ref: dict, res_dee: dict, out_file: Path, z_for_ratio: float = 0.5):
    Pk_ref = res_ref['Pk_z05'] if z_for_ratio < 0.75 else res_ref['Pk_z1']
    Pk_dee = res_dee['Pk_z05'] if z_for_ratio < 0.75 else res_dee['Pk_z1']
    ratio  = Pk_dee / np.maximum(Pk_ref, 1e-30)
    fig, ax = plt.subplots(figsize=(9, 5)); fig.patch.set_facecolor('#0a0a1a'); _ax_style(ax)
    ax.semilogx(res_ref['k_h'], ratio, color=CY, lw=2.5)
    ax.axhline(1.0, color='white', lw=1, ls=':', alpha=0.5, label='ΛCDM ref')
    ax.fill_between(res_ref['k_h'], ratio, 1.0, alpha=0.2, color=CY)
    ax.set_xlabel('k (h/Mpc)', fontsize=10, color=CW)
    ax.set_ylabel(f'P_DEE(k) / P_ΛCDM(k)  [z={z_for_ratio}]', fontsize=10, color=CW)
    ax.set_title(f'Ratio P(k): {res_dee["name"]} / {res_ref["name"]}\nz={z_for_ratio}',
                 fontsize=10, fontweight='bold', color=CY)
    plt.tight_layout(); plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(); print(f"    → {out_file}")

def plot_mu_curves_benchmark1(mu0: float, kc: float, out_file: Path):
    k_arr = np.logspace(-3, 1, 300)
    fig, ax = plt.subplots(figsize=(9, 5)); fig.patch.set_facecolor('#0a0a1a'); _ax_style(ax)
    for a, col, lab in [(1.0, CG, 'z=0'), (0.667, CY, 'z=0.5'), (0.5, CR, 'z=1')]:
        mu_vals = np.array([mu_benchmark1(k, a, mu0, kc) for k in k_arr])
        ax.semilogx(k_arr, mu_vals, color=col, lw=2.5, label=lab)
    ax.axhline(1.0, color='white', lw=1, ls=':', alpha=0.5)
    ax.axvline(kc, color='white', lw=1, ls='--', alpha=0.4, label=f'kc={kc} h/Mpc')
    ax.set_xlabel('k (h/Mpc)', fontsize=10, color=CW)
    ax.set_ylabel('G_eff(k,a)/G', fontsize=10, color=CW)
    ax.set_title(f'Benchmark 1: G_eff = 1 − μ₀k²/(k²+kc²)\nμ₀={mu0}, kc={kc} h/Mpc',
                 fontsize=10, fontweight='bold', color=CY)
    ax.legend(fontsize=9, facecolor=BG, labelcolor=CW)
    plt.tight_layout(); plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(); print(f"    → {out_file}")

def plot_mu_curves_benchmark2(mu0: float, mu1: float,
                               kc0: float, kc1: float,
                               out_file: Path):
    k_arr = np.logspace(-3, 1, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0a0a1a')
    for ax in (ax1, ax2): _ax_style(ax)

    a_vals = [(1.0, CG, 'z=0'), (0.667, CY, 'z=0.5'), (0.5, CR, 'z=1'), (0.333, '#9b59b6', 'z=2')]
    for a, col, lab in a_vals:
        mu_vals = np.array([mu_benchmark2(k, a, mu0, mu1, kc0, kc1) for k in k_arr])
        ax1.semilogx(k_arr, mu_vals, color=col, lw=2.5, label=lab)

    ax1.axhline(1.0, color='white', lw=1, ls=':', alpha=0.5)
    ax1.set_xlabel('k (h/Mpc)', fontsize=10, color=CW)
    ax1.set_ylabel('G_eff(k,a)/G', fontsize=10, color=CW)
    ax1.set_title('G_eff(k,a) para distintos redshifts', fontsize=10, fontweight='bold', color=CY)
    ax1.legend(fontsize=9, facecolor=BG, labelcolor=CW)

    a_arr = np.linspace(0.2, 1.0, 200)
    mu_k1 = np.array([mu_benchmark2(0.10, a, mu0, mu1, kc0, kc1) for a in a_arr])
    mu_k2 = np.array([mu_benchmark2(0.30, a, mu0, mu1, kc0, kc1) for a in a_arr])
    ax2.plot(1/a_arr - 1, mu_k1, color=CY, lw=2.5, label='k=0.1 h/Mpc')
    ax2.plot(1/a_arr - 1, mu_k2, color=CG, lw=2.5, label='k=0.3 h/Mpc')
    ax2.axhline(1.0, color='white', lw=1, ls=':', alpha=0.5)
    ax2.set_xlabel('z', fontsize=10, color=CW)
    ax2.set_ylabel('G_eff(k,a)/G', fontsize=10, color=CW)
    ax2.set_title('Evolución temporal de G_eff(z)', fontsize=10, fontweight='bold', color=CG)
    ax2.legend(fontsize=9, facecolor=BG, labelcolor=CW)

    fig.suptitle(f'Benchmark 2: μ(a)=μ₀+μ₁(1−a), kc(a)=kc₀(1+kc₁(1−a))\n'
                 f'μ₀={mu0}, μ₁={mu1}, kc₀={kc0}, kc₁={kc1}',
                 fontsize=11, fontweight='bold', color=CW)
    plt.tight_layout(); plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(); print(f"    → {out_file}")
