"""
dee_temporal_runner_v2.py
══════════════════════════════════════════════════════════════
Versión compatible con classy estándar (pip install classy).
No requiere patch DEE compilado en CLASS.

Estrategia:
  · Background DEE: usa w0_fld/wa_fld nativos de CLASS
  · G_eff(k,a):     se aplica en post-proceso sobre D(z) de CLASS
  · σ₈(z), fσ₈(z): recalculadas con el factor de crecimiento
    modificado por G_eff

Esto es físicamente correcto en el régimen subhorizonte lineal:
G_eff modifica D(z) pero no el espectro de transferencia T(k).
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from classy import Class

OUT = Path("temporal_outputs"); OUT.mkdir(exist_ok=True)

# ── Parámetros base (CLASS nativo) ───────────────────────
BASE = {
    "output":          "mPk,tCl,lCl",
    "lensing":         "yes",
    "modes":           "s",
    "P_k_max_1/Mpc":   5.0,
    "z_max_pk":        4.0,
    "h":               0.67,
    "omega_b":         0.0224,
    "omega_cdm":       0.12,
    "A_s":             2.1e-9,
    "n_s":             0.965,
    "tau_reio":        0.054,
}
# DEE background: w0/wa nativos de CLASS
DEE_BG = {**BASE,
    "w0_fld":       -0.98,
    "wa_fld":        0.05,
    "Omega_Lambda":  0,
}

# ── G_eff parametrizaciones ──────────────────────────────
def mu_b1(k, a):
    """B1: G_eff/G = 1 − μ₀k²/(k²+kc²) — constante en a"""
    return 1.0 - 0.08*k**2/(k**2 + 0.07**2)

def mu_b2(k, a):
    """B2: G_eff con evolución temporal μ(a)=μ₀+μ₁(1−a)"""
    mu = 0.06 + 0.04*(1-a)
    kc = 0.07*(1 + 0.30*(1-a))
    return 1.0 - mu*k**2/(k**2 + kc**2)

# ── Obtener H(z) y Ω_m(z) desde CLASS ───────────────────
def get_class_background(params, label):
    print(f"  CLASS corriendo {label}...")
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    bg = cosmo.get_background()
    z_bg = bg['z']
    H_bg = bg['H [1/Mpc]'] * 299792.458   # → km/s/Mpc
    h    = cosmo.h()
    Om0  = cosmo.Omega_m()
    sig8_class = cosmo.sigma(8.0/h, 0.0)   # σ₈ CLASS en z=0

    H_itp = interp1d(z_bg, H_bg,  bounds_error=False, fill_value='extrapolate')
    cosmo.struct_cleanup(); cosmo.empty()
    print(f"    H(0)={H_itp(0):.2f} km/s/Mpc  Ω_m={Om0:.4f}  σ₈(CLASS,z=0)={sig8_class:.4f}")
    return H_itp, Om0, sig8_class, h

# ── Solver de crecimiento con G_eff + H(z) de CLASS ──────
def solve_growth_with_Geff(H_itp, Om0, mu_func, k_h=0.15, s8_0=None):
    """
    Usa H(z) exacto de CLASS + G_eff en post-proceso.
    Resuelve: δ'' + (2+H'/H)δ' − 3/2 Ω_m(a) μ(k,a) δ = 0
    Variable: N = ln a
    """
    H0 = H_itp(0)

    def E2(a):
        z = 1/a - 1
        return (H_itp(z)/H0)**2

    def dlogH(a):
        da = a*1e-5
        z0 = 1/a-1; zp = 1/(a+da)-1; zm = 1/(a-da)-1
        H0v = H_itp(z0); Hp = H_itp(max(0,zp)); Hm = H_itp(zm)
        return a/(2*E2(a)) * ((Hp/H0)**2 - (Hm/H0)**2) / (2*da)

    def Om_a(a):
        z = 1/a - 1
        return Om0*(1+z)**3 / E2(a)

    def rhs(ln, y):
        a = np.exp(ln)
        d, dp = y
        return [dp, 1.5*Om_a(a)*mu_func(k_h,a)*d - (2+dlogH(a))*dp]

    ev  = np.linspace(np.log(1e-4), 0, 1200)
    sol = solve_ivp(rhs, (ev[0], 0), [1e-4, 1e-4], t_eval=ev,
                    method='DOP853', rtol=1e-10, atol=1e-13)
    a_s = np.exp(sol.t); d_s = sol.y[0]; dp_s = sol.y[1]

    d_i = interp1d(a_s, d_s,  bounds_error=False, fill_value='extrapolate')
    f_i = interp1d(a_s, dp_s/np.maximum(d_s,1e-30),
                   bounds_error=False, fill_value='extrapolate')
    d1  = d_i(1.0)

    # Normalizar σ₈ al valor de CLASS en z=0
    if s8_0 is None: s8_0 = 0.8111

    D   = lambda z: d_i(1/(1+z))/d1
    f   = lambda z: f_i(1/(1+z))
    sig8= lambda z: s8_0 * D(z)
    fs8 = lambda z: f(z) * sig8(z)
    S8  = lambda z: sig8(z) * np.sqrt(Om0/0.3)
    return D, f, sig8, fs8, S8

# ── Función derivada logarítmica ─────────────────────────
def dlog_dz(y, z):
    y = np.array(y, dtype=float); z = np.array(z, dtype=float)
    d = np.zeros_like(y)
    for i in range(1, len(z)-1):
        d[i] = (np.log(y[i+1])-np.log(y[i-1]))/(z[i+1]-z[i-1])
    d[0]  = (np.log(y[1]) -np.log(y[0])) /(z[1]-z[0])
    d[-1] = (np.log(y[-1])-np.log(y[-2]))/(z[-1]-z[-2])
    return d

def detectability(o1, o2, sigma=0.02):
    return float(np.sqrt(np.sum(((o1-o2)/sigma)**2)))

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
print("="*62)
print("  DEE Temporal Observables — CLASS real + G_eff post-proceso")
print("="*62)

Z = np.linspace(0.01, 2.0, 120)

# 1. Obtener background DEE desde CLASS
H_itp, Om0, sig8_class, h = get_class_background(DEE_BG, "DEE background")

# 2. Resolver crecimiento para 3 modelos usando H(z) de CLASS
print("\nResolviendo crecimiento con G_eff post-proceso...")
models_raw = {}
for label, mu_func in [
    ("LCDM", lambda k,a: 1.0),
    ("B1",   mu_b1),
    ("B2",   mu_b2),
]:
    D,f,sig8,fs8,S8 = solve_growth_with_Geff(H_itp, Om0, mu_func,
                                               s8_0=sig8_class)
    sig8_arr = np.array([sig8(z) for z in Z])
    fs8_arr  = np.array([fs8(z)  for z in Z])
    S8_arr   = np.array([S8(z)   for z in Z])
    models_raw[label] = dict(sigma8_z=sig8_arr, fs8_z=fs8_arr, S8_z=S8_arr)
    df = pd.DataFrame({"z":Z,"sigma8_z":sig8_arr,"S8_z":S8_arr,"f_sigma8_z":fs8_arr})
    df.to_csv(OUT/f"{label}_table.csv", index=False)
    print(f"  {label}: σ₈(0)={sig8_arr[0]:.4f}  "
          f"fσ₈(0.5)={np.interp(0.5,Z,fs8_arr):.4f}  "
          f"S₈(0)={S8_arr[0]:.4f}")

# 3. Observables temporales
print("\nCalculando G_t y F_t...")
temporal = {}
for name, m in models_raw.items():
    temporal[name] = {
        "G_t": dlog_dz(m["sigma8_z"], Z),
        "F_t": dlog_dz(m["fs8_z"],    Z),
    }

# 4. Detectabilidad
print("\n=== DETECTABILIDAD B2 vs B1 (CLASS H(z) + G_eff post-proceso) ===\n")
b1_t = temporal["B1"]
det_results = {}
for lbl, r in temporal.items():
    if lbl == "B1": continue
    dG = detectability(b1_t["G_t"], r["G_t"])
    dF = detectability(b1_t["F_t"], r["F_t"])
    det_results[lbl] = (dG, dF)
    flag = "✓ DETECTABLE" if max(dG,dF) > 2 else "✗ indetectable"
    print(f"  {lbl:6s}: G_t={dG:.3f}σ   F_t={dF:.3f}σ   → {flag}")

pd.DataFrame([
    {"modelo": k, "G_t_vs_B1": v[0], "F_t_vs_B1": v[1],
     "max_sigma": max(v), "detectable_2sigma": max(v)>2}
    for k,v in det_results.items()
]).to_csv(OUT/"detectability_table.csv", index=False)

# 5. Gráfico
BG='#0d1117'; CW='#ecf0f1'; CY='#f1c40f'; CB='#2980b9'
CG='#27ae60'; CR='#e74c3c'; CGR='#7f8c8d'
COLS = {"LCDM":CB, "B1":CY, "B2":CG}
LSTY = {"LCDM":"-", "B1":"--", "B2":":"}

fig, axes = plt.subplots(2, 3, figsize=(16,10))
fig.patch.set_facecolor('#0a0a1a')

def sty(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR, labelsize=9)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True, alpha=0.15)

titles = ['σ₈(z)','fσ₈(z)','S₈(z)']
keys   = ['sigma8_z','fs8_z','S8_z']
ylbls  = ['σ₈(z)','fσ₈(z)','S₈(z)']
for i,(k,t,yl) in enumerate(zip(keys,titles,ylbls)):
    ax=axes[0,i]; sty(ax)
    for lbl,m in models_raw.items():
        ax.plot(Z, m[k], LSTY[lbl], color=COLS[lbl], lw=2.5, label=lbl)
    if k=='S8_z':
        ax.axhspan(0.74,0.78,alpha=0.15,color='cyan',label='S8 tensión ~0.76')
    ax.set_xlim(0,2); ax.set_xlabel('z',color=CW)
    ax.set_ylabel(yl,color=CW)
    ax.set_title(f'{t}\n[CLASS H(z) + G_eff post-proceso]',
                 fontweight='bold',color=list(COLS.values())[i])
    ax.legend(fontsize=9,facecolor=BG,labelcolor=CW)

for i,(obs,col,ttl) in enumerate(zip(['G_t','F_t'],
                                      [CY,CG],
                                      ['G_t = d ln σ₈/dz','F_t = d ln fσ₈/dz'])):
    ax=axes[1,i]; sty(ax)
    for lbl,r in temporal.items():
        ax.plot(Z, r[obs], LSTY[lbl], color=COLS[lbl], lw=2.5, label=lbl)
    ax.set_xlim(0.1,2); ax.set_xlabel('z',color=CW)
    ax.set_ylabel(obs,color=CW)
    ax.set_title(f'{ttl}\n★ CLASS real ★', fontweight='bold', color=col)
    ax.legend(fontsize=9,facecolor=BG,labelcolor=CW)

# Panel 6: tabla de resultados
ax=axes[1,2]; ax.axis('off'); ax.set_facecolor('#0a0e1a')
y=0.92; dy=0.16
for j,hdr in enumerate(['Modelo','G_t [σ]','F_t [σ]','Detectable?']):
    ax.text(0.03+j*0.24,y,hdr,transform=ax.transAxes,
            fontsize=10,color=CY,fontweight='bold',va='top')
y-=0.08
ax.plot([0.01,0.99],[y+0.02,y+0.02],color=CGR,lw=0.5,transform=ax.transAxes)
for lbl,(dG,dF) in det_results.items():
    det = max(dG,dF)
    col_det = CG if det>2 else CR
    for j,(cell,col) in enumerate(zip(
        [lbl, f'{dG:.2f}σ', f'{dF:.2f}σ', '✓' if det>2 else '✗'],
        [CW, CY if dG>2 else CR, CG if dF>2 else CR, col_det]
    )):
        ax.text(0.03+j*0.24,y,cell,transform=ax.transAxes,
                fontsize=10,color=col,va='top',
                fontweight='bold' if j in (1,2,3) else 'normal')
    y-=dy
ax.text(0.5,0.04,
    '★ Resultado con CLASS real (H(z)) + G_eff post-proceso\n'
    'σ=2% por punto de redshift (precisión Euclid)\n'
    'Verde ≥2σ detectable  |  Rojo <2σ indetectable',
    transform=ax.transAxes,ha='center',va='bottom',fontsize=8.5,color=CW,
    bbox=dict(facecolor='#0d1117',alpha=0.9,edgecolor=CY))

fig.suptitle(
    'DEE — G_t(z) y F_t(z): ¿rompen la degeneración B1 vs B2?\n'
    'CLASS real (H(z)) + corrección G_eff en post-proceso',
    fontsize=13,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig(OUT/'temporal_class_real.png',dpi=150,
            bbox_inches='tight',facecolor='#0a0a1a')

print(f"\n  → {OUT}/temporal_class_real.png")
print(f"  → {OUT}/detectability_table.csv")
print(f"\nPara descargar:")
print(f"  import shutil; shutil.make_archive('temporal_outputs','zip','temporal_outputs')")
print(f"  files.download('temporal_outputs.zip')")
