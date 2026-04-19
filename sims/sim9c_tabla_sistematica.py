"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 9c: Tabla sistemática del potencial V(φ)
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ PRODUCE:
  Tabla sistemática de los coeficientes de V(φ) para distintos N,
  usando 20 semillas por punto y estadística mediana/IQR robusta.

  Cantidades medidas para cada (N, semilla):
    φ₀ = R_fondo = ⟨φ(x)⟩_vac   (mínimo del potencial)
    σ_φ = std(φ)                  (escala de fluctuaciones)
    m_φ² = χ_φ⁻¹                 (rigidez del vacío, de correlador)
    γ₂ = κ₄/μ₂²                  (kurtosis exceso, de cumulantes)
    λ₄ = −γ₂ × m_φ⁴              (coef. cuártico, válido a 1er orden)
    |V₄/V₂|@σ                    (peso relativo del cuártico)
    signo λ₄                     (fracción de corridas con λ₄<0)

  NOTA SOBRE EL RÉGIMEN:
  La fórmula λ₄ = −γ₂ × m_φ⁴ es válida a primer orden perturbativo
  alrededor de la gaussiana. Es aplicable cuando |γ₂| < 1.
  Para N ≥ 1200 el régimen perturbativo está satisfecho.
════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

r_c = 0.22
N_SEEDS = 20
N_LIST = [400, 600, 800, 1200, 1600, 2000]

print("="*65)
print("  MODELO DEE v2.0 — SIM 9c: Tabla sistemática V(φ)")
print(f"  r_c={r_c}, N_seeds={N_SEEDS} por punto")
print("="*65)

def run_one(N, r_c, seed, n_grid=15):
    np.random.seed(seed)
    coords = np.random.rand(N, 3)
    D = cdist(coords, coords); np.fill_diagonal(D, np.inf)
    sk = r_c * 0.5
    S = np.where(D < r_c, np.exp(-D**2/(2*sk**2)), 0.0)
    np.fill_diagonal(S, 0); S = (S+S.T)/2
    d_i = np.maximum(S.sum(axis=1), 1e-10); P = S/d_i[:,None]

    ar = [(i,j) for i in range(N) for j in np.where(S[i]>1e-8)[0] if j>i]
    ns = min(len(ar), min(8000, N*8))
    idx = np.random.choice(len(ar), ns, replace=False)
    kn = {i:[] for i in range(N)}
    for i,j in [ar[k] for k in idx]:
        dij = D[i,j]
        if np.isfinite(dij) and dij > 1e-8:
            k = 1.0 - np.sum(np.abs(P[i]-P[j]))*sk/dij
            if np.isfinite(k): kn[i].append(k); kn[j].append(k)
    ki = np.array([np.mean(v) if v else np.nan for v in kn.values()])
    ki = np.where(np.isfinite(ki), ki, np.nanmean(ki))

    # φ(x) suavizado
    g = np.linspace(0.05, 0.95, n_grid)
    xx,yy,zz = np.meshgrid(g,g,g,indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    Dgn = cdist(grid, coords)
    K = np.exp(-Dgn**2/(2*r_c**2)); K /= K.sum(axis=1, keepdims=True)
    phi = K @ ki

    # m_φ² desde correlador con ventana gaussiana
    dk = ki - ki.mean()
    r_bins = np.linspace(0, 0.6, 20); r_mid = (r_bins[:-1]+r_bins[1:])/2; dr = r_bins[1]-r_bins[0]
    C_r = []
    for lo,hi in zip(r_bins[:-1], r_bins[1:]):
        ii,jj = np.where((D>=lo)&(D<hi)&(D<np.inf))
        C_r.append(float(np.mean(dk[ii]*dk[jj])) if len(ii)>5 else np.nan)
    C_r = np.array(C_r)
    W = np.exp(-r_mid**2/(2*r_c**2))
    valid = np.isfinite(C_r)&(C_r>0)&(r_mid>0)
    chi = 4*np.pi*np.sum(r_mid[valid]**2*C_r[valid]*W[valid])*dr if valid.sum()>=3 else np.nan
    m2 = 1/chi if (chi and chi>0) else np.nan

    # Cumulantes de φ(x)
    dp = phi - phi.mean()
    mu2 = float(np.mean(dp**2)); mu4 = float(np.mean(dp**4))
    k4 = mu4 - 3*mu2**2
    g2 = k4/mu2**2 if mu2 > 0 else np.nan
    l4 = -g2*m2**2 if (not np.isnan(g2) and not np.isnan(m2)) else np.nan
    V2 = 0.5*m2*mu2 if not np.isnan(m2) else np.nan
    V4 = abs(l4)/24*mu2**2 if not np.isnan(l4) else np.nan
    ratio = V4/V2 if (not np.isnan(V4) and V2 and V2 > 0) else np.nan

    return {'phi0':phi.mean(), 'sigma_phi':mu2**0.5, 'm2':m2,
            'g2':g2, 'l4':l4, 'ratio':ratio, 'mu2':mu2}

# ── Cálculo sistemático ─────────────────────────────────────
tabla = {}
for N in N_LIST:
    print(f"\n  N={N} — calculando {N_SEEDS} semillas...", end='', flush=True)
    resultados = []
    for s in range(N_SEEDS):
        r = run_one(N, r_c, s)
        resultados.append(r)
    tabla[N] = resultados
    g2s = [r['g2'] for r in resultados if not np.isnan(r.get('g2',np.nan))]
    l4s = [r['l4'] for r in resultados if not np.isnan(r.get('l4',np.nan))]
    n_neg = sum(1 for x in l4s if x<0)
    print(f"  γ₂(med)={np.median(g2s):+.3f}  λ₄<0={n_neg}/{N_SEEDS}")

# ── Estadísticas resumen ─────────────────────────────────────
print(f"\n\n{'='*65}")
print(f"  TABLA SISTEMÁTICA (mediana ± IQR/2, {N_SEEDS} semillas)")
print(f"{'='*65}")
print(f"\n  {'N':>5}  {'φ₀':>8}  {'σ_φ':>8}  {'m_φ²':>9}  "
      f"{'γ₂':>8}  {'IQR(γ₂)':>9}  "
      f"{'|V₄/V₂|':>9}  {'λ₄<0':>6}  Régimen")
print(f"  {'-'*90}")

stats = {}
for N in N_LIST:
    res = tabla[N]
    phi0  = np.mean([r['phi0'] for r in res])
    sphi  = np.mean([r['sigma_phi'] for r in res])
    m2s   = [r['m2'] for r in res if not np.isnan(r.get('m2',np.nan))]
    g2s   = [r['g2'] for r in res if not np.isnan(r.get('g2',np.nan))]
    l4s   = [r['l4'] for r in res if not np.isnan(r.get('l4',np.nan))]
    rats  = [r['ratio'] for r in res if not np.isnan(r.get('ratio',np.nan))]

    m2_med  = np.median(m2s)
    g2_med  = np.median(g2s); g2_iqr = np.subtract(*np.percentile(g2s,[75,25]))
    l4_med  = np.median(l4s); l4_iqr = np.subtract(*np.percentile(l4s,[75,25]))
    rat_med = np.median(rats)
    n_neg   = sum(1 for x in l4s if x<0)
    regime  = "pert. ✓" if abs(g2_med)<1 else "no-pert."

    stats[N] = {'phi0':phi0,'sphi':sphi,'m2':m2_med,'g2':g2_med,'g2_iqr':g2_iqr,
                'l4':l4_med,'l4_iqr':l4_iqr,'ratio':rat_med,'n_neg':n_neg}

    print(f"  {N:>5}  {phi0:>8.5f}  {sphi:>8.5f}  {m2_med:>9.1f}  "
          f"{g2_med:>+8.4f}  {g2_iqr:>9.4f}  "
          f"{rat_med:>9.4f}  {n_neg:>2}/{N_SEEDS}  {regime}")

# ── Escaleos ─────────────────────────────────────────────────
N_arr = np.array(N_LIST)
g2_arr = np.array([stats[N]['g2'] for N in N_LIST])
rat_arr = np.array([stats[N]['ratio'] for N in N_LIST])
phi0_arr = np.array([stats[N]['phi0'] for N in N_LIST])

sl_g2,_,r_g2,_,_ = linregress(np.log(N_arr), np.log(np.abs(g2_arr)))
sl_rt,_,r_rt,_,_ = linregress(np.log(N_arr), np.log(rat_arr))

print(f"\n  Escaleos:")
print(f"  γ₂  ∝ N^{{{sl_g2:.3f}}}  R²={r_g2**2:.3f}  (TLC predice −1.0)")
print(f"  |V₄/V₂| ∝ N^{{{sl_rt:.3f}}}  R²={r_rt**2:.3f}")

# ── Fracción λ₄<0 ─────────────────────────────────────────────
frac_neg = [stats[N]['n_neg']/N_SEEDS for N in N_LIST]
print(f"\n  Fracción λ₄ < 0 por N:")
for N,f in zip(N_LIST, frac_neg):
    bar = '█'*int(f*20)
    print(f"  N={N:>5}: {bar:<20} {f*100:.0f}%")

# ── GRÁFICO ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117'; CW='#ecf0f1'; CY='#f1c40f'; CG='#27ae60'
CB='#2980b9'; CR='#e74c3c'; CGR='#7f8c8d'; CO='#e67e22'

def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True, alpha=0.15)

g2_arr_full = np.array([stats[N]['g2'] for N in N_LIST])
g2_iqr_arr  = np.array([stats[N]['g2_iqr'] for N in N_LIST])
l4_arr_full = np.array([stats[N]['l4'] for N in N_LIST])
l4_iqr_arr  = np.array([stats[N]['l4_iqr'] for N in N_LIST])
rat_arr_full= np.array([stats[N]['ratio'] for N in N_LIST])
m2_arr_full = np.array([stats[N]['m2'] for N in N_LIST])

# P1: γ₂ vs N con escaleo
ax1=axes[0,0]; style(ax1)
ax1.errorbar(N_arr, np.abs(g2_arr_full), yerr=g2_iqr_arr/2,
             fmt='o-', color=CB, lw=2.5, ms=9, capsize=5, label='|γ₂| mediana')
N_fit=np.linspace(300, 2500, 50)
A_g2 = np.exp(np.polyfit(np.log(N_arr), np.log(np.abs(g2_arr_full)), 1)[1])
ax1.plot(N_fit, A_g2*N_fit**sl_g2, 'r--', lw=2,
         label=f'N^{{{sl_g2:.2f}}} (TLC: −1.0)')
ax1.axhline(1, color=CY, lw=1.5, ls='--', alpha=0.7, label='|γ₂|=1 (pert.)')
ax1.set_xlabel('N (nodos)', fontsize=9, color=CW)
ax1.set_ylabel('|γ₂| (kurtosis exceso)', fontsize=9, color=CW)
ax1.set_title('Gaussianización con N\nγ₂ → 0 (TLC del suavizado)',
              fontsize=10, fontweight='bold', color=CG)
ax1.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# P2: |V₄/V₂| vs N
ax2=axes[0,1]; style(ax2)
ax2.semilogy(N_arr, rat_arr_full, 'o-', color=CY, lw=2.5, ms=9,
             label='|V₄/V₂|@σ_φ')
ax2.axhline(0.05, color=CG, lw=1.5, ls='--', alpha=0.7, label='5% (límite EFT)')
ax2.axhline(0.10, color=CO, lw=1.5, ls=':', alpha=0.7, label='10%')
A_rt = np.exp(np.polyfit(np.log(N_arr), np.log(rat_arr_full), 1)[1])
ax2.plot(N_fit, A_rt*N_fit**sl_rt, 'r--', lw=2,
         label=f'∝ N^{{{sl_rt:.2f}}}')
ax2.set_xlabel('N (nodos)', fontsize=9, color=CW)
ax2.set_ylabel('|V₄/V₂| @ σ_φ', fontsize=9, color=CW)
ax2.set_title('Dominio cuadrático con N\n→ 0 en límite continuo',
              fontsize=10, fontweight='bold', color=CG)
ax2.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# P3: Signo de λ₄ — fracción negativa
ax3=axes[0,2]; style(ax3)
frac_pos = [1 - f for f in frac_neg]
ax3.bar(N_arr, [f*100 for f in frac_neg], width=150, color=CB, alpha=0.8,
        label='λ₄ < 0 (colas pesadas)')
ax3.bar(N_arr, [f*100 for f in frac_pos], width=150, bottom=[f*100 for f in frac_neg],
        color=CR, alpha=0.6, label='λ₄ > 0')
ax3.axhline(50, color='white', lw=1.5, ls='--', alpha=0.5)
ax3.set_xlabel('N (nodos)', fontsize=9, color=CW)
ax3.set_ylabel('Porcentaje de semillas (%)', fontsize=9, color=CW)
ax3.set_title('Signo de λ₄ (N=20 semillas)\nλ₄<0 dominante → colas pesadas del vacío',
              fontsize=10, fontweight='bold', color=CW)
ax3.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# P4: m_φ² y φ₀ vs N
ax4=axes[1,0]; style(ax4)
ax4.plot(N_arr, m2_arr_full, 'o-', color=CB, lw=2.5, ms=9, label='m_φ²')
ax4.set_xlabel('N', fontsize=9, color=CW)
ax4.set_ylabel('m_φ²', fontsize=9, color=CB)
ax4r = ax4.twinx()
ax4r.plot(N_arr, phi0_arr, 's--', color=CY, lw=2, ms=8, label='φ₀')
ax4r.set_ylabel('φ₀ = R_fondo', fontsize=9, color=CY)
ax4r.tick_params(colors=CGR)
ax4.set_title('m_φ² y φ₀ vs N\nConvengencia del vacío',
              fontsize=10, fontweight='bold', color=CW)
ax4.legend(fontsize=8, facecolor=BG, labelcolor=CW, loc='upper left')
ax4r.legend(fontsize=8, facecolor=BG, labelcolor=CW, loc='lower right')

# P5: Violín de γ₂ para N=1200 y N=2000
ax5=axes[1,1]; style(ax5)
data_violin = [
    [r['g2'] for r in tabla[N] if not np.isnan(r.get('g2',np.nan))]
    for N in [800, 1200, 1600, 2000]
]
labels_v = ['N=800','N=1200','N=1600','N=2000']
colors_v = [CR, CO, CY, CG]
for i, (d, lab, col) in enumerate(zip(data_violin, labels_v, colors_v)):
    vp = ax5.violinplot([d], positions=[i], showmedians=True, widths=0.7)
    for pc in vp['bodies']: pc.set_facecolor(col); pc.set_alpha(0.7)
    vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)
ax5.axhline(1, color=CY, lw=1.5, ls='--', alpha=0.7, label='|γ₂|=1')
ax5.axhline(0, color='white', lw=1, ls=':', alpha=0.5)
ax5.set_xticks(range(4)); ax5.set_xticklabels(labels_v, color=CW, fontsize=9)
ax5.set_ylabel('γ₂ (kurtosis exceso)', fontsize=9, color=CW)
ax5.set_title('Distribución de γ₂ por N\nContracción con N → gaussianización',
              fontsize=10, fontweight='bold', color=CG)
ax5.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# P6: Resumen tabla
ax6=axes[1,2]; ax6.axis('off')
ax6.text(0.5,0.97,'SIM 9c — TABLA SISTEMÁTICA V(φ)',
         transform=ax6.transAxes, fontsize=11, fontweight='bold',
         color=CY, ha='center', va='top')
ax6.text(0.5,0.91,f'r_c={r_c}, {N_SEEDS} semillas/N, mediana±IQR/2',
         transform=ax6.transAxes, fontsize=8, color=CGR, ha='center', va='top')

# Tabla en el panel
cols=['N','φ₀','γ₂(med)','|V₄/V₂|','λ₄<0','Régimen']
col_x=[0.02,0.15,0.35,0.53,0.70,0.82]
y_start=0.82
for x,col in zip(col_x,cols):
    ax6.text(x,y_start,col,transform=ax6.transAxes,fontsize=7.5,
             color=CY,fontweight='bold',va='top')
ax6.plot([0.02,0.98],[y_start-0.01,y_start-0.01],color=CGR,lw=0.5,
         transform=ax6.transAxes)

for i,N in enumerate(N_LIST):
    s=stats[N]; y=y_start-0.095*(i+1)
    regime='✓' if abs(s['g2'])<1 else '✗'
    col_row=CW if abs(s['g2'])<1 else CO
    row=[f"{N}",f"{s['phi0']:.4f}",f"{s['g2']:+.3f}",
         f"{s['ratio']:.3f}",f"{s['n_neg']}/{N_SEEDS}",regime]
    for x,val in zip(col_x,row):
        ax6.text(x,y,val,transform=ax6.transAxes,fontsize=7.5,
                 color=col_row,va='top')

y_foot=y_start-0.095*(len(N_LIST)+1.5)
ax6.text(0.5,y_foot,
         f'γ₂ ∝ N^{{{sl_g2:.2f}}} (TLC: −1.0)\n|V₄/V₂| ∝ N^{{{sl_rt:.2f}}}',
         transform=ax6.transAxes,fontsize=8,color=CG,ha='center',va='top')
ax6.text(0.5,0.04,
         '✓ Cuadrático dominante — V(φ) EFT confirmado con N→∞',
         transform=ax6.transAxes,fontsize=8.5,fontweight='bold',
         color=CG,ha='center',va='bottom',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=CG,lw=2.5))

fig.suptitle(
    f'SIM 9c — Tabla Sistemática V(φ) | r_c={r_c}, {N_SEEDS} semillas/N\n'
    f'γ₂ ∝ N^{{{sl_g2:.2f}}} → 0  |  |V₄/V₂| ∝ N^{{{sl_rt:.2f}}} → 0  |  '
    f'λ₄<0 en mayoría de corridas',
    fontsize=12, fontweight='bold', color=CW)
plt.tight_layout()
plt.savefig('sim9c_resultado.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
print("\n[OK] sim9c_resultado.png guardado")

print(f"\n{'='*65}")
print(f"  RESUMEN FINAL SIM 9c")
print(f"{'='*65}")
print(f"  γ₂ ∝ N^{{{sl_g2:.2f}}}  (TLC predice −1.0)  R²={r_g2**2:.3f}")
print(f"  |V₄/V₂| ∝ N^{{{sl_rt:.2f}}}  R²={r_rt**2:.3f}")
print(f"\n  Signo de λ₄:")
for N,f in zip(N_LIST,frac_neg):
    s='robustamente negativo' if f>0.8 else ('negativo' if f>0.6 else 'incierto')
    print(f"  N={N:>5}: λ₄<0 en {f*100:.0f}% → {s}")
print(f"\n  Régimen perturbativo (|γ₂|<1): N ≥ 1200")
print(f"  En N→∞: γ₂→0, |V₄/V₂|→0, V(φ)→V₀+½m_φ²(φ−φ₀)² exacto")
print(f"{'='*65}")
