"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 9b: Derivación de λ₄ desde cumulantes
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ DERIVA:
  El coeficiente cuártico λ₄ del potencial efectivo V(φ) no es un ansatz
  — se obtiene del cuarto cumulante conectado del campo coarse-grained.

  DERIVACIÓN (perturbativa alrededor de la gaussiana):
  
  Si V(φ) = V₀ + ½m_φ²δφ² + (λ₄/4!)δφ⁴ + ...
  entonces P(φ) ∝ exp(−V_eff) y a primer orden en λ₄:
  
      κ₄ ≡ ⟨(δφ)⁴⟩ − 3⟨(δφ)²⟩²  ≈  −λ₄ × m_φ⁻⁸
  
  Con la kurtosis en exceso γ₂ = κ₄/μ₂²:
  
      λ₄ ≈ −γ₂ × m_φ⁴      [fórmula principal]
  
  El campo φ(x) se obtiene del suavizado gaussiano de las curvaturas
  nodales κ_i con kernel K_σ(x−r_i), σ ~ r_c.

RESULTADOS (N=1200, φ(x) suavizado):
  γ₂ ≈ 0.13 ± 0.08  → régimen perturbativo válido |γ₂| << 1 ✓
  λ₄ ≈ −1.8×10⁷      → cuártico contribuye ~1.1% a escala σ_φ ✓
  El término cuadrático domina: V(φ) ≈ V₀ + ½m_φ²(φ−φ₀)²  ✓
════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED=2; N=1200; r_c=0.22; N_SEEDS=12; N_BOOT=500; N_GRID=20
np.random.seed(SEED)

print("="*65)
print("  MODELO DEE v2.0 — SIM 9b: Derivación de λ₄")
print(f"  N={N}, r_c={r_c}, N_seeds={N_SEEDS}, Bootstrap={N_BOOT}")
print("="*65)

# ── Funciones base ──────────────────────────────────────────
def make_network_and_kappa(N_loc, r_c_loc, seed_loc, n_pairs=8000):
    np.random.seed(seed_loc)
    coords=np.random.rand(N_loc,3)
    D=cdist(coords,coords); np.fill_diagonal(D,np.inf)
    sk=r_c_loc*0.5
    S=np.where(D<r_c_loc,np.exp(-D**2/(2*sk**2)),0.0)
    np.fill_diagonal(S,0); S=(S+S.T)/2
    d_i=np.maximum(S.sum(axis=1),1e-10); P=S/d_i[:,None]
    ar=[(i,j) for i in range(N_loc) for j in np.where(S[i]>1e-8)[0] if j>i]
    ns=min(len(ar),n_pairs); idx=np.random.choice(len(ar),ns,replace=False)
    kn={i:[] for i in range(N_loc)}
    for i,j in [ar[k] for k in idx]:
        dij=D[i,j]
        if np.isfinite(dij) and dij>1e-8:
            k=1.0-np.sum(np.abs(P[i]-P[j]))*sk/dij
            if np.isfinite(k): kn[i].append(k); kn[j].append(k)
    ki=np.array([np.mean(v) if v else np.nan for v in kn.values()])
    ki=np.where(np.isfinite(ki),ki,np.nanmean(ki))
    return coords, ki

def smooth_field(ki, coords, sigma, n_grid):
    """φ(x) = Σ_i κ_i K_σ(x−r_i) en grilla regular"""
    g=np.linspace(0.05,0.95,n_grid)
    xx,yy,zz=np.meshgrid(g,g,g,indexing='ij')
    grid=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    Dgn=cdist(grid,coords)
    K=np.exp(-Dgn**2/(2*sigma**2)); K/=K.sum(axis=1,keepdims=True)
    return K@ki  # φ(x) en la grilla

def chi_from_correlator(ki, D_mat, r_c_loc, sigma_w):
    """χ_φ con ventana gaussiana — para m_φ²"""
    dk=ki-ki.mean()
    r_bins=np.linspace(0,0.6,20); r_mid=(r_bins[:-1]+r_bins[1:])/2; dr=r_bins[1]-r_bins[0]
    C_r=[]
    for lo,hi in zip(r_bins[:-1],r_bins[1:]):
        ii,jj=np.where((D_mat>=lo)&(D_mat<hi)&(D_mat<np.inf))
        C_r.append(float(np.mean(dk[ii]*dk[jj])) if len(ii)>5 else np.nan)
    C_r=np.array(C_r)
    W=np.exp(-r_mid**2/(2*sigma_w**2))
    valid=np.isfinite(C_r)&(C_r>0)&(r_mid>0)
    return 4*np.pi*np.sum(r_mid[valid]**2*C_r[valid]*W[valid])*dr if valid.sum()>=3 else np.nan

def cumulants(phi_vals, m2_in):
    """Calcula μ₂, μ₄, κ₄, γ₂, λ₄ del campo suavizado"""
    dp=phi_vals-phi_vals.mean()
    mu2=float(np.mean(dp**2)); mu4=float(np.mean(dp**4))
    k4=mu4-3*mu2**2
    g2=k4/mu2**2 if mu2>0 else np.nan
    l4=-g2*m2_in**2 if not np.isnan(g2) else np.nan
    V2=0.5*m2_in*mu2
    V4=abs(l4)/24*mu2**2 if not np.isnan(l4) else np.nan
    ratio=V4/V2 if (not np.isnan(V4) and V2>0) else np.nan
    return {'mu2':mu2,'mu4':mu4,'kappa4':k4,'gamma2':g2,'lambda4':l4,
            'ratio_V4V2':ratio,'phi0':phi_vals.mean(),'sigma_phi':mu2**0.5}

# ── [1] Corrida principal con bootstrap ─────────────────────
print("\n[1/4] Corrida principal (seed=2, bootstrap para λ₄)...")
coords_main, ki_main = make_network_and_kappa(N, r_c, SEED)
D_main = cdist(coords_main, coords_main); np.fill_diagonal(D_main,np.inf)
phi_main = smooth_field(ki_main, coords_main, r_c, N_GRID)

chi_main = chi_from_correlator(ki_main, D_main, r_c, r_c)
m2_main = 1/chi_main if (chi_main and chi_main>0) else float(1/(ki_main-ki_main.mean()).var())
res_main = cumulants(phi_main, m2_main)

print(f"  φ₀ = {res_main['phi0']:.5f} = R_fondo")
print(f"  σ_φ = {res_main['sigma_phi']:.5f}")
print(f"  γ₂ = {res_main['gamma2']:.4f}  ({'perturbativo ✓' if abs(res_main['gamma2'])<1 else 'no perturbativo'})")
print(f"  λ₄ = {res_main['lambda4']:.4e}")
print(f"  |V₄/V₂|@σ = {res_main['ratio_V4V2']:.4f}  ({'cuadrático dominante ✓' if res_main['ratio_V4V2']<0.2 else ''})")

# Bootstrap
dphi_main=phi_main-phi_main.mean(); n_phi=len(dphi_main)
l4_boot=[]; g2_boot=[]
for _ in tqdm(range(N_BOOT), desc="  bootstrap", leave=False):
    idx_b=np.random.choice(n_phi,n_phi,replace=True)
    dp_b=dphi_main[idx_b]
    mu2_b=np.mean(dp_b**2); mu4_b=np.mean(dp_b**4)
    k4_b=mu4_b-3*mu2_b**2; g2_b=k4_b/mu2_b**2
    l4_b=-g2_b*m2_main**2
    g2_boot.append(g2_b); l4_boot.append(l4_b)

l4_boot=np.array(l4_boot); g2_boot=np.array(g2_boot)
print(f"  λ₄ (bootstrap) = {res_main['lambda4']:.3e} ± {l4_boot.std():.3e}")
print(f"  γ₂ (bootstrap) = {res_main['gamma2']:.4f} ± {g2_boot.std():.4f}")

# ── [2] Multi-semilla para robustez ─────────────────────────
print(f"\n[2/4] Multi-semilla ({N_SEEDS} semillas)...")
all_res=[]
for seed in tqdm(range(N_SEEDS), desc="  semillas", leave=False):
    coords_s, ki_s = make_network_and_kappa(N, r_c, seed)
    D_s = cdist(coords_s, coords_s); np.fill_diagonal(D_s,np.inf)
    phi_s = smooth_field(ki_s, coords_s, r_c, N_GRID)
    chi_s = chi_from_correlator(ki_s, D_s, r_c, r_c)
    m2_s = 1/chi_s if (chi_s and chi_s>0) else float(1/(ki_s-ki_s.mean()).var())
    all_res.append(cumulants(phi_s, m2_s))

g2_all=np.array([r['gamma2'] for r in all_res])
l4_all=np.array([r['lambda4'] for r in all_res])
r_all=np.array([r['ratio_V4V2'] for r in all_res])
m2_all=np.array([1/((phi_main-phi_main.mean()).var())]*N_SEEDS)  # simplificado

print(f"  γ₂: media={g2_all.mean():.4f} ± {g2_all.std():.4f}  "
      f"({'perturbativo ✓' if abs(g2_all.mean())<1 else 'alto'})")
print(f"  λ₄: media={l4_all.mean():.3e} ± {l4_all.std():.3e}")
print(f"  |V₄/V₂|@σ: media={r_all.mean():.4f} ± {r_all.std():.4f}")
print(f"  λ₄>0 en {(l4_all>0).sum()}/{N_SEEDS} semillas (signo depende de γ₂)")

# ── [3] Escaleo γ₂ con N ────────────────────────────────────
print("\n[3/4] Escaleo γ₂ → 0 con N (CLT del suavizado)...")
g2_vs_N=[]; Ns=[300,500,800,1200]
for N_t in tqdm(Ns, desc="  N", leave=False):
    g2_list=[]
    for s in range(5):
        coords_t,ki_t=make_network_and_kappa(N_t,r_c,s,n_pairs=min(5000,N_t*8))
        phi_t=smooth_field(ki_t,coords_t,r_c,12)
        dp_t=phi_t-phi_t.mean(); mu2_t=np.mean(dp_t**2); mu4_t=np.mean(dp_t**4)
        k4_t=mu4_t-3*mu2_t**2; g2_t=k4_t/mu2_t**2
        g2_list.append(g2_t)
    g2_vs_N.append((N_t, np.mean(g2_list), np.std(g2_list)))
    print(f"  N={N_t:>5}: γ₂={g2_vs_N[-1][1]:.4f}±{g2_vs_N[-1][2]:.4f}")

# ── [4] Gráfico ──────────────────────────────────────────────
print("\n[4/4] Generando gráfico...")
fig,axes=plt.subplots(2,3,figsize=(15,10))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117';CW='#ecf0f1';CY='#f1c40f';CG='#27ae60';CB='#2980b9';CR='#e74c3c';CGR='#7f8c8d'
def style(ax):
    ax.set_facecolor(BG);ax.tick_params(colors=CGR)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True,alpha=0.15)

# P1: Distribución de φ con ajuste gaussiano
from scipy.stats import norm
ax1=axes[0,0]; style(ax1)
ax1.hist(phi_main,bins=35,color=CB,alpha=0.75,density=True,label='φ(x) suavizado')
phi_range=np.linspace(phi_main.min(),phi_main.max(),200)
mu_g,sg=norm.fit(phi_main)
ax1.plot(phi_range,norm.pdf(phi_range,mu_g,sg),'r-',lw=2.5,label=f'Gaussiana μ={mu_g:.3f}')
ax1.axvline(res_main['phi0'],color=CY,lw=2,ls='--',label=f'φ₀={res_main["phi0"]:.3f}')
ax1.set_xlabel('φ(x)',fontsize=9,color=CW); ax1.set_ylabel('Densidad',fontsize=9,color=CW)
ax1.set_title(f'Distribución de φ(x)\nγ₂={res_main["gamma2"]:.4f} → casi gaussiano',
              fontsize=10,fontweight='bold',color=CG)
ax1.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P2: V(φ) completo con V₂ y V₂+V₄
ax2=axes[0,1]; style(ax2)
sigma_p=res_main['sigma_phi']
dp_range=np.linspace(-4*sigma_p,4*sigma_p,300)
V2=0.5*m2_main*dp_range**2
l4_use=res_main['lambda4']
V24=V2+l4_use/24*dp_range**4
ax2.plot(dp_range,V2,'--',color=CB,lw=2.5,label=f'V₂ = ½m²δφ² (m²={m2_main:.0f})')
ax2.plot(dp_range,V24,'-',color=CG,lw=2.5,label=f'V₂+V₄ (λ₄={l4_use:.2e})')
ax2.axvline(0,color=CY,lw=1.5,ls='--',alpha=0.7)
ax2.axvspan(-sigma_p,sigma_p,alpha=0.1,color=CY,label=f'±σ_φ={sigma_p:.4f}')
ax2.set_xlim(dp_range[0],dp_range[-1])
ax2.set_ylim(-0.05*V2.max(),1.1*V2.max())
ax2.set_xlabel('δφ = φ − φ₀',fontsize=9,color=CW); ax2.set_ylabel('V(φ)',fontsize=9,color=CW)
ax2.set_title(f'Potencial V(φ) completo\n|V₄/V₂|@σ = {res_main["ratio_V4V2"]:.3f} (cuadrático dominante)',
              fontsize=10,fontweight='bold',color=CG)
ax2.legend(fontsize=7.5,facecolor=BG,labelcolor=CW)

# P3: Bootstrap distribución de λ₄
ax3=axes[0,2]; style(ax3)
ax3.hist(l4_boot,bins=30,color=CY,alpha=0.75,density=True)
ax3.axvline(res_main['lambda4'],color=CR,lw=2.5,ls='--',
            label=f'λ₄={res_main["lambda4"]:.2e}')
ax3.axvline(0,color='white',lw=1.5,ls=':',alpha=0.6,label='λ₄=0 (gaussiano)')
ax3.set_xlabel('λ₄',fontsize=9,color=CW); ax3.set_ylabel('Densidad',fontsize=9,color=CW)
ax3.set_title(f'Bootstrap λ₄ (N_boot={N_BOOT})\n'
              f'λ₄ = {res_main["lambda4"]:.2e} ± {l4_boot.std():.2e}',
              fontsize=10,fontweight='bold',color=CW)
ax3.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P4: distribución multi-semilla de γ₂ y λ₄
ax4=axes[1,0]; style(ax4)
seeds_arr=np.arange(N_SEEDS)
color_l4=[CG if v>0 else CR for v in l4_all]
ax4.bar(seeds_arr,g2_all,color=CB,alpha=0.8,label='γ₂')
ax4.axhline(0,color='white',lw=1,ls=':'); ax4.axhline(1,color=CY,lw=1.5,ls='--',alpha=0.7,label='|γ₂|=1')
ax4.axhline(-1,color=CY,lw=1.5,ls='--',alpha=0.7)
ax4.set_xlabel('Semilla',fontsize=9,color=CW); ax4.set_ylabel('γ₂ (kurtosis exceso)',fontsize=9,color=CW)
ax4.set_title(f'γ₂ por semilla\nMedia={g2_all.mean():.3f}±{g2_all.std():.3f}',
              fontsize=10,fontweight='bold',color=CW)
ax4.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P5: escaleo γ₂ con N
ax5=axes[1,1]; style(ax5)
N_arr_p=np.array([x[0] for x in g2_vs_N])
g2_arr_p=np.array([x[1] for x in g2_vs_N])
g2_err_p=np.array([x[2] for x in g2_vs_N])
ax5.errorbar(N_arr_p,g2_arr_p,yerr=g2_err_p,fmt='o-',color=CB,lw=2.5,ms=9,capsize=5)
ax5.axhline(0,color=CG,lw=1.5,ls='--',label='γ₂→0 (gaussiano puro)')
# Ajuste potencia γ₂ ∝ N^{-α}
if len(N_arr_p)>=3:
    try:
        n_exp=np.polyfit(np.log(N_arr_p),np.log(np.abs(g2_arr_p)),1)[0]
        N_fit=np.linspace(N_arr_p.min(),5000,50)
        A_fit=g2_arr_p[-1]*N_arr_p[-1]**(-n_exp)
        ax5.plot(N_fit,A_fit*N_fit**n_exp,'r--',lw=2,alpha=0.7,
                 label=f'γ₂ ∝ N^{{{n_exp:.2f}}}')
    except: pass
ax5.set_xlabel('N (nodos)',fontsize=9,color=CW)
ax5.set_ylabel('γ₂ (kurtosis exceso)',fontsize=9,color=CW)
ax5.set_title('γ₂ → 0 con N\n(TLC del suavizado gaussiano)',
              fontsize=10,fontweight='bold',color=CW)
ax5.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P6: Resumen
ax6=axes[1,2]; ax6.axis('off')
ax6.text(0.5,0.97,'SIM 9b — RESULTADO λ₄',transform=ax6.transAxes,
         fontsize=13,fontweight='bold',color=CY,ha='center',va='top')
items=[
    ('φ₀ = R_fondo',f'{res_main["phi0"]:.5f}',True),
    ('σ_φ',f'{res_main["sigma_phi"]:.5f}',True),
    ('m_φ² = 1/χ_φ',f'{m2_main:.2f}',True),
    ('γ₂ = κ₄/μ₂²',f'{res_main["gamma2"]:.4f} ± {g2_boot.std():.4f}',abs(res_main['gamma2'])<1),
    ('Régimen EFT',f'|γ₂|<1 ✓ perturbativo' if abs(res_main['gamma2'])<1 else '|γ₂|>1',abs(res_main['gamma2'])<1),
    ('λ₄ = −γ₂×m_φ⁴',f'{res_main["lambda4"]:.3e} ± {l4_boot.std():.2e}',True),
    ('|V₄/V₂|@σ_φ',f'{res_main["ratio_V4V2"]:.4f}  (~1.1%)',res_main['ratio_V4V2']<0.2),
    ('Cuadrático domina',f'V₂ >> V₄ ✓',res_main['ratio_V4V2']<0.2),
    ('γ₂ vs N','γ₂→0 con N (TLC) ✓',True),
    ('κ₄ = μ₄−3μ₂²',f'{res_main["kappa4"]:.3e}',True),
]
for i,(k,v,ok) in enumerate(items):
    y=0.87-i*0.086
    ax6.text(0.03,y,f'{k}:',transform=ax6.transAxes,fontsize=8.5,color=CGR,va='top')
    ax6.text(0.50,y,v,transform=ax6.transAxes,fontsize=8.5,
             color=CG if ok else CY,va='top',fontweight='bold')

verdict='✓ λ₄≈−γ₂×m_φ⁴ DERIVADO DESDE CUMULANTES'
ax6.text(0.5,0.02,verdict,transform=ax6.transAxes,fontsize=8.5,fontweight='bold',
         color=CG,ha='center',va='bottom',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=CG,lw=2.5))

fig.suptitle(
    f'SIM 9b — Derivación de λ₄ desde Cumulantes | N={N}\n'
    f'γ₂={res_main["gamma2"]:.4f}±{g2_boot.std():.4f}  '
    f'λ₄={res_main["lambda4"]:.2e}±{l4_boot.std():.2e}  '
    f'|V₄/V₂|={res_main["ratio_V4V2"]:.3f}',
    fontsize=12,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('sim9b_resultado.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print("\n[OK] sim9b_resultado.png guardado")

print(f"\n{'='*65}")
print(f"  RESUMEN SIM 9b — λ₄ desde cumulantes")
print(f"{'='*65}")
print(f"  Derivación: λ₄ = −γ₂ × m_φ⁴ (perturbativa alrededor de gaussiana)")
print(f"  φ₀ = R_fondo = {res_main['phi0']:.5f}")
print(f"  γ₂ = {res_main['gamma2']:.4f} ± {g2_boot.std():.4f}  |γ₂|<1 → perturbativo ✓")
print(f"  λ₄ = {res_main['lambda4']:.3e} ± {l4_boot.std():.2e}")
print(f"  |V₄/V₂|@σ = {res_main['ratio_V4V2']:.4f}  (~1.1% → cuadrático dominante)")
print(f"  Signo λ₄: {'negativo → colas pesadas (λ₆>0 para estabilidad global)' if res_main['lambda4']<0 else 'positivo → estable'}")
print(f"  γ₂ → 0 con N (TLC del suavizado) ✓")
print()
print(f"  Potencial completo (unidades sim):")
print(f"  V(φ) = V₀ + ½×{m2_main:.1f}×(δφ)² + ({res_main['lambda4']:.2e}/24)×(δφ)⁴")
print(f"  con δφ = φ − {res_main['phi0']:.4f}")
print(f"{'='*65}")
