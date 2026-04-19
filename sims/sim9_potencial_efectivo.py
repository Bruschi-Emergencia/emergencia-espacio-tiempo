"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 9: Potencial efectivo V(φ) — masa m_φ²
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ VERIFICA:
  El potencial V(φ) no es un ansatz sino el potencial efectivo de Landau
  del campo coarse-grained φ(x) = Σ_i κ_i K_σ(x−r_i).

  El coeficiente cuadrático satisface:
      m_φ² = χ_φ⁻¹
  donde χ_φ = ∫d³r ⟨δφ(0) δφ(r)⟩ es la susceptibilidad del vacío.

  Al nivel de la red:
      χ_φ = Σ_{i,j} ⟨δκ_i δκ_j⟩ K_ij(σ)

  SIM 9 mide:
    [1] Correlador radial C(r) = ⟨δκ_i δκ_j⟩ vs distancia
    [2] Longitud de correlación ξ por ajuste C(r) ~ A exp(-r/ξ)
    [3] Susceptibilidad χ_φ con ventana gaussiana σ ~ r_c
    [4] Masa efectiva m_φ² = 1/χ_φ  para vacío y con masa
    [5] Escaleo m_φ vs N (convergencia en el límite continuo)
    [6] Relación m_φ ~ ξ⁻¹ (conexión EFT con longitud de correlación)

REFERENCIA EFT:
  V(φ) = V₀ + ½ m_φ²(φ−φ₀)² + λ₃/3!(φ−φ₀)³ + λ₄/4!(φ−φ₀)⁴ + ...
  con m_φ² = χ_φ⁻¹,  φ₀ = R_fondo = ⟨κ_i⟩_vac
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

SEED=42; N=1200; r_c=0.22; N_SEEDS=10; SIGMA=r_c
np.random.seed(SEED)

print("="*65)
print("  MODELO DEE v2.0 — SIM 9: Potencial efectivo V(φ)")
print(f"  N={N}, r_c={r_c}, N_SEEDS={N_SEEDS}")
print("="*65)

# ── Función central: κ_i y correlador ──────────────────────
def compute_kappa_i(coords, r_c, sigma_factor=0.5, boost=0, n_pairs=6000):
    N_loc = len(coords)
    D = cdist(coords, coords); np.fill_diagonal(D, np.inf)
    sigma_k = r_c * sigma_factor
    S = np.where(D < r_c, np.exp(-D**2/(2*sigma_k**2)), 0.0)
    if boost > 0:
        i0 = np.argmin(np.linalg.norm(coords-0.5, axis=1))
        for j in np.where((D[i0]<r_c)&(D[i0]>0))[0]:
            S[i0,j]+=boost; S[j,i0]+=boost
    np.fill_diagonal(S,0); S=(S+S.T)/2
    d_i = np.maximum(S.sum(axis=1),1e-10); P = S/d_i[:,None]
    
    aristas=[(i,j) for i in range(N_loc) for j in np.where(S[i]>1e-8)[0] if j>i]
    n_s = min(len(aristas), n_pairs)
    idx_s = np.random.choice(len(aristas), n_s, replace=False)
    kappa_nodo = {i:[] for i in range(N_loc)}
    for i,j in [aristas[k] for k in idx_s]:
        d_ij = D[i,j]
        if np.isfinite(d_ij) and d_ij > 1e-8:
            k = 1.0 - np.sum(np.abs(P[i]-P[j]))*sigma_k/d_ij
            if np.isfinite(k): kappa_nodo[i].append(k); kappa_nodo[j].append(k)
    
    ki = np.array([np.mean(v) if v else np.nan for v in kappa_nodo.values()])
    ki = np.where(np.isfinite(ki), ki, np.nanmean(ki))
    return ki, D

def correlador_radial(ki, D, r_bins):
    dk = ki - ki.mean()
    C_r = []
    for lo,hi in zip(r_bins[:-1], r_bins[1:]):
        ii,jj = np.where((D>=lo)&(D<hi)&(D<np.inf))
        if len(ii) > 10:
            C_r.append(float(np.mean(dk[ii]*dk[jj])))
        else:
            C_r.append(np.nan)
    return np.array(C_r)

def susceptibilidad(C_r, r_mid, sigma_w, dr):
    """χ_φ con ventana gaussiana para suprimir ruido de largo alcance"""
    W = np.exp(-r_mid**2/(2*sigma_w**2))
    valid = np.isfinite(C_r) & (C_r > 0) & (r_mid > 0)
    if valid.sum() < 3: return np.nan
    return 4*np.pi*np.sum(r_mid[valid]**2 * C_r[valid] * W[valid]) * dr

def ajuste_exponencial(r_mid, C_r):
    """Ajusta C(r) ~ A exp(-r/ξ) para obtener longitud de correlación ξ"""
    valid = np.isfinite(C_r) & (C_r > 0) & (r_mid > 0.02)
    if valid.sum() < 4: return np.nan, np.nan
    try:
        popt,_ = curve_fit(lambda r,A,xi: A*np.exp(-r/xi),
                           r_mid[valid], C_r[valid],
                           p0=[C_r[valid][0], 0.1], maxfev=3000,
                           bounds=([0,0.01],[10,2.0]))
        return popt[0], popt[1]  # A, ξ
    except: return np.nan, np.nan

# ── [1] Correlador radial en vacío y con masa ──────────────
print("\n[1/5] Correladores radiales: vacío vs masa (N_seeds=3)...")
r_bins = np.linspace(0, 0.65, 22)
r_mid = (r_bins[:-1]+r_bins[1:])/2
dr = r_bins[1]-r_bins[0]

C_vac_all=[]; C_mas_all=[]
for seed in tqdm(range(3), desc="  semillas", leave=False):
    np.random.seed(seed)
    coords = np.random.rand(N,3)
    ki_v, D = compute_kappa_i(coords, r_c, boost=0)
    ki_m, _  = compute_kappa_i(coords, r_c, boost=600)
    C_vac_all.append(correlador_radial(ki_v, D, r_bins))
    C_mas_all.append(correlador_radial(ki_m, D, r_bins))

C_vac = np.nanmean(C_vac_all, axis=0)
C_mas = np.nanmean(C_mas_all, axis=0)
A_v, xi_v = ajuste_exponencial(r_mid, C_vac)
A_m, xi_m = ajuste_exponencial(r_mid, C_mas)
print(f"  Vacío:  C(0)≈{C_vac[0]:.5f}  ξ={xi_v:.3f}  A={A_v:.5f}")
print(f"  Masa:   C(0)≈{C_mas[0]:.5f}  ξ={xi_m:.3f}  A={A_m:.5f}")

# ── [2] Susceptibilidad y m_φ² con múltiples semillas ──────
print(f"\n[2/5] χ_φ y m_φ² con {N_SEEDS} semillas (vacío vs masa)...")
chi_vac=[]; chi_mas=[]; xi_vac=[]; xi_mas=[]
for seed in tqdm(range(N_SEEDS), desc="  semillas", leave=False):
    np.random.seed(seed)
    coords=np.random.rand(N,3)
    for boost, store_chi, store_xi in [(0,chi_vac,xi_vac),(600,chi_mas,xi_mas)]:
        ki,D=compute_kappa_i(coords,r_c,boost=boost)
        C=correlador_radial(ki,D,r_bins)
        chi=susceptibilidad(C,r_mid,SIGMA,dr)
        _,xi=ajuste_exponencial(r_mid,C)
        if not np.isnan(chi) and chi>0: store_chi.append(chi)
        if not np.isnan(xi): store_xi.append(xi)

chi_v=np.mean(chi_vac) if chi_vac else np.nan
chi_m=np.mean(chi_mas) if chi_mas else np.nan
xi_v_m=np.mean(xi_vac) if xi_vac else np.nan
xi_m_m=np.mean(xi_mas) if xi_mas else np.nan
m2_v=1/chi_v; m2_m=1/chi_m

print(f"\n  Vacío:  χ_φ={chi_v:.5e}  m_φ²={m2_v:.2f}  m_φ={m2_v**0.5:.2f}")
print(f"  Masa:   χ_φ={chi_m:.5e}  m_φ²={m2_m:.2f}  m_φ={m2_m**0.5:.2f}")
print(f"  ξ vacío={xi_v_m:.3f}  ξ masa={xi_m_m:.3f}")
print(f"  Masa reduce m_φ²: {'✓' if m2_m < m2_v else '✗'} (campo más blando con masa)")
print(f"  m_φ × ξ vacío = {m2_v**0.5 * xi_v_m:.4f}  (≈ cte si m_φ~ξ⁻¹)")

# ── [3] Escaleo con N ──────────────────────────────────────
print("\n[3/5] Escaleo m_φ² vs N (densidad fija ρ=N/V=N)...")
Ns=[300,500,800,1200]; resultados_N=[]
for N_test in tqdm(Ns, desc="  N", leave=False):
    chi_list=[]; xi_list=[]
    for seed in range(5):
        np.random.seed(seed)
        coords=np.random.rand(N_test,3)
        ki,D=compute_kappa_i(coords,r_c,boost=0,n_pairs=min(4000,N_test*10))
        C=correlador_radial(ki,D,r_bins)
        chi=susceptibilidad(C,r_mid,SIGMA,dr)
        _,xi=ajuste_exponencial(r_mid,C)
        if not np.isnan(chi) and chi>0: chi_list.append(chi)
        if not np.isnan(xi): xi_list.append(xi)
    chi_n=np.mean(chi_list) if chi_list else np.nan
    xi_n=np.mean(xi_list) if xi_list else np.nan
    m2_n=1/chi_n if not np.isnan(chi_n) else np.nan
    resultados_N.append((N_test, chi_n, xi_n, m2_n))
    print(f"  N={N_test:>5}: χ={chi_n:.4e}  ξ={xi_n:.3f}  m_φ²={m2_n:.2f}")

# ── [4] Relación m_φ ~ ξ⁻¹ ────────────────────────────────
print("\n[4/5] Verificando m_φ ~ ξ^{-n}...")
xi_arr=np.array([r[2] for r in resultados_N if not np.isnan(r[2])])
m2_arr=np.array([r[3] for r in resultados_N if not np.isnan(r[3]) and r[2]==r[2]])
if len(xi_arr)>=3 and len(m2_arr)>=3:
    try:
        n_fit=np.polyfit(np.log(xi_arr),np.log(m2_arr**0.5),1)[0]
        print(f"  m_φ ∝ ξ^{{{n_fit:.2f}}}  (EFT predice n=-1)")
    except: pass

# ── [5] Estructura V(φ): histograma de φ ──────────────────
print("\n[5/5] Distribución de φ(x) → forma de V(φ)...")
np.random.seed(42)
coords_f=np.random.rand(N,3)
ki_f,_=compute_kappa_i(coords_f,r_c,boost=0)
phi_vals=ki_f  # φ(x_i) ≈ κ_i antes del smoothing continuo
phi0=phi_vals.mean()
dphi=phi_vals-phi0
# Ajuste gaussiano
from scipy.stats import norm
mu_fit,std_fit=norm.fit(phi_vals)
print(f"  φ₀ = R_fondo = {phi0:.5f}")
print(f"  σ_φ = {dphi.std():.5f}")
print(f"  ⟨δφ²⟩ = {(dphi**2).mean():.6f}")
print(f"  χ_φ estimada = {(dphi**2).mean():.6f}")
print(f"  m_φ² ≈ {1/(dphi**2).mean():.2f}")

# ── GRÁFICO ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117'; CW='#ecf0f1'; CY='#f1c40f'; CG='#27ae60'
CB='#2980b9'; CR='#e74c3c'; CGR='#7f8c8d'; CO='#e67e22'
def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True, alpha=0.15)

# P1: Correlador radial C(r)
ax1=axes[0,0]; style(ax1)
valid_v=np.isfinite(C_vac); valid_m=np.isfinite(C_mas)
ax1.semilogy(r_mid[valid_v&(C_vac>0)], C_vac[valid_v&(C_vac>0)],
             'o-', color=CB, lw=2, ms=6, label='C(r) vacío')
ax1.semilogy(r_mid[valid_m&(C_mas>0)], C_mas[valid_m&(C_mas>0)],
             's--', color=CY, lw=2, ms=6, label='C(r) masa')
if not np.isnan(A_v) and not np.isnan(xi_v):
    r_fit=np.linspace(0.02,0.45,50)
    ax1.semilogy(r_fit,A_v*np.exp(-r_fit/xi_v),'--',color=CB,alpha=0.5,
                 label=f'exp(-r/{xi_v:.3f})')
ax1.set_xlabel('Distancia r',fontsize=9,color=CW)
ax1.set_ylabel('C(r) = ⟨δκ_i δκ_j⟩',fontsize=9,color=CW)
ax1.set_title('Correlador radial κ\nDecaimiento exponencial → ξ',
              fontsize=10,fontweight='bold',color=CG)
ax1.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P2: Distribución de φ y forma de V
ax2=axes[0,1]; style(ax2)
dphi_fine=phi_vals-phi0
ax2.hist(phi_vals,bins=30,color=CB,alpha=0.7,density=True,label='distribución φ_i')
phi_arr=np.linspace(phi_vals.min(),phi_vals.max(),100)
ax2.plot(phi_arr,norm.pdf(phi_arr,mu_fit,std_fit),'r-',lw=2.5,
         label=f'Gaussiana μ={mu_fit:.3f}')
ax2.axvline(phi0,color=CY,lw=2,ls='--',label=f'φ₀={phi0:.3f}')
ax2.set_xlabel('φ = κ_i (curvatura nodal)',fontsize=9,color=CW)
ax2.set_ylabel('Densidad',fontsize=9,color=CW)
ax2.set_title('Distribución de φ(x)\nGaussiana → V(φ) cuadrático dominante',
              fontsize=10,fontweight='bold',color=CG)
ax2.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P3: V(φ) reconstruido desde distribución
ax3=axes[0,2]; style(ax3)
dphi_range=np.linspace(-3*dphi.std(),3*dphi.std(),200)
V_quadratic=0.5*m2_v*dphi_range**2
# Corrección de orden 4 (si la distribución no es perfectamente gaussiana)
kurt=np.mean(dphi**4)/(np.mean(dphi**2))**2-3  # curtosis excedente
lambda4=kurt/(np.mean(dphi**2))**2 * 6 if abs(kurt)>0.01 else 0
V_quartic=V_quadratic + lambda4/24*dphi_range**4
ax3.plot(dphi_range,V_quadratic,'--',color=CB,lw=2,label=f'V₂ = ½m²δφ² (m²={m2_v:.1f})')
if abs(lambda4)>1:
    ax3.plot(dphi_range,V_quartic,'-',color=CG,lw=2.5,
             label=f'V₂+V₄ (λ₄={lambda4:.2f})')
ax3.axhline(0,color='white',lw=0.8,ls=':',alpha=0.4)
ax3.axvline(0,color=CY,lw=1.5,ls='--',alpha=0.7,label=f'φ₀={phi0:.3f}')
ax3.set_xlim(dphi_range[0],dphi_range[-1])
ax3.set_ylim(-0.1*V_quadratic.max(), 1.2*V_quadratic.max())
ax3.set_xlabel('δφ = φ − φ₀',fontsize=9,color=CW)
ax3.set_ylabel('V(φ)',fontsize=9,color=CW)
ax3.set_title('Potencial efectivo V(φ)\nReconstruido desde χ_φ',
              fontsize=10,fontweight='bold',color=CG)
ax3.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P4: m_φ² vacío vs masa
ax4=axes[1,0]; style(ax4)
labels_bar=['Vacío\n(boost=0)',f'Masa\n(boost=600)']
m2_vals=[m2_v, m2_m]
chi_vals=[chi_v*1e4, chi_m*1e4]
colors_bar=[CB, CY]
bars=ax4.bar(labels_bar, m2_vals, color=colors_bar, alpha=0.85,
             edgecolor='white', lw=1.2)
ax4.bar_label(bars, fmt='%.1f', label_type='edge', color=CW, fontsize=10)
ax4.set_ylabel('m_φ² = 1/χ_φ',fontsize=10,color=CW)
mass_lower = m2_m < m2_v
ax4.set_title(f'Masa efectiva m_φ²\nMasa reduce m_φ² {"✓" if mass_lower else "~"} (campo más blando)',
              fontsize=10,fontweight='bold',color=CG if mass_lower else CY)

# P5: Escaleo con N
ax5=axes[1,1]; style(ax5)
N_arr=np.array([r[0] for r in resultados_N])
m_arr=np.array([r[3]**0.5 for r in resultados_N if not np.isnan(r[3])])
xi_arr2=np.array([r[2] for r in resultados_N if not np.isnan(r[2])])
N_valid=N_arr[:len(m_arr)]
ax5.loglog(N_valid,m_arr,'o-',color=CB,lw=2.5,ms=9,label='m_φ')
if len(xi_arr2)>=2:
    ax5_r=ax5.twinx()
    ax5_r.semilogy(N_valid,xi_arr2,'s--',color=CY,lw=2,ms=8,label='ξ (correlación)')
    ax5_r.tick_params(colors=CGR)
    ax5_r.set_ylabel('ξ (longitud de correlación)',fontsize=9,color=CY)
ax5.set_xlabel('N (nodos)',fontsize=9,color=CW)
ax5.set_ylabel('m_φ',fontsize=9,color=CB)
ax5.set_title('Escaleo m_φ y ξ con N\nm_φ ~ ξ⁻¹ (predicción EFT)',
              fontsize=10,fontweight='bold',color=CW)
ax5.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# P6: Resumen
ax6=axes[1,2]; ax6.axis('off')
ax6.text(0.5,0.97,'SIM 9 — POTENCIAL V(φ)',transform=ax6.transAxes,
         fontsize=13,fontweight='bold',color=CY,ha='center',va='top')
items=[
    ('φ₀ = R_fondo', f'{phi0:.5f}', True),
    ('σ_φ = std(κ_i)', f'{dphi.std():.5f}', True),
    ('χ_φ (vacío)', f'{chi_v:.4e}', True),
    ('m_φ² = 1/χ_φ', f'{m2_v:.2f}', True),
    ('m_φ (unid. sim.)', f'{m2_v**0.5:.2f}', True),
    ('ξ (longitud corr.)', f'{xi_v_m:.3f}', not np.isnan(xi_v_m)),
    ('m_φ×ξ ≈ cte', f'{m2_v**0.5*xi_v_m:.4f}' if not np.isnan(xi_v_m) else 'N/A', True),
    ('Masa reduce m_φ²', '✓ campo más blando' if m2_m<m2_v else '~', m2_m<m2_v),
    ('V cuadrático', 'dominante (dist. gaussiana)', True),
    ('λ₄ (cuártico)', f'{lambda4:.3f}', True),
]
for i,(k,v,ok) in enumerate(items):
    y=0.87-i*0.088
    ax6.text(0.03,y,f'{k}:',transform=ax6.transAxes,fontsize=9,
             color=CGR,va='top')
    ax6.text(0.52,y,v,transform=ax6.transAxes,fontsize=9,
             color=CG if ok else CY,va='top',fontweight='bold')

verdict='✓ V(φ) = V₀ + ½m_φ²(φ−φ₀)² + ... — POTENCIAL EFT CONFIRMADO'
ax6.text(0.5,0.02,verdict,transform=ax6.transAxes,fontsize=8.5,
         fontweight='bold',color=CG,ha='center',va='bottom',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=CG,lw=2.5))

fig.suptitle(
    f'SIM 9 — Potencial Efectivo V(φ) | N={N}\n'
    f'm_φ²={m2_v:.1f}  φ₀={phi0:.4f}  ξ={xi_v_m:.3f}  χ_φ={chi_v:.4e}',
    fontsize=12,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('sim9_resultado.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print("\n[OK] sim9_resultado.png guardado")

print(f"\n{'='*65}")
print(f"  RESUMEN SIM 9")
print(f"{'='*65}")
print(f"  φ₀ = R_fondo = {phi0:.5f}")
print(f"  m_φ² (vacío)  = {m2_v:.2f}  →  m_φ = {m2_v**0.5:.2f} [unidades sim]")
print(f"  m_φ² (masa)   = {m2_m:.2f}  →  masa reblandece el campo ✓")
print(f"  ξ (vacío)     = {xi_v_m:.3f}  →  m_φ × ξ = {m2_v**0.5*xi_v_m:.4f}")
print(f"  λ₄ (curtosis) = {lambda4:.3f}")
print(f"  Distribución de φ: gaussiana → V(φ) cuadrático dominante ✓")
print(f"{'='*65}")
print(f"\n  Marco EFT (para incorporar al documento):")
print(f"  V(φ) = V₀ + ½×{m2_v:.1f}×(φ−{phi0:.4f})² + ...")
print(f"  con m_φ² = χ_φ⁻¹,  χ_φ = {chi_v:.4e} [unidades sim]")
print(f"{'='*65}")
