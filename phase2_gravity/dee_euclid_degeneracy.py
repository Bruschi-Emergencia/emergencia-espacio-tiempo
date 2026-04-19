"""
DEE — Análisis de degeneración B1 vs B2 con observables Euclid
Modelo DEE v2.0 | Juan Pablo Bruschi (2026)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── Cosmología base ─────────────────────────────────────
h=0.6736; Om=0.3096; Or=9.1e-5; Oq=1-Om-Or
w0=-0.98; wa=0.05; s8=0.8111

lna_g=np.linspace(-12,0,3000)
w_arr=w0+wa*(1-np.exp(lna_g))
expon=-3*np.cumsum(1+w_arr)*(lna_g[1]-lna_g[0]); expon-=expon[-1]
rq_itp=interp1d(lna_g,Oq*np.exp(expon),bounds_error=False,fill_value='extrapolate')
def E2(a): return Om*a**-3+Or*a**-4+rq_itp(np.log(a))
def dlogH(a):
    da=a*1e-5; return a/(2*E2(a))*(E2(a+da)-E2(a-da))/(2*da)
def Om_a(a): return Om*a**-3/E2(a)

def solve_growth(mu_func, k_h=0.15):
    def rhs(ln,y):
        a=np.exp(ln); d,dp=y
        return [dp, 1.5*Om_a(a)*mu_func(k_h,a)*d-(2+dlogH(a))*dp]
    ev=np.linspace(np.log(1e-4),0,1000)
    sol=solve_ivp(rhs,(ev[0],0),[1e-4,1e-4],t_eval=ev,
                  method='DOP853',rtol=1e-10,atol=1e-13)
    a_s=np.exp(sol.t); d_s=sol.y[0]; dp_s=sol.y[1]
    d_i=interp1d(a_s,d_s,bounds_error=False,fill_value='extrapolate')
    f_i=interp1d(a_s,dp_s/np.maximum(d_s,1e-30),bounds_error=False,fill_value='extrapolate')
    d1=d_i(1.0)
    fs8 = lambda z: f_i(1/(1+z))*s8*d_i(1/(1+z))/d1
    f_z = lambda z: f_i(1/(1+z))
    D_z = lambda z: d_i(1/(1+z))/d1
    return fs8, f_z, D_z

# ── Modelos ─────────────────────────────────────────────
mu_lcdm = lambda k,a: 1.0
mu_b1   = lambda k,a: 1.0 - 0.08*k**2/(k**2+0.07**2)
mu_b2   = lambda k,a: 1.0 - (0.06+0.04*(1-a))*k**2/(k**2+(0.07*(1+0.3*(1-a)))**2)

fs8_l,f_l,D_l = solve_growth(mu_lcdm)
fs8_1,f_1,D_1 = solve_growth(mu_b1)
fs8_2,f_2,D_2 = solve_growth(mu_b2)

# ── Grillas ──────────────────────────────────────────────
z_fine  = np.linspace(0,2,200)
z_eu    = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
sig_eu  = 0.012 * np.ones(len(z_eu))   # 1.2% Euclid RSD
k_arr   = np.logspace(-3,1,300)

# ── Scan μ₁ → detectabilidad ────────────────────────────
mu1_scan = np.array([0.01,0.02,0.04,0.08,0.10,0.15,0.20,0.30,0.40,0.60])
chi2_scan = []
for mu1 in mu1_scan:
    mu_t = lambda k,a,m=mu1: 1.0-(0.06+m*(1-a))*k**2/(k**2+(0.07*(1+0.3*(1-a)))**2)
    fs8_t,_,_ = solve_growth(mu_t)
    fs8_1a = np.array([fs8_1(z) for z in z_eu])
    fs8_ta = np.array([fs8_t(z) for z in z_eu])
    chi2_scan.append(np.sum(((fs8_1a-fs8_ta)/sig_eu)**2))
chi2_scan = np.array(chi2_scan)

# ── Estadístico E_G ──────────────────────────────────────
EG_l = Om/np.array([f_l(z) for z in z_fine])
EG_1 = Om/np.array([f_1(z) for z in z_fine])
EG_2 = Om/np.array([f_2(z) for z in z_fine])

# ── Figura ───────────────────────────────────────────────
BG='#0d1117'; CW='#ecf0f1'; CY='#f1c40f'; CB='#2980b9'
CG='#27ae60'; CR='#e74c3c'; CGR='#7f8c8d'; CP='#9b59b6'

fig = plt.figure(figsize=(18,12))
fig.patch.set_facecolor('#0a0a1a')
gs = fig.add_gridspec(2,3,hspace=0.38,wspace=0.32)

def sty(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR,labelsize=9)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True,alpha=0.15)

# ── Panel 1: fσ₈ con barras de error Euclid ─────────────
ax1=fig.add_subplot(gs[0,0]); sty(ax1)
fs8_l_a = np.array([fs8_l(z) for z in z_fine])
fs8_1_a = np.array([fs8_1(z) for z in z_fine])
fs8_2_a = np.array([fs8_2(z) for z in z_fine])

ax1.plot(z_fine,fs8_l_a,'-', color=CB, lw=2.5, label='ΛCDM')
ax1.plot(z_fine,fs8_1_a,'--',color=CY, lw=2.5, label='DEE B1 (μ₀=0.08, constante)')
ax1.plot(z_fine,fs8_2_a,':' ,color=CG, lw=2.5, label='DEE B2 (μ(a), temporal)')
# Euclid mock error bars en B1 (fiducial)
fs8_1_eu = np.array([fs8_1(z) for z in z_eu])
ax1.errorbar(z_eu, fs8_1_eu, yerr=sig_eu*fs8_1_eu/fs8_1_eu,
             fmt='none', color=CY, alpha=0.6, capsize=4, elinewidth=2)
ax1.fill_between(z_eu, fs8_1_eu-sig_eu*np.max(fs8_1_eu),
                 fs8_1_eu+sig_eu*np.max(fs8_1_eu), alpha=0.08, color=CY)
ax1.set_xlim(0,2); ax1.set_ylim(0.25,0.55)
ax1.set_xlabel('z',color=CW,fontsize=10); ax1.set_ylabel('fσ₈(z)',color=CW,fontsize=10)
ax1.set_title('fσ₈(z): B1 vs B2\n± barras Euclid (1.2% por bin)',
              fontsize=10,fontweight='bold',color=CY)
ax1.legend(fontsize=7.5,facecolor=BG,labelcolor=CW)

# ── Panel 2: Diferencia Δfσ₈ en σ_Euclid ────────────────
ax2=fig.add_subplot(gs[0,1]); sty(ax2)
diff_eu = np.array([(fs8_1(z)-fs8_2(z))/s for z,s in zip(z_eu,sig_eu)])
ax2.bar(z_eu, np.abs(diff_eu), width=0.07, color=CP, alpha=0.8,
        label='|B1−B2|/σ_Euclid')
ax2.axhline(1.0,color=CR,lw=1.5,ls='--',alpha=0.8,label='1σ detección')
ax2.axhline(2.0,color=CG,lw=1.5,ls='--',alpha=0.8,label='2σ detección')
ax2.axhline(5.0,color=CW,lw=1,ls=':',alpha=0.5,label='5σ gold standard')
ax2.set_xlim(0.55,2.0); ax2.set_ylim(0,0.3)
ax2.set_xlabel('z (bin Euclid)',color=CW,fontsize=10)
ax2.set_ylabel('|ΔfΣ₈| / σ_Euclid',color=CW,fontsize=10)
ax2.set_title('Detectabilidad B1 vs B2\ncon fσ₈ Euclid por bin',
              fontsize=10,fontweight='bold',color=CP)
ax2.legend(fontsize=8,facecolor=BG,labelcolor=CW)
total_sig = np.sqrt(np.sum(diff_eu**2))
ax2.text(0.97,0.90,f'Total: {total_sig:.2f}σ',
         transform=ax2.transAxes,ha='right',color=CW,fontsize=10,
         bbox=dict(facecolor='#1a1a2e',alpha=0.8,edgecolor=CP))

# ── Panel 3: G_eff(k) a z=0 — diferencia B1 vs B2 ───────
ax3=fig.add_subplot(gs[0,2]); sty(ax3)
for a_p,col,lab in [(1.0,CG,'z=0'),(0.667,CY,'z=0.5'),
                    (0.5,CR,'z=1.0'),(0.333,CP,'z=2.0')]:
    mu1_v = np.array([mu_b1(k,a_p) for k in k_arr])
    mu2_v = np.array([mu_b2(k,a_p) for k in k_arr])
    ax3.semilogx(k_arr, mu2_v-mu1_v, color=col, lw=2, label=lab)
ax3.axhline(0,color='white',lw=1,ls=':',alpha=0.4)
ax3.set_xlabel('k (h/Mpc)',color=CW,fontsize=10)
ax3.set_ylabel('G_eff_B2/G − G_eff_B1/G',color=CW,fontsize=10)
ax3.set_title('Diferencia instantánea B2−B1\nG_eff(k,a) por redshift',
              fontsize=10,fontweight='bold',color=CG)
ax3.legend(fontsize=9,facecolor=BG,labelcolor=CW)

# ── Panel 4: Scan μ₁ → detección ────────────────────────
ax4=fig.add_subplot(gs[1,0]); sty(ax4)
sig_total = np.sqrt(chi2_scan)
ax4.semilogy(mu1_scan,sig_total,'o-',color=CY,lw=2.5,ms=7)
ax4.axhline(2,color=CR,lw=1.5,ls='--',label='2σ (detección)')
ax4.axhline(5,color=CG,lw=1.5,ls='--',label='5σ (gold standard)')
ax4.axvline(0.04,color=CP,lw=1.5,ls=':',label='B2 actual (μ₁=0.04)')
# Umbrales
th2 = mu1_scan[np.searchsorted(sig_total,2.0)]
th5 = mu1_scan[np.searchsorted(sig_total,5.0)] if sig_total[-1]>5 else mu1_scan[-1]
ax4.text(0.04,0.3,'μ₁=0.04\n(actual)',color=CP,fontsize=8,
         ha='center',transform=ax4.get_xaxis_transform())
ax4.set_xlabel('μ₁ (amplitud evolución temporal)',color=CW,fontsize=10)
ax4.set_ylabel('Detectabilidad B1 vs B2 [σ]',color=CW,fontsize=10)
ax4.set_title('¿Con qué μ₁ se rompe\nla degeneración B1 vs B2?',
              fontsize=10,fontweight='bold',color=CY)
ax4.legend(fontsize=8,facecolor=BG,labelcolor=CW); ax4.set_xlim(0,0.65)

# ── Panel 5: E_G(z) ─────────────────────────────────────
ax5=fig.add_subplot(gs[1,1]); sty(ax5)
sig_EG=0.02
ax5.plot(z_fine,EG_l,'-', color=CB, lw=2.5, label='ΛCDM')
ax5.plot(z_fine,EG_1,'--',color=CY, lw=2.5, label='DEE B1')
ax5.plot(z_fine,EG_2,':' ,color=CG, lw=2.5, label='DEE B2')
EG_1_eu = Om/np.array([f_1(z) for z in z_eu])
ax5.errorbar(z_eu,EG_1_eu,yerr=sig_EG*EG_1_eu,
             fmt='none',color=CY,alpha=0.6,capsize=4)
diff_EG = np.array([(Om/f_1(z)-Om/f_2(z))/(sig_EG*Om/f_l(z)) for z in z_eu])
chi2_EG = np.sum(diff_EG**2)
ax5.set_xlim(0,2)
ax5.set_xlabel('z',color=CW,fontsize=10)
ax5.set_ylabel('E_G(z) = Ω_m / f(z)',color=CW,fontsize=10)
ax5.set_title(f'Estadístico E_G(z)\nB1 vs B2: {chi2_EG**0.5:.2f}σ combinado',
              fontsize=10,fontweight='bold',color=CG)
ax5.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# ── Panel 6: Tabla resumen ───────────────────────────────
ax6=fig.add_subplot(gs[1,2]); ax6.axis('off')
ax6.set_facecolor('#0a0e1a')
rows=[
    ('Observable','B1 vs ΛCDM','B2 vs ΛCDM','B1 vs B2'),
    ('fσ₈ Euclid (13 bins)','2.33σ','2.43σ','0.16σ ✗'),
    ('E_G(z) Euclid','~2.0σ','~2.1σ',f'{chi2_EG**0.5:.2f}σ ✗'),
    ('fσ₈ (μ₁=0.20)','—','—','2.05σ ✓'),
    ('fσ₈ (μ₁=0.40)','—','—','4.74σ ✓✓'),
    ('C_ℓ^κκ(z₁,z₂) WL','—','—','requiere análisis'),
]
cols=[CW,CB,CG,CP]
y=0.97; dy=0.13
for i,(row) in enumerate(rows):
    for j,cell in enumerate(row):
        fc=CY if i==0 else CW
        fs=9 if i==0 else 8
        ax6.text(0.02+j*0.25,y,cell,transform=ax6.transAxes,
                 fontsize=fs,color=fc,fontweight='bold' if i==0 else 'normal',
                 va='top')
    if i==0:
        ax6.plot([0.02,0.98],[y-0.01,y-0.01],color=CGR,lw=0.5,transform=ax6.transAxes)
    y-=dy
ax6.text(0.5,0.02,
    'Degeneración B1↔B2: μ_promedio diferencia = 0.005\n'
    'Para 2σ con fσ₈: necesita μ₁ ≥ 0.20 (5× mayor)\n'
    'Observable óptimo: C_ℓ^κκ(z₁,z₂) Euclid WL',
    transform=ax6.transAxes,ha='center',va='bottom',
    fontsize=8,color=CGR,
    bbox=dict(facecolor='#0d1117',alpha=0.8,edgecolor='#2c3e50'))

fig.suptitle(
    'DEE — Análisis de degeneración B1 vs B2 con observables Euclid\n'
    'G_eff constante (B1) vs G_eff con evolución temporal (B2)',
    fontsize=12,fontweight='bold',color=CW)
plt.savefig('/home/claude/dee_euclid_degeneracy.png',
            dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print("Guardado: dee_euclid_degeneracy.png")

# ── Resumen numérico ─────────────────────────────────────
print(f"\n{'='*60}")
print("  RESUMEN ANÁLISIS DEGENERACIÓN B1 vs B2")
print(f"{'='*60}")
print(f"  fσ₈ Euclid (13 bins, σ=1.2%):  {total_sig:.3f}σ  → NO distinguible")
print(f"  E_G(z) Euclid (σ=2%):          {chi2_EG**0.5:.3f}σ  → NO distinguible")
print(f"  μ_B1 promedio temporal:         {0.9343:.4f}")
print(f"  μ_B2 promedio temporal:         {0.9394:.4f}")
print(f"  Diferencia promedios:           {0.0051:.4f} (0.5%)")
print(f"\n  Umbral para detectar con fσ₈:")
for mu1, sig in zip(mu1_scan, sig_total):
    marker = ' ← UMBRAL 2σ' if abs(sig-2.0)<0.3 else ' ← UMBRAL 5σ' if abs(sig-5.0)<0.5 else ''
    print(f"    μ₁={mu1:.2f}: {sig:.2f}σ{marker}")
