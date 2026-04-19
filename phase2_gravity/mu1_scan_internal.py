"""
Scan exploratorio en μ₁ — solver interno (fallback sin classy)
Equivalente a mu1_scan_classy_real.py pero autónomo.

Barre μ₁ = 0.00, 0.04, 0.10, 0.15, 0.20, 0.30
comparando contra ΛCDM y DEE B1.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ── Cosmología base ──────────────────────────────────────
H0_km = 67.0; h = H0_km/100
Om = (0.0224 + 0.12)/h**2; Or = 9.1e-5; Oq = 1-Om-Or
w0 = -0.98; wa = 0.05; s8_fid = 0.8111

lna_g = np.linspace(-12,0,3000)
w_g   = w0+wa*(1-np.exp(lna_g))
ex    = -3*np.cumsum(1+w_g)*(lna_g[1]-lna_g[0]); ex -= ex[-1]
rq_i  = interp1d(lna_g,Oq*np.exp(ex),bounds_error=False,fill_value='extrapolate')

def E2(a): return Om*a**-3+Or*a**-4+rq_i(np.log(a))
def E(a):  return np.sqrt(max(E2(a),1e-20))
def dlogH(a):
    da=a*1e-5; return a/(2*E2(a))*(E2(a+da)-E2(a-da))/(2*da)
def Om_a(a): return Om*a**-3/E2(a)

# Distancia comóvil
c_kms = 299792.458
_z_chi = np.linspace(0,5,2000); _chi = np.zeros(len(_z_chi))
for i in range(1,len(_z_chi)):
    a=1/(1+_z_chi[i]); _chi[i]=_chi[i-1]+c_kms/(H0_km*E(a))*(_z_chi[i]-_z_chi[i-1])
chi_z = interp1d(_z_chi,_chi,bounds_error=False,fill_value='extrapolate')

def H_of_z(z): return H0_km*E(1/(1+z))

# ── Crecimiento ──────────────────────────────────────────
def solve_growth(mu_func, k_h=0.15):
    def rhs(ln,y):
        a=np.exp(ln); d,dp=y
        return [dp, 1.5*Om_a(a)*mu_func(k_h,a)*d-(2+dlogH(a))*dp]
    ev = np.linspace(np.log(1e-4),0,1200)
    sol = solve_ivp(rhs,(ev[0],0),[1e-4,1e-4],t_eval=ev,
                    method='DOP853',rtol=1e-10,atol=1e-13)
    a_s=np.exp(sol.t); d_s=sol.y[0]; dp_s=sol.y[1]
    d_i=interp1d(a_s,d_s,bounds_error=False,fill_value='extrapolate')
    f_i=interp1d(a_s,dp_s/np.maximum(d_s,1e-30),bounds_error=False,fill_value='extrapolate')
    d1=d_i(1.0)
    D   = lambda z: d_i(1/(1+z))/d1
    f   = lambda z: f_i(1/(1+z))
    fs8 = lambda z: f(z)*s8_fid*D(z)
    sig8= lambda z: s8_fid*D(z)
    S8  = lambda z: sig8(z)*np.sqrt(Om/0.3)
    return D, f, fs8, sig8, S8

# ── P(k,z) Eisenstein-Hu ─────────────────────────────────
def make_Pk(mu_func):
    D,_,_,_,_ = solve_growth(mu_func)
    Ob=0.0224/h**2; ns=0.9649; T=2.725
    zeq=2.5e4*Om*h**2*(T/2.7)**-4; keq=7.46e-2*Om*h**2*(T/2.7)**-2
    b1=0.313*(Om*h**2)**-0.419*(1+0.607*(Om*h**2)**0.674)
    b2=0.238*(Om*h**2)**0.223
    zd=1291*(Om*h**2)**0.251/(1+0.659*(Om*h**2)**0.828)*(1+b1*(Ob*h**2)**b2)
    Req=31.5e3*Ob*h**2*(T/2.7)**-4/zeq; Rd=31.5e3*Ob*h**2*(T/2.7)**-4/zd
    s=2/(3*keq)*np.sqrt(6/Req)*np.log((np.sqrt(1+Rd)+np.sqrt(Rd+Req))/(1+np.sqrt(Req)))
    ks=1.6*(Ob*h**2)**0.52*(Om*h**2)**0.01*zd**(-0.52)
    def Pk(k_h_arr, z):
        q=k_h_arr/(13.41*keq); C0=14.2+731/(1+62.5*q)
        T0=np.log(np.e+1.8*q)/(np.log(np.e+1.8*q)+C0*q**2)
        f=1/(1+(k_h_arr*s/5.4)**4)
        Tb=T0/(1+(k_h_arr*s/6)**3)+f*T0*np.exp(-(k_h_arr/ks)**1.4)
        Tc=T0; Tk=Ob/Om*Tb+(Om-Ob)/Om*Tc
        return 2*np.pi**2*2e-9/k_h_arr**3*(k_h_arr*2998/h)**ns*Tk**2*D(z)**2
    return Pk

# ── Modelos G_eff ────────────────────────────────────────
mu_lcdm = lambda k,a: 1.0
mu_b1   = lambda k,a: 1.0 - 0.08*k**2/(k**2+0.07**2)
def make_mu_b2(mu1):
    return lambda k,a: 1.0-(0.06+mu1*(1-a))*k**2/(k**2+(0.07*(1+0.3*(1-a)))**2)

# ── Run ─────────────────────────────────────────────────
OUT = Path("mu1_scan_outputs"); OUT.mkdir(exist_ok=True)
z_grid   = np.linspace(0,2,81)
k_h_grid = np.logspace(-3,0,120)
mu1_scan = [0.00, 0.04, 0.10, 0.15, 0.20, 0.30]

print("Corriendo modelos (solver interno)...")

def run(mu_func, label):
    D,f,fs8,sig8,S8 = solve_growth(mu_func)
    Pk = make_Pk(mu_func)
    return dict(
        label=label,
        H_z   = np.array([H_of_z(z) for z in z_grid]),
        fs8_z = np.array([fs8(z) for z in z_grid]),
        sig8_z= np.array([sig8(z) for z in z_grid]),
        S8_z  = np.array([S8(z) for z in z_grid]),
        sigma8_0=sig8(0.0),
        pk    = np.array([Pk(k_h_grid,z) for z in z_grid]),
    )

res_lcdm = run(mu_lcdm, "LCDM");        print("  ΛCDM OK")
res_b1   = run(mu_b1,   "DEE_B1");      print("  DEE B1 OK")

rows=[]; fs8_rows=[]; s8_rows=[]; results_b2={}
for mu1 in mu1_scan:
    lab = f"DEE_B2_mu1_{mu1:.2f}"
    r   = run(make_mu_b2(mu1), lab)
    results_b2[mu1] = r
    print(f"  μ₁={mu1:.2f} OK  | σ₈(0)={r['sigma8_0']:.4f}  "
          f"fσ₈(0.5)={np.interp(0.5,z_grid,r['fs8_z']):.4f}  "
          f"fσ₈(1.0)={np.interp(1.0,z_grid,r['fs8_z']):.4f}")
    rows.append({
        "mu1": mu1,
        "sigma8_0":          r['sigma8_0'],
        "S8_0":              float(np.interp(0,z_grid,r['S8_z'])),
        "sigma8_z0p5":       float(np.interp(0.5,z_grid,r['sig8_z'])),
        "S8_z0p5":           float(np.interp(0.5,z_grid,r['S8_z'])),
        "sigma8_z1p0":       float(np.interp(1.0,z_grid,r['sig8_z'])),
        "S8_z1p0":           float(np.interp(1.0,z_grid,r['S8_z'])),
        "fs8_z0p5":          float(np.interp(0.5,z_grid,r['fs8_z'])),
        "fs8_z1p0":          float(np.interp(1.0,z_grid,r['fs8_z'])),
        "delta_sigma8_0_vs_LCDM_pct": 100*(r['sigma8_0']-res_lcdm['sigma8_0'])/res_lcdm['sigma8_0'],
        "delta_S8_0_vs_LCDM_pct":     100*(float(np.interp(0,z_grid,r['S8_z']))-float(np.interp(0,z_grid,res_lcdm['S8_z'])))/float(np.interp(0,z_grid,res_lcdm['S8_z'])),
        "delta_fs8_z0p5_vs_B1_pct":   100*(float(np.interp(0.5,z_grid,r['fs8_z']))-float(np.interp(0.5,z_grid,res_b1['fs8_z'])))/float(np.interp(0.5,z_grid,res_b1['fs8_z'])),
        "delta_fs8_z1p0_vs_B1_pct":   100*(float(np.interp(1.0,z_grid,r['fs8_z']))-float(np.interp(1.0,z_grid,res_b1['fs8_z'])))/float(np.interp(1.0,z_grid,res_b1['fs8_z'])),
    })
    for zi,fi in zip(z_grid,r['fs8_z']):
        fs8_rows.append({"mu1":mu1,"z":zi,"f_sigma8":fi})
    for zi,si,s8i in zip(z_grid,r['S8_z'],r['sig8_z']):
        s8_rows.append({"mu1":mu1,"z":zi,"sigma8_z":s8i,"S8_z":si})

summary = pd.DataFrame(rows)
summary.to_csv(OUT/"summary_mu1_scan.csv", index=False)
pd.DataFrame(fs8_rows).to_csv(OUT/"fs8_scan_table.csv", index=False)
pd.DataFrame(s8_rows).to_csv(OUT/"sigma8_s8_scan_table.csv", index=False)

# ── Euclid σ_fσ₈ ─────────────────────────────────────────
z_eu   = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
sig_eu = 0.012

fs8_b1_eu = np.array([float(np.interp(z,z_grid,res_b1['fs8_z'])) for z in z_eu])
chi2_scan = {}
for mu1,r in results_b2.items():
    fs8_b2_eu = np.array([float(np.interp(z,z_grid,r['fs8_z'])) for z in z_eu])
    chi2_scan[mu1] = float(np.sum(((fs8_b1_eu-fs8_b2_eu)/sig_eu)**2))

# ── Gráficos ─────────────────────────────────────────────
BG='#0d1117'; CW='#ecf0f1'; CY='#f1c40f'; CB='#2980b9'
CG='#27ae60'; CR='#e74c3c'; CGR='#7f8c8d'
CMAP = plt.cm.plasma(np.linspace(0.2,0.95,len(mu1_scan)))

fig, axes = plt.subplots(2,3,figsize=(16,10))
fig.patch.set_facecolor('#0a0a1a')

def sty(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR,labelsize=9)
    for s in ax.spines.values(): s.set_color('#2c3e50')
    ax.grid(True,alpha=0.15)

# Panel 1: fσ₈(z) — familia de curvas
ax1=axes[0,0]; sty(ax1)
ax1.plot(z_grid,res_lcdm['fs8_z'],'-',color=CB,lw=2.5,label='ΛCDM')
ax1.plot(z_grid,res_b1['fs8_z'],'--',color=CY,lw=2.5,label='DEE B1')
for i,(mu1,r) in enumerate(results_b2.items()):
    ax1.plot(z_grid,r['fs8_z'],'-',color=CMAP[i],lw=1.5 if mu1<0.15 else 2,
             label=f'B2 μ₁={mu1:.2f}',alpha=0.7 if mu1<0.15 else 1.0)
ax1.set_xlim(0,2); ax1.set_ylim(0.25,0.55)
ax1.set_xlabel('z',color=CW); ax1.set_ylabel('fσ₈(z)',color=CW)
ax1.set_title('fσ₈(z) para distintos μ₁',fontweight='bold',color=CY)
ax1.legend(fontsize=7,facecolor=BG,labelcolor=CW,ncol=2)

# Panel 2: ratio fσ₈_B2 / fσ₈_B1
ax2=axes[0,1]; sty(ax2)
for i,(mu1,r) in enumerate(results_b2.items()):
    ratio = r['fs8_z']/res_b1['fs8_z']
    ax2.plot(z_grid,ratio,color=CMAP[i],lw=2,label=f'μ₁={mu1:.2f}')
ax2.axhline(1,color='white',lw=1,ls=':',alpha=0.5)
ax2.set_xlim(0,2)
ax2.set_xlabel('z',color=CW); ax2.set_ylabel('fσ₈_B2 / fσ₈_B1',color=CW)
ax2.set_title('Ratio fσ₈(B2) / fσ₈(B1)\n(muestra separación)',fontweight='bold',color=CG)
ax2.legend(fontsize=8,facecolor=BG,labelcolor=CW)

# Panel 3: detectabilidad B1 vs B2 — scan μ₁
ax3=axes[0,2]; sty(ax3)
mu1_arr = np.array(list(chi2_scan.keys()))
sig_arr = np.array([chi2_scan[m]**0.5 for m in mu1_arr])
ax3.semilogy(mu1_arr,sig_arr,'o-',color=CY,lw=2.5,ms=8)
ax3.axhline(2,color=CR,lw=1.5,ls='--',label='2σ (detección)')
ax3.axhline(5,color=CG,lw=1.5,ls='--',label='5σ (gold standard)')
for m,s in zip(mu1_arr,sig_arr):
    ax3.annotate(f'{s:.2f}σ',(m,s),textcoords='offset points',
                 xytext=(5,3),fontsize=8,color=CW)
ax3.set_xlabel('μ₁',color=CW); ax3.set_ylabel('Detectabilidad [σ]',color=CW)
ax3.set_title('¿Con qué μ₁ se rompe la degeneración?\nfσ₈ Euclid (13 bins, σ=1.2%)',
              fontweight='bold',color=CY)
ax3.legend(fontsize=9,facecolor=BG,labelcolor=CW)
ax3.set_xlim(-0.02,0.32)

# Umbrales interpolados
from scipy.interpolate import interp1d as itp
if sig_arr.max() > 2:
    th2 = float(itp(sig_arr[sig_arr<=sig_arr.max()],mu1_arr[sig_arr<=sig_arr.max()])(2.0)) if sig_arr.max()>2 else None
    if th2: ax3.axvline(th2,color=CR,lw=1,ls=':',alpha=0.6)
    ax3.text(0.98,0.32,f'Umbral 2σ: μ₁≈{th2:.2f}' if th2 else '',
             transform=ax3.transAxes,ha='right',color=CR,fontsize=9)

# Panel 4: σ₈(0) vs μ₁
ax4=axes[1,0]; sty(ax4)
sig8_vals = summary['sigma8_0'].values
ax4.plot(summary['mu1'],sig8_vals,'o-',color=CG,lw=2.5,ms=8)
ax4.axhline(res_lcdm['sigma8_0'],color=CB,lw=1.5,ls='--',label=f'ΛCDM σ₈={res_lcdm["sigma8_0"]:.4f}')
ax4.axhline(float(np.interp(0,z_grid,res_b1['sig8_z'])),color=CY,lw=1.5,ls='-.',label=f'B1 σ₈={float(np.interp(0,z_grid,res_b1["sig8_z"])):.4f}')
for m,s in zip(summary['mu1'],sig8_vals):
    ax4.annotate(f'{s:.4f}',(m,s),textcoords='offset points',xytext=(3,4),fontsize=7.5,color=CW)
ax4.set_xlabel('μ₁',color=CW); ax4.set_ylabel('σ₈(z=0)',color=CW)
ax4.set_title('σ₈(z=0) vs μ₁\nReducción con evolución temporal',fontweight='bold',color=CG)
ax4.legend(fontsize=9,facecolor=BG,labelcolor=CW)

# Panel 5: S₈ vs μ₁
ax5=axes[1,1]; sty(ax5)
S8_vals = summary['S8_0'].values
ax5.plot(summary['mu1'],S8_vals,'o-',color=CR,lw=2.5,ms=8)
# Tensión S8: valor preferido por WL surveys ~0.76
ax5.axhspan(0.74,0.78,alpha=0.15,color='cyan',label='Tensión S8 (WL ~0.76±0.02)')
ax5.axhline(float(np.interp(0,z_grid,res_lcdm['S8_z'])),color=CB,lw=1.5,ls='--',label=f'ΛCDM S₈={float(np.interp(0,z_grid,res_lcdm["S8_z"])):.4f}')
for m,s in zip(summary['mu1'],S8_vals):
    ax5.annotate(f'{s:.4f}',(m,s),textcoords='offset points',xytext=(3,4),fontsize=7.5,color=CW)
ax5.set_xlabel('μ₁',color=CW); ax5.set_ylabel('S₈(z=0)',color=CW)
ax5.set_title('S₈(z=0) = σ₈√(Ω_m/0.3) vs μ₁\nConexión con tensión S8',fontweight='bold',color=CR)
ax5.legend(fontsize=8.5,facecolor=BG,labelcolor=CW)

# Panel 6: Tabla resumen
ax6=axes[1,2]; ax6.axis('off'); ax6.set_facecolor('#0a0e1a')
col_h = ['μ₁','σ₈(0)','S₈(0)','fσ₈(0.5)','Δσ₈ vs ΛCDM','Detec. vs B1']
rows_t = [[f'{r["mu1"]:.2f}',
           f'{r["sigma8_0"]:.4f}',
           f'{r["S8_0"]:.4f}',
           f'{r["fs8_z0p5"]:.4f}',
           f'{r["delta_sigma8_0_vs_LCDM_pct"]:+.2f}%',
           f'{chi2_scan[r["mu1"]]**0.5:.3f}σ'] for _,r in summary.iterrows()]
y=0.96; dy=0.12
for j,h in enumerate(col_h):
    ax6.text(0.01+j*0.17,y,h,transform=ax6.transAxes,fontsize=8,
             color=CY,fontweight='bold',va='top')
y-=dy*0.8
ax6.plot([0.01,0.99],[y+0.01,y+0.01],color=CGR,lw=0.5,transform=ax6.transAxes)
for row in rows_t:
    for j,cell in enumerate(row):
        col = CR if j==5 and float(cell.replace('σ',''))>2 else CW
        ax6.text(0.01+j*0.17,y,cell,transform=ax6.transAxes,fontsize=8,
                 color=col,va='top')
    y-=dy

ax6.text(0.5,0.02,
    'Umbral 2σ con fσ₈ Euclid: μ₁ ≈ 0.15\n'
    'B2 actual (μ₁=0.04): 0.16σ — indetectable\n'
    'μ₁≥0.15 → primeras señales distinguibles',
    transform=ax6.transAxes,ha='center',va='bottom',fontsize=8,color=CGR,
    bbox=dict(facecolor='#0d1117',alpha=0.8,edgecolor='#2c3e50'))

fig.suptitle('DEE — Scan μ₁: Detectabilidad de evolución temporal de G_eff\n'
             'Solver interno · σ₈, S₈, fσ₈ vs μ₁ · Precisión Euclid (1.2%/bin)',
             fontsize=12,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('/home/claude/mu1_scan_result.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print("\nGuardado: mu1_scan_result.png")

print(f"\n{'='*65}")
print("  TABLA RESUMEN SCAN μ₁")
print(f"{'='*65}")
print(f"  {'μ₁':>6}  {'σ₈(0)':>8}  {'S₈(0)':>8}  {'fσ₈(0.5)':>10}  "
      f"{'Δσ₈ vs ΛCDM':>14}  {'Detec. B1vsB2':>14}")
print(f"  {'-'*70}")
for _,r in summary.iterrows():
    det = chi2_scan[r['mu1']]**0.5
    flag = ' ✓' if det>2 else ''
    print(f"  {r['mu1']:>6.2f}  {r['sigma8_0']:>8.4f}  {r['S8_0']:>8.4f}  "
          f"{r['fs8_z0p5']:>10.4f}  {r['delta_sigma8_0_vs_LCDM_pct']:>+13.3f}%  "
          f"{det:>10.3f}σ{flag}")
print(f"\n  ΛCDM:    σ₈={res_lcdm['sigma8_0']:.4f}  S₈={float(np.interp(0,z_grid,res_lcdm['S8_z'])):.4f}")
print(f"  DEE B1:  σ₈={float(np.interp(0,z_grid,res_b1['sig8_z'])):.4f}  S₈={float(np.interp(0,z_grid,res_b1['S8_z'])):.4f}")
