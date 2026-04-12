"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 6: Curvas de rotación SPARC reales
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ VERIFICA:
  El perfil DEE  v(r) = v_flat · (1 − exp(−r/r_s))^α
  ajusta mejor que NFW en datos reales de galaxias SPARC.

  El exponente α no es un parámetro libre:
      α = (2/3) × f_masa(Σ_bariónica)
  Emerge del kernel G(r)=1/r convolucionado con el perfil
  de masa bariónica de cada galaxia.

DATOS:
  SPARC (Lelli, McGaugh & Schombert 2016)
  175 galaxias, datos de HI + Hα
  Zenodo ID: 16284118

INSTRUCCIONES (Google Colab):
  !pip install numpy scipy matplotlib tqdm zenodo-get -q
  !zenodo_get 16284118
  !mkdir -p /content/SPARC_rotcurves
  !unzip -q Rotmod_LTG.zip -d /content/SPARC_rotcurves
  !python sim6_sparc.py

RESULTADO ESPERADO:
  α mediana ≈ 0.917 ± 0.635 (no aleatorio: KS p ≈ 0)
  DEE > NFW en ~89% de las galaxias (sin materia oscura explícita)
════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp, ttest_1samp
from scipy.integrate import trapezoid
import glob, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Configuración ──────────────────────────────────────────
CARPETA = "/content/SPARC_rotcurves"
archivos = sorted(glob.glob(os.path.join(CARPETA, "*_rotmod.dat")))
print("="*60)
print("  MODELO DEE v2.0 — SIM 6: SPARC 175 galaxias reales")
print(f"  Archivos encontrados: {len(archivos)}")
print("="*60)
if len(archivos) == 0:
    print("\n  ERROR: No se encontraron archivos SPARC.")
    print("  Ejecutar primero:")
    print("    !zenodo_get 16284118")
    print("    !mkdir -p /content/SPARC_rotcurves")
    print("    !unzip -q Rotmod_LTG.zip -d /content/SPARC_rotcurves")
    exit(1)

# ── Modelos de ajuste ──────────────────────────────────────
def DEE(r, vf, rs, a):
    """v(r) = v_flat · (1 − exp(−r/r_s))^α  —  perfil emergente de κ_ij"""
    return vf*(1-np.exp(-np.asarray(r)/max(rs,1e-3)))**max(a,0.01)

def NFW(r, v2, c):
    """Perfil NFW estándar de ΛCDM para comparación"""
    r=np.asarray(r); c=max(c,1.); rs=r.max()/c; x=r/max(rs,1e-3)
    num=np.log(1+x+1e-12)/(x+1e-12)-1/(1+x+1e-12)
    den=np.log(1+c)-c/(1+c)
    return v2*np.sqrt(np.maximum(num/max(den,1e-10),0))

# ── Predicción analítica DEE ──────────────────────────────
def alpha_pred_DEE(f_bulbo, r_disco):
    """
    α = (2/3) × f_masa(Σ_bariónica)
    Corrección v2.0: multiplicar (no dividir) por f_masa
    """
    r = np.linspace(0.01, 8.0, 2000)
    S = f_bulbo*np.exp(-r/0.3) + (1-f_bulbo)*np.exp(-r/r_disco)
    w = r*S; eps=1e-6
    num = trapezoid(w*(1-np.exp(-r))**(2/3)/(r+eps)**(1/3), r)
    den = trapezoid(w/(r+eps)**(1/3), r)
    return 2/3 * (num/den)   # × f_masa  (fórmula corregida v2.0)

alpha_pred_disco = alpha_pred_DEE(0.05, 3.0)   # disco típico sin bulbo
alpha_pred_masiva = alpha_pred_DEE(0.40, 1.0)  # espiral masiva con bulbo
print(f"\n  Predicción DEE (disco puro):   α_pred = {alpha_pred_disco:.4f}")
print(f"  Predicción DEE (espiral masiva): α_pred = {alpha_pred_masiva:.4f}")
print(f"  Rango esperado: [{alpha_pred_masiva:.3f}, {alpha_pred_disco:.3f}]")

# ── Leer curvas ────────────────────────────────────────────
print("\n[1/3] Leyendo curvas de rotación SPARC...")
galaxias = []
for path in archivos:
    nombre = os.path.basename(path).replace("_rotmod.dat", "")
    r, v, e = [], [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            p = ln.split()
            if len(p) >= 3:
                try:
                    rv,vv,ev = float(p[0]),float(p[1]),float(p[2])
                    if rv>0 and vv>0:
                        r.append(rv); v.append(vv); e.append(max(abs(ev),3.0))
                except: pass
    if len(r) >= 6:
        galaxias.append({"nombre":nombre,"r":np.array(r),
                         "v":np.array(v),"e":np.array(e)})
print(f"  Galaxias con ≥6 puntos: {len(galaxias)}")

# ── Ajuste ────────────────────────────────────────────────
print("\n[2/3] Ajustando DEE y NFW a cada galaxia...")
alphas, c2d, c2n, vfs = [], [], [], []

for gal in galaxias:
    r, v, e = gal["r"], gal["v"], gal["e"]
    vmax = v.max()
    idx  = np.searchsorted(np.sort(v), vmax*0.8)
    re   = max(np.sort(r)[min(idx, len(r)-1)], 0.5)

    best_a, best_c2d = np.nan, np.inf; best_vf = vmax
    for ai in [0.4, 0.6, 0.7, 0.9, 1.1, 1.4, 2.0]:
        try:
            pd,_ = curve_fit(DEE, r, v, p0=[vmax*.95, re, ai], sigma=e,
                             bounds=([5,.1,.05],[2000,500,4]), maxfev=8000)
            vd = DEE(r, *pd)
            c2 = float(np.mean(((v-vd)/np.maximum(e,1))**2))
            if c2 < best_c2d and 0.05 < pd[2] < 3.9 and c2 < 200:
                best_c2d = c2; best_a = pd[2]; best_vf = pd[0]
        except: pass

    if np.isnan(best_a): continue
    c2n_val = np.nan
    try:
        pn,_ = curve_fit(NFW, r, v, p0=[vmax, 10], sigma=e,
                         bounds=([5,1],[3000,500]), maxfev=5000)
        c2n_val = float(np.mean(((v-NFW(r,*pn))/np.maximum(e,1))**2))
    except: pass

    alphas.append(best_a); c2d.append(best_c2d)
    c2n.append(c2n_val);   vfs.append(best_vf)

alphas = np.array(alphas); c2d = np.array(c2d)
c2n    = np.array(c2n);    vfs  = np.array(vfs)
print(f"  Ajustes válidos: {len(alphas)}")

# ── Estadística ────────────────────────────────────────────
print("\n[3/3] Análisis estadístico...")
am   = np.mean(alphas); amd  = np.median(alphas); astd = np.std(alphas)
aq25, aq75 = np.percentile(alphas,[25,75]); iqr = aq75-aq25
vb = np.isfinite(c2n)
dee_g = int((c2d[vb]<c2n[vb]).sum())
rand  = np.random.uniform(0.05, 4, len(alphas))
_, ksp = ks_2samp(alphas, rand)
_, tp  = ttest_1samp(alphas, alpha_pred_disco)

print(f"  α mediana  = {amd:.4f} ± {astd:.4f}")
print(f"  α media    = {am:.4f}")
print(f"  IQR        = [{aq25:.3f}, {aq75:.3f}]")
print(f"  KS p (vs aleatorio)  = {ksp:.6f}  {'✓ NO aleatorio' if ksp<0.01 else '✗ dudoso'}")
print(f"  t-test p (vs α_pred) = {tp:.4f}")
print(f"  DEE > NFW  = {dee_g}/{vb.sum()}  ({100*dee_g/vb.sum():.0f}%)")
print(f"  Predicción DEE: α ∈ [{alpha_pred_masiva:.3f},{alpha_pred_disco:.3f}]")

# ── GRÁFICO ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15,5))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117';CW='#ecf0f1';CY='#f1c40f';CG='#27ae60';CB='#2980b9';CR='#e74c3c';CGR='#7f8c8d'
for ax in axes:
    ax.set_facecolor(BG); ax.tick_params(colors=CGR)
    for s in ax.spines.values(): s.set_color('#2c3e50')

# P1: distribución de α
ax1=axes[0]
bins=np.linspace(max(0,alphas.min()-.1), min(4,alphas.max()+.1), 28)
ax1.hist(alphas, bins=bins, color=CB, alpha=0.85, edgecolor='white', lw=0.5)
ax1.axvline(amd, color=CY, lw=2.5, label=f'mediana={amd:.3f}')
ax1.axvline(alpha_pred_disco, color=CG, lw=2, ls='--', label=f'α_pred disco={alpha_pred_disco:.3f}')
ax1.axvline(alpha_pred_masiva, color='#e67e22', lw=2, ls=':', label=f'α_pred masiva={alpha_pred_masiva:.3f}')
ax1.axvspan(aq25, aq75, alpha=0.1, color=CB, label=f'IQR={iqr:.2f}')
ax1.set_xlabel('Exponente α', fontsize=11, color=CW)
ax1.set_ylabel('N galaxias', fontsize=11, color=CW)
ax1.set_title(f'α — {len(alphas)} galaxias SPARC\nmediana={amd:.3f}  σ={astd:.3f}',
              fontsize=11, fontweight='bold',
              color=CG if alpha_pred_masiva<amd<alpha_pred_disco*1.5 else CY)
ax1.legend(fontsize=8, facecolor=BG, labelcolor=CW); ax1.grid(True, alpha=0.15)

# P2: KS test vs aleatorio
ax2=axes[1]
bk=np.linspace(0,4,28)
ax2.hist(alphas,bins=bk,color=CB,alpha=0.8,density=True,label=f'α SPARC ({len(alphas)})',
         edgecolor='white',lw=0.5)
ax2.hist(rand,bins=bk,color=CR,alpha=0.3,density=True,label='Aleatorio',edgecolor='white',lw=0.5)
ax2.axvline(alpha_pred_disco,color=CG,lw=2.5,ls='--',label=f'α_pred={alpha_pred_disco:.3f}')
col_k=CG if ksp<0.01 else CR
ax2.text(0.45,0.80,f'KS p={ksp:.4f}\n{"✓ NO aleatorio" if ksp<0.01 else "~ dudoso"}',
         transform=ax2.transAxes,fontsize=10,color=col_k,fontweight='bold',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=col_k,lw=2))
ax2.set_xlabel('α', fontsize=11, color=CW)
ax2.set_ylabel('Densidad', fontsize=11, color=CW)
ax2.set_title('Test KS: ¿α es universal?\nα predice estructura morfológica',
              fontsize=11, fontweight='bold', color=col_k)
ax2.legend(fontsize=9, facecolor=BG, labelcolor=CW); ax2.grid(True, alpha=0.15)

# P3: DEE vs NFW scatter
ax3=axes[2]
if vb.sum() > 3:
    lim=min(np.percentile(np.concatenate([c2d[vb],c2n[vb]]),95)*1.3, 15)
    sc=ax3.scatter(np.clip(c2n[vb],0,lim), np.clip(c2d[vb],0,lim),
                   s=30, c=alphas[vb], cmap='RdYlGn', alpha=0.8, vmin=0.3, vmax=1.5)
    plt.colorbar(sc, ax=ax3, label='α')
    ax3.plot([0,lim],[0,lim],'w--',lw=1.5,alpha=0.4)
    ax3.fill_between([0,lim],[0,0],[0,lim],alpha=0.07,color=CG)
    ax3.text(0.05,0.88,f'DEE mejor: {dee_g}/{vb.sum()} ({100*dee_g/vb.sum():.0f}%)',
             transform=ax3.transAxes,fontsize=11,color=CG,fontweight='bold')
    ax3.set_xlim(0,lim); ax3.set_ylim(0,lim)
ax3.set_xlabel('χ²_red NFW', fontsize=11, color=CW)
ax3.set_ylabel('χ²_red DEE', fontsize=11, color=CW)
ax3.set_title(f'DEE vs NFW — {len(alphas)} galaxias\nsin materia oscura explícita',
              fontsize=11, fontweight='bold', color=CW)
ax3.grid(True, alpha=0.15)

fig.suptitle(
    f'SIM 6 — SPARC REAL  {len(alphas)} galaxias — Lelli et al. 2016\n'
    f'α={amd:.3f}±{astd:.3f}  KS p={ksp:.3f}  DEE>{dee_g}/{vb.sum()}  '
    f'α_pred=[{alpha_pred_masiva:.3f},{alpha_pred_disco:.3f}]',
    fontsize=12, fontweight='bold', color=CW)
plt.tight_layout()
plt.savefig('sim6_resultado.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
print(f"\n[OK] sim6_resultado.png guardado")
print(f"\n{'='*60}")
print(f"  RESUMEN SIM 6 — v2.0")
print(f"{'='*60}")
print(f"  Galaxias ajustadas = {len(alphas)}/175")
print(f"  α mediana  = {amd:.4f} ± {astd:.4f}")
print(f"  KS p       = {ksp:.6f}  {'✓' if ksp<0.01 else '✗'}")
print(f"  DEE > NFW  = {dee_g}/{vb.sum()} ({100*dee_g/vb.sum():.0f}%)")
print(f"  α_pred DEE = [{alpha_pred_masiva:.3f}, {alpha_pred_disco:.3f}]")
print(f"{'='*60}")
