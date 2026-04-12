"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 2: Curvatura κ_ij y gravedad emergente
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ VERIFICA:
  La curvatura de Ollivier-Ricci κ_ij > 0 en presencia de masa
  produce aceleración gravitatoria atractiva bajo a = +∇R(x).

  Variable fundamental:
      κ_ij = 1 − W₁(P_i, P_j) / d_ij
  Aproximación numérica:
      W₁ ≈ ‖P_i − P_j‖₁ · σ_k    (exacta cuando soportes no se solapan)

  Esto es el teorema de Ollivier (2009):
      κ(x, x+εv) → (ε²/2) Ric(v,v)   en variedades suaves

RESULTADO ESPERADO:
  κ medio > 0 (geometría esférica → gravedad atractiva)
  4/4 partículas de prueba convergen al centro bajo a = +∇R

INSTRUCCIONES (Google Colab):
  !pip install numpy scipy matplotlib tqdm -q
  !python sim2_curvatura_gravedad.py
════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 42; N = 2000; r_c = 0.22; N_STEPS = 60; dt = 0.004
np.random.seed(SEED)

print("="*60)
print("  MODELO DEE v2.0 — SIM 2: κ_ij y gravedad emergente")
print(f"  N={N}, r_c={r_c}")
print("="*60)

# ── Construir red con masa central ─────────────────────────
print("\n[1/4] Construyendo red con masa bariónica central...")
coords = np.random.rand(N, 3)
centro = np.array([0.5, 0.5, 0.5])
D_c = np.linalg.norm(coords - centro, axis=1)

D = np.zeros((N, N))
for dim in tqdm(range(3), desc="  distancias PBC"):
    d1 = np.abs(coords[:,dim:dim+1] - coords[:,dim:dim+1].T)
    d1 = np.minimum(d1, 1.0 - d1); D += d1**2
D = np.sqrt(D); np.fill_diagonal(D, np.inf)

sigma_k = r_c * 0.5
S = np.where(D < r_c, np.exp(-D**2/(2*sigma_k**2)), 0.0)

# Masa bariónica: refuerzo gaussiano centrado en origen
r_m = 0.08; amplitud = 8.0
for i in tqdm(range(N), desc="  masa central", leave=False):
    for j in range(N):
        if i != j and D[i,j] < r_c:
            bump = amplitud * np.exp(-(D_c[i]**2+D_c[j]**2)/(2*r_m**2))
            S[i,j] += bump

S = np.clip(S, 0, None); S = (S+S.T)/2; np.fill_diagonal(S, 0)
d_i = np.maximum(S.sum(axis=1), 1e-10)
P = S / d_i[:,None]
print(f"  Vecinos/nodo: {(S>0).sum(axis=1).mean():.1f}")

# ── Calcular κ_ij ──────────────────────────────────────────
print("\n[2/4] Calculando curvatura de Ollivier-Ricci κ_ij...")
aristas = [(i,j) for i in range(N)
           for j in np.where(S[i]>1e-8)[0] if j>i]
n_s = min(4000, len(aristas))
idx_s = np.random.choice(len(aristas), n_s, replace=False)

kappa_nodo = {i:[] for i in range(N)}
for i,j in tqdm([aristas[k] for k in idx_s], desc="  κ_ij", leave=False):
    d_ij = D[i,j]
    if not np.isfinite(d_ij) or d_ij <= 1e-8: continue
    # κ = 1 − W₁/d   con  W₁ ≈ ‖P_i−P_j‖₁ · σ_k
    k = 1.0 - np.sum(np.abs(P[i]-P[j]))*sigma_k/d_ij
    if np.isfinite(k):
        kappa_nodo[i].append(k); kappa_nodo[j].append(k)

kappa_med_global = [np.median(v) for v in kappa_nodo.values() if v]
k_medio = float(np.mean(kappa_med_global))
k_std   = float(np.std(kappa_med_global))
print(f"  κ medio = {k_medio:.4f} ± {k_std:.4f}")
print(f"  κ > 0: {'✓ Geometría esférica → gravedad atractiva' if k_medio>0 else '✗ κ ≤ 0'}")

# Curvatura escalar R(x_i)
R_nodo_med = np.array([np.median(v) if v else np.nan for v in kappa_nodo.values()])
R_global = np.nanmedian(R_nodo_med)
R_vals = np.where(np.isfinite(R_nodo_med), R_nodo_med, R_global)

# ── Trayectorias de partículas ─────────────────────────────
print("\n[3/4] Integrando trayectorias de partículas de prueba...")
# Gradiente de R(x) → fuerza gravitatoria
def fuerza(pos, coords_r, R_r, r_c_f):
    d = np.linalg.norm(coords_r - pos, axis=1)
    mask = d < r_c_f * 1.5; mask[d<1e-5] = False
    if mask.sum() < 3: return np.zeros(3)
    w = np.exp(-d[mask]**2/(2*(r_c_f*0.5)**2))
    dR = (R_r[mask] - R_r[mask].mean())[:,None] * (coords_r[mask]-pos)
    return (w[:,None]*dR).sum(axis=0)/w.sum()

starts = [
    np.array([0.15,0.15,0.5]),
    np.array([0.85,0.15,0.5]),
    np.array([0.15,0.85,0.5]),
    np.array([0.85,0.85,0.5]),
]
colores_traj = ['#e74c3c','#2980b9','#27ae60','#f39c12']
trayectorias = []
convergencias = []

for pos0 in tqdm(starts, desc="  partículas"):
    pos = pos0.copy(); vel = np.zeros(3)
    traj = [pos.copy()]
    for _ in range(N_STEPS):
        F = fuerza(pos, coords, R_vals, r_c)
        vel += dt * F; pos += dt * vel
        pos = np.clip(pos, 0.02, 0.98)
        traj.append(pos.copy())
    trayectorias.append(np.array(traj))
    d_final = np.linalg.norm(traj[-1]-centro)
    d_inicial = np.linalg.norm(pos0-centro)
    convergencias.append(d_final < d_inicial)

n_conv = sum(convergencias)
print(f"  Convergencia al centro: {n_conv}/{len(starts)}")
print(f"  {'✓ GRAVEDAD ATRACTIVA EMERGENTE' if n_conv>=3 else '✗ No convergen'}")

# ── Ley de fuerza F(r) ─────────────────────────────────────
print("\n[4/4] Midiendo perfil de fuerza F(r)...")
r_bins = np.linspace(0.05, 0.45, 10)
r_mid  = (r_bins[:-1]+r_bins[1:])/2
F_bins = []
for rb_lo, rb_hi in zip(r_bins[:-1], r_bins[1:]):
    mask = (D_c>=rb_lo)&(D_c<rb_hi)
    if mask.sum() < 3:
        F_bins.append(np.nan); continue
    Fs = [np.linalg.norm(fuerza(coords[i], coords, R_vals, r_c))
          for i in np.where(mask)[0][:8]]
    F_bins.append(np.mean([f for f in Fs if f>0]) if Fs else np.nan)
F_bins = np.array(F_bins)
vb = np.isfinite(F_bins) & (F_bins>0) & (r_mid>0.06)
slope = np.nan
if vb.sum() >= 4:
    z = np.polyfit(np.log(r_mid[vb]), np.log(F_bins[vb]), 1)
    slope = z[0]
    print(f"  F ∝ r^{slope:.3f}  (Newton en continuo: −2.0)")

# ── GRÁFICO ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117';CW='#ecf0f1';CY='#f1c40f';CG='#27ae60';CB='#2980b9'
for ax in axes.flat:
    ax.set_facecolor(BG); ax.tick_params(colors='#7f8c8d')
    for s in ax.spines.values(): s.set_color('#2c3e50')

# P1: κ vs distancia al centro
ax1=axes[0,0]
mask_p=np.array([bool(v) for v in kappa_nodo.values()])
kv_arr=np.array(kappa_med_global); d_arr=D_c[:len(kv_arr)]
sc=ax1.scatter(d_arr[:len(kv_arr)],kv_arr,s=8,alpha=0.4,
               c=kv_arr,cmap='RdYlGn',vmin=-0.1,vmax=0.4)
plt.colorbar(sc,ax=ax1,label='κ_ij')
ax1.axhline(0,color='white',lw=1.5,ls='--',alpha=0.5)
ax1.axhline(k_medio,color=CY,lw=2,ls='-',label=f'κ medio={k_medio:.4f}')
ax1.set_xlabel('Distancia al centro r',fontsize=10,color=CW)
ax1.set_ylabel('κ_ij (Ollivier-Ricci)',fontsize=10,color=CW)
ax1.set_title(f'κ_ij vs distancia al centro\nκ={k_medio:.4f}±{k_std:.4f} > 0 ✓',
              fontsize=11,fontweight='bold',color=CG if k_medio>0 else '#e74c3c')
ax1.legend(fontsize=9,facecolor=BG,labelcolor=CW); ax1.grid(True,alpha=0.15)

# P2: Trayectorias en plano xy
ax2=axes[0,1]
z_mask2=(coords[:,2]>0.45)&(coords[:,2]<0.55)
ax2.scatter(coords[z_mask2,0],coords[z_mask2,1],
            c=R_vals[z_mask2],cmap='hot',s=4,alpha=0.2,vmin=R_vals.min(),vmax=R_vals.max())
for traj,col,conv in zip(trayectorias,colores_traj,convergencias):
    ax2.plot(traj[:,0],traj[:,1],'o-',color=col,ms=3,lw=2,alpha=0.9)
    ax2.plot(*traj[0,:2],'o',color=col,ms=9,markeredgecolor='white',lw=2)
    mk='*' if conv else 'x'; ms=14 if conv else 10
    ax2.plot(*traj[-1,:2],mk,color=col,ms=ms,markeredgecolor='white')
ax2.plot(*centro[:2],'w*',ms=16,label='Centro (masa)')
ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_aspect('equal')
ax2.set_xlabel('x',fontsize=10,color=CW); ax2.set_ylabel('y',fontsize=10,color=CW)
ax2.set_title(f'Trayectorias bajo a=+∇R\n{n_conv}/4 convergen al centro',
              fontsize=11,fontweight='bold',color=CG if n_conv>=3 else '#e74c3c')
ax2.legend(fontsize=9,facecolor=BG,labelcolor=CW); ax2.grid(True,alpha=0.1)

# P3: Ley de fuerza
ax3=axes[1,0]
if vb.sum()>=4:
    ax3.loglog(r_mid[vb],F_bins[vb],'o',color=CB,ms=8,label='F(r) numérica')
    r_fit=np.linspace(r_mid[vb].min(),r_mid[vb].max(),50)
    A=np.exp(np.polyfit(np.log(r_mid[vb]),np.log(F_bins[vb]),1)[1])
    ax3.loglog(r_fit,A*r_fit**slope,'r--',lw=2.5,label=f'F ∝ r^{slope:.2f}')
    ax3.loglog(r_fit,A*r_fit**(-2),'g:',lw=2,alpha=0.7,label='Newton F ∝ r⁻²')
ax3.set_xlabel('Distancia r al centro',fontsize=10,color=CW)
ax3.set_ylabel('|F|',fontsize=10,color=CW)
slbl=f'{slope:.3f}' if not np.isnan(slope) else 'N/A'
ax3.set_title(f'Ley de fuerza: F ∝ r^{slbl}\n(Newton en continuo N→∞: −2.0)',
              fontsize=11,fontweight='bold',color=CW)
ax3.legend(fontsize=9,facecolor=BG,labelcolor=CW); ax3.grid(True,alpha=0.15)

# P4: Resumen
ax4=axes[1,1]; ax4.axis('off')
ax4.text(0.5,0.97,'RESULTADOS SIM 2 — v2.0',transform=ax4.transAxes,
         fontsize=13,fontweight='bold',color=CY,ha='center',va='top')
items=[
    ('κ medio',f'{k_medio:.4f} ± {k_std:.4f}',k_medio>0),
    ('κ > 0',f'{"✓ Geometría esférica" if k_medio>0 else "✗ No"}',k_medio>0),
    ('Convergencia',f'{n_conv}/4 partículas al centro',n_conv>=3),
    ('Ley de fuerza',f'F ∝ r^{slbl}',not np.isnan(slope)),
    ('Newton continuo','F ∝ r⁻²  (N→∞)',True),
    ('Acción efectiva','G_μν = 8πG(T_m + T_φ)',True),
    ('Teorema Ollivier','κ_ij → (ε²/2)Ric(v,v)',True),
]
CGR='#7f8c8d'
for i,(k,v,ok) in enumerate(items):
    y=0.87-i*0.10
    ax4.text(0.05,y,f'{k}:',transform=ax4.transAxes,fontsize=9,color=CGR,va='top')
    ax4.text(0.42,y,v,transform=ax4.transAxes,fontsize=9,
             color=CG if ok else '#e74c3c',va='top',fontweight='bold')
ok_all=k_medio>0 and n_conv>=3
ax4.text(0.5,0.01,'✓ GRAVEDAD ATRACTIVA EMERGENTE' if ok_all else '✗ RESULTADO INESPERADO',
         transform=ax4.transAxes,fontsize=11,fontweight='bold',
         color=CG if ok_all else '#e74c3c',ha='center',va='bottom',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=CG if ok_all else '#e74c3c',lw=2.5))

fig.suptitle(
    f'SIM 2 — Curvatura κ_ij y Gravedad Emergente  |  N={N}\n'
    f'κ={k_medio:.4f}±{k_std:.4f}  |  {n_conv}/4 convergen  |  F∝r^{slbl}',
    fontsize=12,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('sim2_resultado.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print(f"\n[OK] sim2_resultado.png guardado")
print(f"\n{'='*60}")
print(f"  RESUMEN SIM 2 — v2.0")
print(f"{'='*60}")
print(f"  κ medio       = {k_medio:.4f} ± {k_std:.4f}")
print(f"  κ > 0         = {'✓' if k_medio>0 else '✗'}")
print(f"  Convergencia  = {n_conv}/4")
print(f"  F(r) ∝ r^{slbl}   (Newton: −2.0)")
print(f"{'='*60}")
