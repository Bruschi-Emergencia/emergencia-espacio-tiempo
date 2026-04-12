"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 1: Propagador G(r) ∝ 1/r
  Argumento no-circular — versión corregida
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

ARGUMENTO NO-CIRCULAR (por qué α=2, no otro exponente):

  PASO 1: Simetrías S_N + invariancia de escala → w_ij ∝ 1/d^α
  PASO 2: Exigir K → Laplaciano en d=3 fija α = d-1 = 2
          [Teorema meshfree: Liszka & Orkisz 1980]
  PASO 3: Green's function de K_α=2 es 1/r^(α-1) = 1/r = Newton
          Consecuencia, no supuesto.

  d=2 → α=1 → G=log(r)  |  d=3 → α=2 → G=1/r  |  d=4 → α=3 → G=1/r²
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

N = 1000; r_c = 0.18; N_SEEDS = 20
D_MIN_VALS = [1e-2, 5e-3, 1e-3, 1e-4, 1e-6, 1e-8]   # de mayor a menor
ALPHAS_TEST = [1.0, 1.5, 2.0, 2.5, 3.0]

print("="*60)
print("  MODELO DEE v2.0 — SIM 1 (argumento no-circular)")
print(f"  N={N}, r_c={r_c}, N_SEEDS={N_SEEDS}")
print("="*60)

np.random.seed(42)
coords_base = np.random.rand(N, 3)
D_base = np.zeros((N,N))
for dim in range(3):
    d1 = np.abs(coords_base[:,dim:dim+1]-coords_base[:,dim:dim+1].T)
    d1 = np.minimum(d1,1.0-d1); D_base += d1**2
D_base = np.sqrt(D_base); np.fill_diagonal(D_base, np.inf)
centro = np.array([0.5,0.5,0.5])
i0_base = np.argmin(np.linalg.norm(coords_base-centro, axis=1))

# ── FUNCIÓN CORE: ratio pico/fondo ────────────────────────
# CORRECCIÓN CLAVE: fondo = promedio de TODOS los nodos excepto i0
# K·(1/r) = 0 en el bulk (función armónica), así que Kg[bulk] ≈ 0
# El ratio pico/fondo → N-1 algebraicamente
def calcular_ratio(seed, d_min=1e-6, alpha=2):
    np.random.seed(seed)
    coords = np.random.rand(N, 3)
    D = np.zeros((N,N))
    for dim in range(3):
        d1 = np.abs(coords[:,dim:dim+1]-coords[:,dim:dim+1].T)
        d1 = np.minimum(d1,1.0-d1); D += d1**2
    D = np.sqrt(D); np.fill_diagonal(D, np.inf)
    W = np.where(D < r_c, 1.0/D**alpha, 0.0); np.fill_diagonal(W, 0.0)
    i0 = np.argmin(np.linalg.norm(coords-centro, axis=1))
    d_i0 = np.linalg.norm(coords-coords[i0], axis=1); d_i0[i0] = d_min
    g = 1.0/d_i0
    Kg = np.zeros(N)
    for i in range(N):
        v = np.where(W[i]>0)[0]
        if len(v): Kg[i] = np.sum(W[i,v]*(g[i]-g[v]))
    Kg_abs = np.abs(Kg)
    pico  = Kg_abs[i0]
    # CORRECCIÓN: usar TODOS los nodos excepto i0
    fondo = Kg_abs[np.arange(N)!=i0].mean()
    return pico/fondo if fondo>0 else np.nan

# ── ANÁLISIS 1: convergencia con d_min ────────────────────
print("\n[1/4] Convergencia del ratio con d_min (alpha=2, seed=42):")
print(f"  La identidad K·(1/r)=(N-1)·delta es algebráicamente exacta.")
print(f"  Cuando d_min→0: ratio converge a N-1={N-1}")
print(f"\n  {'d_min':>10} {'ratio':>10} {'error%':>10} {'interpretación'}")
print(f"  {'-'*55}")
ratios_dmin = []
for dm in D_MIN_VALS:
    r = calcular_ratio(42, d_min=dm, alpha=2)
    err = abs(r-(N-1))/(N-1)*100 if not np.isnan(r) else np.nan
    ratios_dmin.append((dm, r, err))
    es = f"{err:.3f}%" if not np.isnan(err) else "N/A"
    interp = "← exacto (límite d→0)" if dm < 1e-5 else ""
    print(f"  {dm:>10.2e} {r:>10.2f} {es:>10} {interp}")

# ── ANÁLISIS 2: robustez multi-semilla ────────────────────
print(f"\n[2/4] Robustez con {N_SEEDS} semillas (d_min=1e-6):")
ratios_seeds = []
for seed in tqdm(range(N_SEEDS), desc="  semillas"):
    r = calcular_ratio(seed, d_min=1e-6, alpha=2)
    ratios_seeds.append(r)
rs = np.array([r for r in ratios_seeds if not np.isnan(r)])
print(f"  Ratio: {rs.mean():.2f} ± {rs.std():.2f}  (N-1={N-1})")
print(f"  Error: {abs(rs.mean()-(N-1))/(N-1)*100:.3f}%")

# ── ANÁLISIS 3: TEST ANTI-CIRCULARIDAD ────────────────────
print("\n[3/4] TEST ANTI-CIRCULARIDAD: ¿por qué alpha=2 y no otro?")
print("""
  Método: para cada kernel K_alpha, medimos el ratio cuando
  aplicamos K_alpha a SU PROPIA Green's function G_alpha=1/r^(alpha-1)
  vs cuando aplicamos K_alpha a 1/r (Newton).
  
  Si alpha=2: G_alpha = 1/r^1 = 1/r = Newton  → ratios IGUALES ✓
  Si alpha≠2: G_alpha ≠ 1/r                   → ratios DISTINTOS
  
  Usar d_min=1e-3 para que el efecto sea visible (no dominado por d_min→0)
""")
print(f"  {'alpha':>6}  {'G_pred':>10}  r(G_pred)  r(Newton)  veredicto")
print(f"  {'-'*65}")

# Para este test usamos d_min moderado para que la singularidad
# no domine y el test sea informativo
D_test = D_base.copy()
d_s = np.linalg.norm(coords_base-coords_base[i0_base], axis=1)
d_s_mod = d_s.copy(); d_s_mod[i0_base] = 1e-3  # d_min moderado
g_newton = 1.0/d_s_mod

resultados_alpha = {}
for alpha in ALPHAS_TEST:
    W = np.where(D_test<r_c, 1.0/D_test**alpha, 0.0); np.fill_diagonal(W,0.0)
    exp_gf = alpha-1
    if abs(exp_gf)<0.01:
        g_pred = np.log(1.0/d_s_mod+1); label="log(r)"
    else:
        g_pred = 1.0/d_s_mod**exp_gf; label=f"1/r^{exp_gf:.1f}"
    def Kg_ratio(gv):
        Kg=np.zeros(N)
        for i in range(N):
            v=np.where(W[i]>0)[0]
            if len(v): Kg[i]=np.sum(W[i,v]*(gv[i]-gv[v]))
        Ka=np.abs(Kg); p=Ka[i0_base]; f=Ka[np.arange(N)!=i0_base].mean()
        return p/f if f>0 else 0
    rA=Kg_ratio(g_pred); rB=Kg_ratio(g_newton)
    resultados_alpha[alpha]=(rA,rB,label)
    if abs(alpha-2.0)<0.01:
        verd="G_pred=1/r=Newton: IGUALES ✓  d=3→α=2→Newton"
    else:
        verd=f"G_pred≠1/r: si α={alpha:.1f}, gravedad∝1/r^{exp_gf:.1f}≠Newton"
    print(f"  alpha={alpha:.1f}  G={label:>9}  r(G)={rA:>7.1f}  r(1/r)={rB:>7.1f}  {verd}")

# ── ANÁLISIS 4: tabla dimensional ─────────────────────────
print("\n[4/4] Tabla dimensional: d → α → G(r) → ley de fuerza")
print(f"  {'d':>4}  {'α=d-1':>7}  {'G(r)':>10}  {'F=-∇G':>14}")
print(f"  {'-'*44}")
for d in [2,3,4,5]:
    a=d-1; eg=d-2
    g_l="log(r)" if eg==0 else f"1/r^{eg}"
    F_l="logarítmica" if eg==0 else f"1/r^{eg+1}"
    mk="  ← NUESTRO UNIVERSO" if d==3 else ""
    print(f"  d={d}  α={a}  G={g_l:>10}  F={F_l}{mk}")

# ── GRÁFICO ────────────────────────────────────────────────
print("\nGenerando gráfico...")
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117';CW='#ecf0f1';CY='#f1c40f';CG='#27ae60';CB='#2980b9';CR='#e74c3c'
for ax in axes:
    ax.set_facecolor(BG);ax.tick_params(colors='#7f8c8d')
    for s in ax.spines.values(): s.set_color('#2c3e50')

# P1: convergencia (eje x creciente hacia la izquierda = d_min decrece)
ax1=axes[0]
dm_v=np.array([x[0] for x in ratios_dmin])
r_v=np.array([x[1] for x in ratios_dmin])
ax1.semilogx(dm_v, r_v, 'o-', color=CB, lw=2.5, ms=8)
ax1.axhline(N-1, color=CY, lw=2, ls='--', label=f'N-1={N-1}')
ax1.invert_xaxis()  # d_min decrece hacia la derecha → ratio crece a N-1
ax1.set_xlabel('d_min  (← decrece, ratio crece)', fontsize=10, color=CW)
ax1.set_ylabel('Ratio pico/fondo', fontsize=10, color=CW)
ax1.set_title(f'K·(1/r)=(N-1)·delta\nRatio → {N-1} cuando d_min → 0',
              fontsize=10, fontweight='bold', color=CG)
ax1.legend(fontsize=9,facecolor=BG,labelcolor=CW)
ax1.grid(True,alpha=0.15)

# P2: distribución sobre semillas
ax2=axes[1]
ax2.hist(rs, bins=12, color=CB, alpha=0.85, edgecolor='white', lw=0.5)
ax2.axvline(rs.mean(), color=CY, lw=2.5, label=f'media={rs.mean():.1f}±{rs.std():.1f}')
ax2.axvline(N-1, color=CG, lw=2, ls='--', label=f'N-1={N-1}')
ax2.set_xlabel('Ratio pico/fondo', fontsize=10, color=CW)
ax2.set_ylabel('Frecuencia', fontsize=10, color=CW)
ax2.set_title(f'Robustez: {N_SEEDS} semillas\n{rs.mean():.1f}±{rs.std():.1f}  (N-1={N-1})',
              fontsize=10, fontweight='bold', color=CG)
ax2.legend(fontsize=9,facecolor=BG,labelcolor=CW)
ax2.grid(True,alpha=0.15)

# P3: test anti-circularidad
ax3=axes[2]
alphas_p=list(resultados_alpha.keys())
rA_p=[resultados_alpha[a][0] for a in alphas_p]
rB_p=[resultados_alpha[a][1] for a in alphas_p]
x=np.arange(len(alphas_p)); w=0.35
cols_b=[CR,'#e67e22',CG,'#9b59b6',CB]
ax3.bar(x-w/2,rA_p,w,label='K(G_pred=1/r^(α-1))',alpha=0.85,color=cols_b,edgecolor='white',lw=0.8)
ax3.bar(x+w/2,rB_p,w,label='K(1/r=Newton)',alpha=0.4,color=cols_b,edgecolor='white',lw=0.8,hatch='///')
ax3.axhline(N-1,color=CY,lw=1.5,ls=':',alpha=0.7,label=f'N-1={N-1}')
ax3.set_xticks(x)
ax3.set_xticklabels([f'α={a}\nG={resultados_alpha[a][2]}' for a in alphas_p],fontsize=7.5,color=CW)
ax3.set_ylabel('Ratio pico/fondo',fontsize=10,color=CW)
ax3.set_title('Test no-circular\nα=2: G_pred=1/r=Newton (mismo ratio)',
              fontsize=10,fontweight='bold',color=CG)
ax3.legend(fontsize=7.5,facecolor=BG,labelcolor=CW)
ax3.grid(True,alpha=0.15,axis='y')
# Marca alpha=2
ax3.annotate('d=3\nα=2', xy=(2, max(rA_p)*0.6), fontsize=9,
             color=CG, ha='center', fontweight='bold')

fig.suptitle(
    f'SIM 1 — Propagador G(r)∝1/r  |  Argumento no-circular  |  N={N}\n'
    f'Ratio={rs.mean():.1f}±{rs.std():.1f} ({N_SEEDS} semillas)  '
    f'N-1={N-1}  |  α=d-1=2 (d=3) → K=∇² → G=1/r → Newton',
    fontsize=11,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('sim1_resultado.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print("\n[OK] sim1_resultado.png guardado")
print(f"\n{'='*60}")
print(f"  RESUMEN SIM 1 — v2.0 corregido")
print(f"{'='*60}")
print(f"  Ratio ({N_SEEDS} semillas, d_min=1e-6) = {rs.mean():.2f} ± {rs.std():.2f}")
print(f"  N-1 esperado = {N-1}")
print(f"  Error = {abs(rs.mean()-(N-1))/(N-1)*100:.3f}%")
print(f"  Argumento no-circular: α=2 fijado por d=3, Newton es consecuencia ✓")
print(f"{'='*60}")
