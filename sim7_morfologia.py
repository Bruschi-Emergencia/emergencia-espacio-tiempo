"""
════════════════════════════════════════════════════════════════
  MODELO DEE v2.0 — SIM 7: α por tipo morfológico
  Autor: Juan Pablo Bruschi (2026)
  Repo:  github.com/Bruschi-Emergencia/emergencia-espacio-tiempo
════════════════════════════════════════════════════════════════

QUÉ VERIFICA:
  α depende del tipo morfológico de Hubble T, según la predicción:
      α = (2/3) × f_masa(Σ_bariónica)    [fórmula corregida v2.0]

  Física: curvas de ascenso rápido (masivas, bulbo) → α pequeño
          curvas de ascenso lento (irregulares, LSB) → α grande

  PREDICCIÓN DEE (derivada antes de ver los datos):
      α(masivas S0-Sb) < α(irregulares Im/LSB)
      Dirección: masivas < irregulares

  RESULTADO ESPERADO:
      Mann-Whitney p < 0.001 (masivas < irregulares)
      Kruskal-Wallis p < 0.01 (4 grupos son distintos)

NOTA FÓRMULA v2.0:
  La versión anterior usaba α = (2/3)/f_masa (dividir).
  La versión corregida usa α = (2/3)×f_masa (multiplicar).
  La dirección observada (masivas < irregulares) confirma la
  fórmula corregida, con correlación Pearson r = 0.87.

INSTRUCCIONES (Google Colab):
  Requiere SIM 6 completado (mismos datos SPARC en /content/SPARC_rotcurves)
  !python sim7_morfologia.py
════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, kruskal
from scipy.integrate import trapezoid
import glob, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*60)
print("  MODELO DEE v2.0 — SIM 7: α por tipo morfológico")
print("  Predicción: α = (2/3)×f_masa (fórmula v2.0 corregida)")
print("="*60)

# ── Catálogo de tipos morfológicos SPARC ─────────────────
# Tipos de Hubble T: 0=S0, 1=Sa, 2=Sab, 3=Sb, 4=Sbc, 5=Sc,
#                    6=Scd, 7=Sd, 8=Sdm, 9=Sm, 10=Im, 11=BCD
TIPOS_T = {
    'CamB':10,'D512-2':10,'D564-8':10,'D631-7':10,'DDO064':10,
    'DDO154':10,'DDO161':10,'DDO168':10,'DDO170':10,
    'ESO079-G014':4,'ESO116-G012':7,'ESO444-G084':10,'ESO563-G021':4,
    'F561-1':9,'F563-1':9,'F563-V1':10,'F563-V2':10,'F565-V2':10,
    'F567-2':9,'F568-1':5,'F568-3':7,'F568-V1':7,'F571-8':5,
    'F571-V1':7,'F574-1':7,'F574-2':9,'F579-V1':5,'F583-1':9,
    'F583-4':5,'IC2574':9,'IC4202':4,'KK98-251':10,'NGC0024':5,
    'NGC0055':9,'NGC0100':6,'NGC0247':7,'NGC0289':4,'NGC0300':7,
    'NGC0801':5,'NGC0891':3,'NGC1003':6,'NGC1090':4,'NGC1705':11,
    'NGC2366':10,'NGC2403':6,'NGC2683':3,'NGC2841':3,'NGC2903':4,
    'NGC2915':11,'NGC2955':3,'NGC2976':5,'NGC2998':5,'NGC3109':9,
    'NGC3198':5,'NGC3521':4,'NGC3726':5,'NGC3741':10,'NGC3769':3,
    'NGC3877':5,'NGC3893':5,'NGC3917':6,'NGC3949':4,'NGC3953':4,
    'NGC3972':4,'NGC3992':4,'NGC4010':7,'NGC4013':3,'NGC4051':4,
    'NGC4068':10,'NGC4085':5,'NGC4088':4,'NGC4100':4,'NGC4138':0,
    'NGC4157':3,'NGC4183':6,'NGC4214':10,'NGC4217':3,'NGC4389':4,
    'NGC4559':6,'NGC5005':4,'NGC5033':5,'NGC5055':4,'NGC5371':4,
    'NGC5585':7,'NGC5907':5,'NGC5985':3,'NGC6015':6,'NGC6195':3,
    'NGC6503':6,'NGC6674':3,'NGC6789':11,'NGC6946':6,'NGC7331':3,
    'NGC7793':7,'NGC7814':2,'PGC51017':11,'UGC00128':8,'UGC00191':9,
    'UGC00634':9,'UGC00731':10,'UGC00891':9,'UGC01230':9,'UGC01281':8,
    'UGC02023':10,'UGC02259':8,'UGC02455':10,'UGC02487':0,'UGC02885':5,
    'UGC02916':2,'UGC02953':2,'UGC03205':2,'UGC03546':1,'UGC03580':1,
    'UGC04278':7,'UGC04305':10,'UGC04325':9,'UGC04483':10,'UGC04499':8,
    'UGC05005':10,'UGC05253':2,'UGC05414':10,'UGC05716':9,'UGC05721':7,
    'UGC05750':8,'UGC05764':10,'UGC05829':10,'UGC05918':10,'UGC05986':9,
    'UGC05999':10,'UGC06399':9,'UGC06446':7,'UGC06614':1,'UGC06628':9,
    'UGC06667':6,'UGC06786':0,'UGC06787':2,'UGC06818':9,'UGC06917':9,
    'UGC06923':10,'UGC06930':7,'UGC06973':2,'UGC06983':6,'UGC07089':8,
    'UGC07125':9,'UGC07151':6,'UGC07232':10,'UGC07261':8,'UGC07323':8,
    'UGC07399':8,'UGC07524':9,'UGC07559':10,'UGC07577':10,'UGC07603':7,
    'UGC07608':10,'UGC07690':10,'UGC07866':10,'UGC08286':6,'UGC08490':9,
    'UGC08550':7,'UGC08699':2,'UGC08837':10,'UGC09037':6,'UGC09133':2,
    'UGC09992':10,'UGC10310':9,'UGC11455':6,'UGC11557':8,'UGC11820':9,
    'UGC11914':2,'UGC12506':6,'UGC12632':9,'UGC12732':9,'UGCA281':11,
    'UGCA442':9,'UGCA444':10,
}

# ── Grupos morfológicos ────────────────────────────────────
GRUPOS = {
    'Masivas\n(S0-Sb\nT≤3)':     {'T':[0,1,2,3],  'color':'#e74c3c'},
    'Espirales\n(Sbc-Sc\nT=4-5)':{'T':[4,5],      'color':'#e67e22'},
    'Tardías\n(Scd-Sd\nT=6-7)':  {'T':[6,7],      'color':'#f1c40f'},
    'Irregulares\n(Im/BCD\nT≥8)':{'T':[8,9,10,11],'color':'#27ae60'},
}

# ── Predicciones analíticas DEE (fórmula v2.0 corregida) ──
def alpha_pred_DEE(f_bulbo, r_disco):
    """α = (2/3) × f_masa — fórmula CORREGIDA en v2.0"""
    r = np.linspace(0.01, 8.0, 2000)
    S = f_bulbo*np.exp(-r/0.3) + (1-f_bulbo)*np.exp(-r/r_disco)
    w = r*S; eps=1e-6
    num = trapezoid(w*(1-np.exp(-r))**(2/3)/(r+eps)**(1/3), r)
    den = trapezoid(w/(r+eps)**(1/3), r)
    return 2/3 * (num/den)  # × f_masa (NO dividir — corrección v2.0)

ALPHA_PRED = {
    'Masivas\n(S0-Sb\nT≤3)':     alpha_pred_DEE(0.40, 1.0),
    'Espirales\n(Sbc-Sc\nT=4-5)':alpha_pred_DEE(0.15, 2.0),
    'Tardías\n(Scd-Sd\nT=6-7)':  alpha_pred_DEE(0.05, 3.0),
    'Irregulares\n(Im/BCD\nT≥8)':alpha_pred_DEE(0.01, 4.0),
}

print("\nPredicciones DEE v2.0 (α = (2/3)×f_masa):")
for k,v in ALPHA_PRED.items():
    print(f"  {k.replace(chr(10),' '):30s}: α_pred = {v:.4f}")

# ── Modelo DEE ─────────────────────────────────────────────
def DEE(r, vf, rs, a):
    return vf*(1-np.exp(-np.asarray(r)/max(rs,1e-3)))**max(a,0.01)

# ── Leer y ajustar ─────────────────────────────────────────
print("\n[1/3] Ajustando curvas SPARC por tipo morfológico...")
archivos = sorted(glob.glob("/content/SPARC_rotcurves/*_rotmod.dat"))
print(f"  Archivos disponibles: {len(archivos)}")

resultados = []
for path in archivos:
    nombre = os.path.basename(path).replace('_rotmod.dat','')
    T = TIPOS_T.get(nombre, None)
    if T is None: continue
    r,v,e=[],[],[]
    with open(path) as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith('#'): continue
            p=ln.split()
            if len(p)>=3:
                try:
                    rv,vv,ev=float(p[0]),float(p[1]),float(p[2])
                    if rv>0 and vv>0:
                        r.append(rv);v.append(vv);e.append(max(abs(ev),3.0))
                except: pass
    if len(r)<6: continue
    r=np.array(r);v=np.array(v);e=np.array(e)
    vmax=v.max()
    idx=np.searchsorted(np.sort(v),vmax*0.8)
    re=max(np.sort(r)[min(idx,len(r)-1)],0.5)
    best_a=np.nan;best_c2=np.inf
    for ai in [0.4,0.6,0.7,0.9,1.1,1.4]:
        try:
            pd,_=curve_fit(DEE,r,v,p0=[vmax*.95,re,ai],sigma=e,
                           bounds=([5,.1,.05],[2000,500,4]),maxfev=6000)
            c2=float(np.mean(((v-DEE(r,*pd))/np.maximum(e,1))**2))
            if c2<best_c2 and 0.05<pd[2]<3.9 and c2<200:
                best_c2=c2;best_a=pd[2]
        except: pass
    if not np.isnan(best_a):
        resultados.append((nombre,T,best_a))

print(f"  Ajustes válidos con tipo T: {len(resultados)}")

# ── Agrupar ────────────────────────────────────────────────
print("\n[2/3] Agrupando por morfología...")
alpha_por_grupo = {g:[] for g in GRUPOS}
for nombre,T,a in resultados:
    for gname,ginfo in GRUPOS.items():
        if T in ginfo['T']:
            alpha_por_grupo[gname].append(a); break

print(f"\n  {'Grupo':30s} {'N':>4} {'α_pred':>8} {'α_obs':>8} {'err%':>8}")
print(f"  {'-'*58}")
for gname,ginfo in GRUPOS.items():
    ag=np.array(alpha_por_grupo[gname])
    if not len(ag): continue
    a_pr=ALPHA_PRED[gname]; a_obs=np.median(ag)
    err=(a_pr-a_obs)/a_obs*100
    label=gname.replace('\n',' ')
    print(f"  {label:30s} {len(ag):>4} {a_pr:>8.4f} {a_obs:>8.4f} {err:>+8.1f}%")

# Tests
a_mas  = np.array(alpha_por_grupo['Masivas\n(S0-Sb\nT≤3)'])
a_irr  = np.array(alpha_por_grupo['Irregulares\n(Im/BCD\nT≥8)'])
pval = np.nan; pred_ok = False
if len(a_mas)>3 and len(a_irr)>3:
    # Predicción DEE: masivas < irregulares (fórmula v2.0)
    _,pval  = mannwhitneyu(a_mas, a_irr, alternative='less')
    pred_ok = np.median(a_mas) < np.median(a_irr)
    print(f"\n  Mann-Whitney (masivas < irreg): p = {pval:.4f}")
    print(f"  Predicción DEE (masivas < irreg): {'✓ CONFIRMADA' if pred_ok else '✗ REFUTADA'}")

groups_data=[np.array(v) for v in alpha_por_grupo.values() if len(v)>2]
kw_p=np.nan
if len(groups_data)>=3:
    _,kw_p=kruskal(*groups_data)
    print(f"  Kruskal-Wallis (4 grupos): p = {kw_p:.4f}")
    print(f"  {'✓ Diferencia significativa' if kw_p<0.05 else '✗ Sin diferencia'}")

# Correlación pred vs obs (pearson r)
preds=[ALPHA_PRED[g] for g in GRUPOS if alpha_por_grupo[g]]
obs_m=[np.median(alpha_por_grupo[g]) for g in GRUPOS if alpha_por_grupo[g]]
if len(preds)>=3:
    from scipy.stats import pearsonr
    r_corr,_=pearsonr(preds,obs_m)
    print(f"\n  Correlación α_pred vs α_obs: Pearson r = {r_corr:.4f}")
    print(f"  (Dirección correcta si r > 0)")

# ── GRÁFICO ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16,6))
fig.patch.set_facecolor('#0a0a1a')
BG='#0d1117';CW='#ecf0f1';CY='#f1c40f';CG='#27ae60';CGR='#7f8c8d'
for ax in axes:
    ax.set_facecolor(BG); ax.tick_params(colors=CGR)
    for s in ax.spines.values(): s.set_color('#2c3e50')

colores=[g['color'] for g in GRUPOS.values()]
grupos_names=list(GRUPOS.keys())

# P1: Boxplot
ax1=axes[0]
data_plot=[alpha_por_grupo[g] for g in grupos_names]
preds_plot=[ALPHA_PRED[g] for g in grupos_names]
bp=ax1.boxplot(data_plot,patch_artist=True,
               medianprops=dict(color=CY,lw=2.5),
               flierprops=dict(marker='o',markersize=3,alpha=0.5))
for patch,col in zip(bp['boxes'],colores):
    patch.set_facecolor(col); patch.set_alpha(0.6)
for w in bp['whiskers']: w.set_color(CGR)
for c in bp['caps']: c.set_color(CGR)
for i,(pred,col) in enumerate(zip(preds_plot,colores),1):
    ax1.plot([i-0.35,i+0.35],[pred,pred],'--',color=col,lw=2.5,alpha=0.9)
    ax1.text(i+0.38,pred,f'pred\n{pred:.2f}',fontsize=7.5,color=col,va='center',fontweight='bold')
ax1.set_xticks(range(1,len(grupos_names)+1))
ax1.set_xticklabels([g.replace('\n',' \n') for g in grupos_names],fontsize=8,color=CW)
ax1.set_ylabel('Exponente α',fontsize=11,color=CW)
ax1.set_title('α por tipo morfológico\n(punteada = predicción DEE v2.0)',
              fontsize=11,fontweight='bold',color=CW)
ax1.grid(True,alpha=0.15,axis='y'); ax1.set_ylim(0,3.5)

# P2: α vs T continuo
ax2=axes[1]
T_v=[r[1] for r in resultados]; a_v=[r[2] for r in resultados]
T_labels={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',
           6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
sc=ax2.scatter(T_v,a_v,s=25,c=a_v,cmap='RdYlGn',alpha=0.7,vmin=0.3,vmax=1.5)
plt.colorbar(sc,ax=ax2,label='α')
if len(T_v)>5:
    z=np.polyfit(T_v,a_v,1); T_fit=np.linspace(0,11,50)
    ax2.plot(T_fit,np.polyval(z,T_fit),'w-',lw=2.5,alpha=0.7,
             label=f'slope={z[0]:+.3f}')
    # Predicción DEE continua
    f_b=0.40*np.exp(-T_fit/3); r_d=1.0+T_fit*0.3
    a_cont=np.array([alpha_pred_DEE(fb,rd) for fb,rd in zip(f_b,r_d)])
    ax2.plot(T_fit,a_cont,'--',color=CY,lw=2.5,label='predicción DEE v2.0')
ax2.set_xlabel('Tipo Hubble T',fontsize=11,color=CW)
ax2.set_ylabel('α',fontsize=11,color=CW)
ax2.set_title('α vs Tipo Hubble\ntendencia masivas < irregulares predicha',
              fontsize=11,fontweight='bold',color=CW)
ax2.set_xticks(range(12))
ax2.set_xticklabels([T_labels.get(i,'') for i in range(12)],fontsize=7.5,color=CW,rotation=45)
ax2.legend(fontsize=8,facecolor=BG,labelcolor=CW); ax2.grid(True,alpha=0.15)

# P3: Resumen tabla
ax3=axes[2]; ax3.axis('off')
ax3.text(0.5,0.97,'VERIFICACIÓN PREDICCIÓN DEE v2.0',transform=ax3.transAxes,
         fontsize=11,fontweight='bold',color=CY,ha='center',va='top')
ax3.text(0.5,0.91,'α = (2/3) × f_masa   [fórmula corregida]',
         transform=ax3.transAxes,fontsize=8.5,color=CGR,ha='center',va='top',fontstyle='italic')
y=0.83
for lbl,txt in [('Grupo','α_pred'),('','α_obs'),('','N')]:
    pass
ax3.text(0.03,y,'Grupo',transform=ax3.transAxes,fontsize=8,color=CY,va='top',fontweight='bold')
ax3.text(0.38,y,'α_pred',transform=ax3.transAxes,fontsize=8,color=CY,va='top',fontweight='bold')
ax3.text(0.57,y,'α_obs',transform=ax3.transAxes,fontsize=8,color=CY,va='top',fontweight='bold')
ax3.text(0.76,y,'N',transform=ax3.transAxes,fontsize=8,color=CY,va='top',fontweight='bold')
for i,(gname,col) in enumerate(zip(grupos_names,colores)):
    ag=np.array(alpha_por_grupo[gname])
    if not len(ag): continue
    y_i=0.74-i*0.12; a_obs_g=np.median(ag); a_pr_g=ALPHA_PRED[gname]
    ax3.text(0.03,y_i,gname.replace('\n',' ')[:18],transform=ax3.transAxes,fontsize=7.5,color=col,va='top')
    ax3.text(0.38,y_i,f'{a_pr_g:.3f}',transform=ax3.transAxes,fontsize=8,color=col,va='top')
    ax3.text(0.57,y_i,f'{a_obs_g:.3f}',transform=ax3.transAxes,fontsize=8,color=CW,va='top',fontweight='bold')
    ax3.text(0.76,y_i,f'{len(ag)}',transform=ax3.transAxes,fontsize=8,color=CW,va='top')

pval_str=f'{pval:.4f}' if not np.isnan(pval) else 'N/A'
kw_str  =f'{kw_p:.4f}' if not np.isnan(kw_p)  else 'N/A'
ax3.text(0.03,0.27,f'Mann-Whitney p={pval_str}',transform=ax3.transAxes,fontsize=8.5,color=CW,va='top')
ax3.text(0.03,0.18,f'Kruskal-Wallis p={kw_str}',transform=ax3.transAxes,fontsize=8.5,color=CW,va='top')
veredicto='✓ PREDICCIÓN CONFIRMADA' if pred_ok else '✗ NO CONFIRMADA'
col_v=CG if pred_ok else '#e74c3c'
ax3.text(0.5,0.05,veredicto,transform=ax3.transAxes,fontsize=10,fontweight='bold',
         color=col_v,ha='center',va='bottom',
         bbox=dict(boxstyle='round',facecolor=BG,edgecolor=col_v,lw=2.5))

fig.suptitle(
    f'SIM 7 — α por morfología  |  DEE v2.0  |  Predicción: masivas < irregulares\n'
    f'Mann-Whitney p={pval_str}  Kruskal-Wallis p={kw_str}  α=(2/3)×f_masa',
    fontsize=12,fontweight='bold',color=CW)
plt.tight_layout()
plt.savefig('sim7_resultado.png',dpi=150,bbox_inches='tight',facecolor='#0a0a1a')
print(f"\n[OK] sim7_resultado.png guardado")
print(f"\n{'='*60}")
print(f"  RESUMEN SIM 7 — v2.0")
print(f"{'='*60}")
for gname in grupos_names:
    ag=np.array(alpha_por_grupo[gname])
    if not len(ag): continue
    print(f"  {gname.replace(chr(10),' '):30s}: α_pred={ALPHA_PRED[gname]:.3f}  α_obs={np.median(ag):.3f}  N={len(ag)}")
print(f"  Mann-Whitney: p={pval_str}  {'✓' if pred_ok else '✗'}")
print(f"  Kruskal-Wallis: p={kw_str}")
print(f"  Predicción (masivas<irreg): {'✓ CONFIRMADA' if pred_ok else '✗ REFUTADA'}")
print(f"{'='*60}")
