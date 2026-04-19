from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
try:
    from classy import Class
    HAS_CLASSY = True
except ImportError:
    HAS_CLASSY = False

OUT = Path("Ft_scan_outputs")
OUT.mkdir(exist_ok=True)

BASE = {
    "output": "mPk,tCl,lCl",
    "lensing": "yes",
    "modes": "s",
    "P_k_max_1/Mpc": 5.0,
    "z_max_pk": 4.0,
    "h": 0.67,
    "omega_b": 0.0224,
    "omega_cdm": 0.12,
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "tau_reio": 0.054,
}

DEE_BG = {
    **BASE,
    "w0_fld": -0.98,
    "wa_fld": 0.05,
    "Omega_Lambda": 0,
}

def mu_b1(k, a):
    return 1.0 - 0.08 * k**2 / (k**2 + 0.07**2)

def make_mu_b2(mu1):
    def mu_b2(k, a):
        mu = 0.06 + mu1 * (1.0 - a)
        kc = 0.07 * (1.0 + 0.30 * (1.0 - a))
        return 1.0 - mu * k**2 / (k**2 + kc**2)
    return mu_b2

def get_class_background(params):
    if HAS_CLASSY:
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        bg = cosmo.get_background()
        z_bg = bg["z"]
        H_bg = bg["H [1/Mpc]"] * 299792.458
        h = cosmo.h()
        Om0 = cosmo.Omega_m()
        sig8_0 = cosmo.sigma(8.0 / h, 0.0)
        H_itp = interp1d(z_bg, H_bg, bounds_error=False, fill_value="extrapolate")
        cosmo.struct_cleanup(); cosmo.empty()
    else:
        # Internal solver fallback
        h_val=0.67; H0=h_val*100; Om0=0.3096; Or=9.1e-5; Oq=1-Om0-Or
        w0=params.get("w0_fld",-1.0); wa=params.get("wa_fld",0.0)
        lna=np.linspace(-12,0,3000)
        w_arr=w0+wa*(1-np.exp(lna))
        ex=-3*np.cumsum(1+w_arr)*(lna[1]-lna[0]); ex-=ex[-1]
        from scipy.interpolate import interp1d as itp1d
        rq=itp1d(lna,Oq*np.exp(ex),bounds_error=False,fill_value="extrapolate")
        def E2i(a): return Om0*a**-3+Or*a**-4+rq(np.log(a))
        c_kms=299792.458
        z_arr=np.linspace(0,5,3000); H_arr=np.array([H0*np.sqrt(E2i(1/(1+z))) for z in z_arr])
        H_itp=itp1d(z_arr,H_arr,bounds_error=False,fill_value="extrapolate")
        h=h_val; sig8_0=0.8111
    return H_itp, float(Om0), float(sig8_0), float(h)

def solve_growth_with_Geff(H_itp, Om0, mu_func, k_h=0.15, s8_0=0.8111):
    H0 = float(H_itp(0.0))
    def E2(a):
        z = 1.0 / a - 1.0
        return (H_itp(z) / H0) ** 2
    def dlogH(a):
        da = a * 1e-5
        ap = min(a + da, 0.999999)
        am = max(a - da, 1e-6)
        zp = 1.0 / ap - 1.0
        zm = 1.0 / am - 1.0
        return a / (2.0 * E2(a)) * (((H_itp(zp) / H0) ** 2) - ((H_itp(zm) / H0) ** 2)) / (ap - am)
    def Om_a(a):
        z = 1.0 / a - 1.0
        return Om0 * (1.0 + z) ** 3 / E2(a)
    def rhs(ln_a, y):
        a = np.exp(ln_a)
        d, dp = y
        return [dp, 1.5 * Om_a(a) * mu_func(k_h, a) * d - (2.0 + dlogH(a)) * dp]
    t_eval = np.linspace(np.log(1e-4), 0.0, 1400)
    sol = solve_ivp(rhs, (t_eval[0], 0.0), [1e-4, 1e-4], t_eval=t_eval,
                    method="DOP853", rtol=1e-10, atol=1e-13)
    a_s = np.exp(sol.t)
    d_s = sol.y[0]
    dp_s = sol.y[1]
    d_i = interp1d(a_s, d_s, bounds_error=False, fill_value="extrapolate")
    f_i = interp1d(a_s, dp_s / np.maximum(d_s, 1e-30), bounds_error=False, fill_value="extrapolate")
    d1 = float(d_i(1.0))
    D = lambda z: float(d_i(1.0 / (1.0 + z)) / d1)
    f = lambda z: float(f_i(1.0 / (1.0 + z)))
    sigma8 = lambda z: s8_0 * D(z)
    fs8 = lambda z: f(z) * sigma8(z)
    return D, f, sigma8, fs8

def dlog_dz(y, z):
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(y)
    for i in range(1, len(z) - 1):
        out[i] = (np.log(y[i + 1]) - np.log(y[i - 1])) / (z[i + 1] - z[i - 1])
    out[0] = (np.log(y[1]) - np.log(y[0])) / (z[1] - z[0])
    out[-1] = (np.log(y[-1]) - np.log(y[-2])) / (z[-1] - z[-2])
    return out

def binned_Ft(fs8_values, z_bins):
    fs8_values = np.asarray(fs8_values, dtype=float)
    z_bins = np.asarray(z_bins, dtype=float)
    return (np.log(fs8_values[1:]) - np.log(fs8_values[:-1])) / (z_bins[1:] - z_bins[:-1])

def detectability(obs1, obs2, sigma):
    return float(np.sqrt(np.sum(((obs1 - obs2) / sigma) ** 2)))

print("=" * 72)
print("DEE phase-2 closure test: F_t(z) with CLASS-real H(z)")
print("=" * 72)

Z_FINE = np.linspace(0.01, 2.0, 160)
Z_BINS = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
FT_SIGMA_PER_BIN = 0.05
K_PROBE = 0.15
MU1_SCAN = [0.00, 0.04, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]

H_itp, Om0, sig8_0, h = get_class_background(DEE_BG)

models = {}
for label, mu_func in [("LCDM", lambda k, a: 1.0), ("B1", mu_b1)]:
    D, f, sigma8, fs8 = solve_growth_with_Geff(H_itp, Om0, mu_func, k_h=K_PROBE, s8_0=sig8_0)
    sig_arr = np.array([sigma8(z) for z in Z_FINE])
    fs8_arr = np.array([fs8(z) for z in Z_FINE])
    Ft_arr = dlog_dz(fs8_arr, Z_FINE)
    fs8_bins = np.array([fs8(z) for z in Z_BINS])
    Ft_bins = binned_Ft(fs8_bins, Z_BINS)
    models[label] = {"sigma8_z": sig_arr, "fs8_z": fs8_arr, "Ft_z": Ft_arr, "fs8_bins": fs8_bins, "Ft_bins": Ft_bins}
    pd.DataFrame({"z": Z_FINE, "sigma8_z": sig_arr, "f_sigma8_z": fs8_arr, "F_t_z": Ft_arr}).to_csv(OUT / f"{label}_temporal_table.csv", index=False)

summary_rows = []
det_rows = []
b1_ft = models["B1"]["Ft_bins"]

for mu1 in MU1_SCAN:
    mu_func = make_mu_b2(mu1)
    D, f, sigma8, fs8 = solve_growth_with_Geff(H_itp, Om0, mu_func, k_h=K_PROBE, s8_0=sig8_0)
    sig_arr = np.array([sigma8(z) for z in Z_FINE])
    fs8_arr = np.array([fs8(z) for z in Z_FINE])
    Ft_arr = dlog_dz(fs8_arr, Z_FINE)
    fs8_bins = np.array([fs8(z) for z in Z_BINS])
    Ft_bins = binned_Ft(fs8_bins, Z_BINS)
    det_ft = detectability(b1_ft, Ft_bins, FT_SIGMA_PER_BIN)
    max_delta = float(np.max(np.abs(Ft_bins - b1_ft)))
    summary_rows.append({"mu1": mu1, "sigma8_0": sig_arr[0], "fs8_z0p5": float(np.interp(0.5, Z_FINE, fs8_arr)),
                         "Ft_detectability_vs_B1_sigma": det_ft, "Ft_max_bin_difference": max_delta})
    det_rows.append({"modelo": f"B2_mu1_{mu1:.2f}", "F_t_vs_B1_sigma": det_ft, "detectable_2sigma": det_ft > 2.0})
    models[f"B2_{mu1:.2f}"] = {"sigma8_z": sig_arr, "fs8_z": fs8_arr, "Ft_z": Ft_arr, "fs8_bins": fs8_bins, "Ft_bins": Ft_bins}

summary = pd.DataFrame(summary_rows)
det_table = pd.DataFrame(det_rows)
summary.to_csv(OUT / "summary_Ft_scan.csv", index=False)
det_table.to_csv(OUT / "detectability_Ft_scan.csv", index=False)

BG = "#0d1117"; CW = "#ecf0f1"; CY = "#f1c40f"; CB = "#2980b9"; CG = "#27ae60"; CR = "#e74c3c"; CGR = "#7f8c8d"
cmap = plt.cm.plasma(np.linspace(0.15, 0.95, len(MU1_SCAN)))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#0a0a1a")
def sty(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=CGR, labelsize=9)
    for s in ax.spines.values(): s.set_color("#2c3e50")
    ax.grid(True, alpha=0.15)

ax = axes[0,0]; sty(ax)
ax.plot(Z_FINE, models["LCDM"]["fs8_z"], color=CB, lw=2.5, label="LCDM")
ax.plot(Z_FINE, models["B1"]["fs8_z"], color=CY, lw=2.5, ls="--", label="B1")
for i, mu1 in enumerate(MU1_SCAN):
    ax.plot(Z_FINE, models[f"B2_{mu1:.2f}"]["fs8_z"], color=cmap[i], lw=1.8, alpha=0.9, label=f"B2 μ1={mu1:.2f}")
ax.set_xlim(0,2); ax.set_xlabel("z", color=CW); ax.set_ylabel("fσ8(z)", color=CW); ax.set_title("fσ8(z) family", color=CY, fontweight="bold")
ax.legend(fontsize=7, ncol=2, facecolor=BG, labelcolor=CW)

ax = axes[0,1]; sty(ax)
ax.plot(Z_FINE, models["LCDM"]["Ft_z"], color=CB, lw=2.5, label="LCDM")
ax.plot(Z_FINE, models["B1"]["Ft_z"], color=CY, lw=2.5, ls="--", label="B1")
for i, mu1 in enumerate(MU1_SCAN):
    ax.plot(Z_FINE, models[f"B2_{mu1:.2f}"]["Ft_z"], color=cmap[i], lw=1.8, alpha=0.9, label=f"B2 μ1={mu1:.2f}")
ax.set_xlim(0.1,2); ax.set_xlabel("z", color=CW); ax.set_ylabel("F_t = d ln fσ8 / dz", color=CW); ax.set_title("Temporal observable F_t(z)", color=CG, fontweight="bold")
ax.legend(fontsize=7, ncol=2, facecolor=BG, labelcolor=CW)

ax = axes[0,2]; sty(ax)
for i, mu1 in enumerate(MU1_SCAN):
    delta = models[f"B2_{mu1:.2f}"]["Ft_z"] - models["B1"]["Ft_z"]
    ax.plot(Z_FINE, delta, color=cmap[i], lw=2.0, label=f"μ1={mu1:.2f}")
ax.axhline(0, color="white", ls=":", lw=1, alpha=0.5)
ax.set_xlim(0.1,2); ax.set_xlabel("z", color=CW); ax.set_ylabel("ΔF_t = F_t(B2)-F_t(B1)", color=CW); ax.set_title("Instantaneous temporal separation", color=CR, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

ax = axes[1,0]; sty(ax)
ax.semilogy(summary["mu1"], summary["Ft_detectability_vs_B1_sigma"], "o-", color=CY, lw=2.5, ms=8)
ax.axhline(2, color=CR, lw=1.5, ls="--", label="2σ detection")
ax.axhline(5, color=CG, lw=1.5, ls="--", label="5σ strong")
for m, s in zip(summary["mu1"], summary["Ft_detectability_vs_B1_sigma"]):
    ax.annotate(f"{s:.2f}σ", (m, s), textcoords="offset points", xytext=(4, 4), fontsize=8, color=CW)
ax.set_xlabel("μ1", color=CW); ax.set_ylabel("Detectability [σ]", color=CW); ax.set_title("F_t detectability scan", color=CY, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

ax = axes[1,1]; sty(ax)
for i, mu1 in enumerate(MU1_SCAN):
    ratio = models[f"B2_{mu1:.2f}"]["fs8_z"] / models["B1"]["fs8_z"]
    ax.plot(Z_FINE, ratio, color=cmap[i], lw=2.0, label=f"μ1={mu1:.2f}")
ax.axhline(1, color="white", ls=":", lw=1, alpha=0.5)
ax.set_xlim(0.1,2); ax.set_xlabel("z", color=CW); ax.set_ylabel("fσ8(B2)/fσ8(B1)", color=CW); ax.set_title("Integrated growth ratio", color=CB, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

ax = axes[1,2]; ax.axis("off"); ax.set_facecolor("#0a0e1a")
headers = ["μ1", "fσ8(0.5)", "F_t [σ]", "Detectable?"]
rows = [[f"{r['mu1']:.2f}", f"{r['fs8_z0p5']:.4f}", f"{r['Ft_detectability_vs_B1_sigma']:.2f}σ", "✓" if r["Ft_detectability_vs_B1_sigma"] > 2 else "✗"] for _, r in summary.iterrows()]
y = 0.95
for j, htxt in enumerate(headers):
    ax.text(0.03 + 0.22*j, y, htxt, transform=ax.transAxes, fontsize=10, color=CY, fontweight="bold", va="top")
y -= 0.08
ax.plot([0.02, 0.98], [y, y], transform=ax.transAxes, color=CGR, lw=0.5)
y -= 0.03
for row in rows:
    for j, cell in enumerate(row):
        col = CW
        if j == 3:
            col = CG if cell == "✓" else CR
        ax.text(0.03 + 0.22*j, y, cell, transform=ax.transAxes, fontsize=10, color=col, va="top")
    y -= 0.11
ax.text(0.5, 0.04, "Recommended rigorous phase-2 closure test:\nuse binned F_t with conservative finite-difference errors.\nThis targets temporal response rather than integrated growth.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=CW,
        bbox=dict(facecolor="#0d1117", alpha=0.9, edgecolor=CY))

fig.suptitle("DEE phase-2 gravity closure test\nCLASS-real H(z) + post-processed G_eff + binned temporal observable F_t",
             fontsize=13, fontweight="bold", color=CW)
plt.tight_layout()
plt.savefig(OUT / "Ft_scan_summary.png", dpi=150, bbox_inches="tight", facecolor="#0a0a1a")

print(OUT / "summary_Ft_scan.csv")
print(OUT / "detectability_Ft_scan.csv")
print(OUT / "Ft_scan_summary.png")
