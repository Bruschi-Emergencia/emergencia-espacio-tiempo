from __future__ import annotations

"""
DEE phase-2 follow-up: gravitational slip test
----------------------------------------------
This script is a practical scanner for a phenomenological slip parameter:

    eta(z) = Phi / Psi

or, equivalently, a deviation from unity in the lensing vs growth response.

Why this test
-------------
If B1 vs B2 remain degenerate in:
- fσ8(z)
- P(k,z)
- weak-lensing integrals
- temporal derivatives
- mixed scale-time observables

then the next natural quantity to probe is a direct difference between metric potentials.

Minimal phenomenological implementation
---------------------------------------
We define a redshift-dependent slip model:

    eta(z) = 1 + eta1 * z/(1+z)

and test whether a nonzero eta1 would induce a detectable difference
between B1-like and B2-like scenarios.

This script does NOT solve the full Einstein-Boltzmann hierarchy with slip.
Instead, it builds a diagnostic observable:

    S_slip(z) = (eta(z) - 1)

and compares:
- null case:   eta = 1
- slip case:   eta(z) != 1

This is a phase-2 screening tool to see what amplitude would be needed
before a full CLASS-level modified-potential implementation is worth it.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("slip_test_outputs")
OUT.mkdir(exist_ok=True)

# --------------------------------
# User-configurable scan
# --------------------------------
ETA1_SCAN = [0.00, 0.02, 0.05, 0.10, 0.20]
SIGMA_PER_BIN = 0.05   # conservative phenomenological uncertainty
Z = np.linspace(0.01, 2.0, 120)
Z_BINS = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])

# --------------------------------
# Slip model
# --------------------------------
def eta_of_z(z, eta1):
    z = np.asarray(z, dtype=float)
    return 1.0 + eta1 * z / (1.0 + z)

def binned_average(y, z, z_bins):
    out = []
    dz = z_bins[1] - z_bins[0]
    for z0 in z_bins:
        mask = (z >= z0 - dz/2) & (z <= z0 + dz/2)
        if np.sum(mask) < 2:
            out.append(np.nan)
        else:
            out.append(np.mean(y[mask]))
    return np.array(out, dtype=float)

def detectability(obs1, obs2, sigma):
    mask = np.isfinite(obs1) & np.isfinite(obs2)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.sqrt(np.sum(((obs1[mask] - obs2[mask]) / sigma) ** 2)))

# --------------------------------
# Main
# --------------------------------
print("=" * 72)
print("DEE phase-2 follow-up: gravitational slip screening test")
print("=" * 72)

null_eta = eta_of_z(Z, 0.0)
null_bins = binned_average(null_eta, Z, Z_BINS)

summary_rows = []

for eta1 in ETA1_SCAN:
    eta_curve = eta_of_z(Z, eta1)
    eta_bins = binned_average(eta_curve, Z, Z_BINS)

    sig_vs_null = detectability(null_bins, eta_bins, SIGMA_PER_BIN)
    max_dev = float(np.max(np.abs(eta_curve - 1.0)))

    summary_rows.append({
        "eta1": eta1,
        "max_abs_eta_minus_1": max_dev,
        "detectability_vs_null_sigma": sig_vs_null,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary_slip_scan.csv", index=False)

# --------------------------------
# Plots
# --------------------------------
BG = "#0d1117"; CW = "#ecf0f1"; CY = "#f1c40f"; CB = "#2980b9"; CG = "#27ae60"; CR = "#e74c3c"; CGR = "#7f8c8d"
cmap = plt.cm.plasma(np.linspace(0.15, 0.95, len(ETA1_SCAN)))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0a0a1a")

def sty(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=CGR, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#2c3e50")
    ax.grid(True, alpha=0.15)

# Panel 1: eta(z)
ax = axes[0]; sty(ax)
for i, eta1 in enumerate(ETA1_SCAN):
    ax.plot(Z, eta_of_z(Z, eta1), color=cmap[i], lw=2, label=f"eta1={eta1:.2f}")
ax.axhline(1.0, color="white", ls=":", lw=1, alpha=0.5)
ax.set_xlim(0, 2)
ax.set_xlabel("z", color=CW)
ax.set_ylabel("eta(z) = Phi/Psi", color=CW)
ax.set_title("Phenomenological gravitational slip", color=CY, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# Panel 2: eta-1
ax = axes[1]; sty(ax)
for i, eta1 in enumerate(ETA1_SCAN):
    ax.plot(Z, eta_of_z(Z, eta1) - 1.0, color=cmap[i], lw=2, label=f"eta1={eta1:.2f}")
ax.axhline(0.0, color="white", ls=":", lw=1, alpha=0.5)
ax.set_xlim(0, 2)
ax.set_xlabel("z", color=CW)
ax.set_ylabel("eta(z)-1", color=CW)
ax.set_title("Slip departure from GR", color=CG, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

# Panel 3: detectability
ax = axes[2]; sty(ax)
ax.semilogy(summary["eta1"], summary["detectability_vs_null_sigma"], "o-", color=CY, lw=2.5, ms=8)
ax.axhline(2, color=CR, lw=1.5, ls="--", label="2σ detection")
ax.axhline(5, color=CG, lw=1.5, ls="--", label="5σ strong")
for e, s in zip(summary["eta1"], summary["detectability_vs_null_sigma"]):
    ax.annotate(f"{s:.2f}σ", (e, s), textcoords="offset points", xytext=(4, 4), fontsize=8, color=CW)
ax.set_xlabel("eta1", color=CW)
ax.set_ylabel("Detectability [σ]", color=CW)
ax.set_title("Slip detectability scan", color=CY, fontweight="bold")
ax.legend(fontsize=8, facecolor=BG, labelcolor=CW)

fig.suptitle(
    "DEE phase-2 follow-up: gravitational slip screening",
    fontsize=13, fontweight="bold", color=CW
)
plt.tight_layout()
plt.savefig(OUT / "slip_scan_summary.png", dpi=150, bbox_inches="tight", facecolor="#0a0a1a")

print(summary.to_string(index=False))
print("\nSaved:")
print(OUT / "summary_slip_scan.csv")
print(OUT / "slip_scan_summary.png")
