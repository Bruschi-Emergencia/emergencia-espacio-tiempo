
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

OUT = Path("kz_full_outputs")
OUT.mkdir(exist_ok=True)

LCDM_FILE = "LCDM_pk_table.csv"
B1_FILE   = "B1_pk_table.csv"
B2_FILE   = "B2_pk_table.csv"

K_PAIRS = [
    (0.03, 0.15),
    (0.05, 0.20),
    (0.05, 0.30),
    (0.07, 0.20),
]

SIGMA = 0.03

def dlog_dz(y, z):
    y = np.array(y)
    z = np.array(z)
    out = np.zeros_like(y)
    for i in range(1, len(z)-1):
        out[i] = (np.log(y[i+1]) - np.log(y[i-1])) / (z[i+1]-z[i-1])
    out[0] = (np.log(y[1])-np.log(y[0]))/(z[1]-z[0])
    out[-1] = (np.log(y[-1])-np.log(y[-2]))/(z[-1]-z[-2])
    return out

def detectability(a,b):
    return np.sqrt(np.sum(((a-b)/SIGMA)**2))

def load_pk(file):
    df = pd.read_csv(file)
    k = df["k_h_Mpc"].values
    z_cols = [c for c in df.columns if c.startswith("z_")]
    z_vals = [float(re.findall(r"[0-9.]+", c)[0]) for c in z_cols]
    order = np.argsort(z_vals)
    z = np.array(z_vals)[order]
    pk = df[z_cols].values[:,order].T
    return z,k,pk

z,k,pk_LCDM = load_pk(LCDM_FILE)
_,_,pk_B1 = load_pk(B1_FILE)
_,_,pk_B2 = load_pk(B2_FILE)

summary = []

for k1,k2 in K_PAIRS:
    i1 = np.argmin(abs(k-k1))
    i2 = np.argmin(abs(k-k2))

    R_LCDM = pk_LCDM[:,i1]/pk_LCDM[:,i2]
    R_B1   = pk_B1[:,i1]/pk_B1[:,i2]
    R_B2   = pk_B2[:,i1]/pk_B2[:,i2]

    Okz_LCDM = dlog_dz(R_LCDM,z)
    Okz_B1   = dlog_dz(R_B1,z)
    Okz_B2   = dlog_dz(R_B2,z)

    sig = detectability(Okz_B1,Okz_B2)

    summary.append({
        "k1":k[i1],
        "k2":k[i2],
        "B2_vs_B1_sigma":sig
    })

    plt.figure()
    plt.plot(z,Okz_LCDM,label="LCDM")
    plt.plot(z,Okz_B1,label="B1")
    plt.plot(z,Okz_B2,label="B2")
    plt.legend()
    plt.title(f"O_kz k1={k[i1]:.3f}, k2={k[i2]:.3f}")
    plt.savefig(OUT/f"Okz_{k[i1]:.3f}_{k[i2]:.3f}.png")
    plt.close()

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT/"summary.csv",index=False)

print(summary_df)
