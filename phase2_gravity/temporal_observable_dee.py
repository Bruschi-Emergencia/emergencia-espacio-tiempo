import numpy as np
import matplotlib.pyplot as plt

def dlog_dz(y, z):
    y = np.array(y)
    z = np.array(z)
    dydz = np.zeros_like(y)
    for i in range(1, len(z)-1):
        dydz[i] = (np.log(y[i+1]) - np.log(y[i-1])) / (z[i+1] - z[i-1])
    dydz[0]  = (np.log(y[1]) - np.log(y[0])) / (z[1] - z[0])
    dydz[-1] = (np.log(y[-1]) - np.log(y[-2])) / (z[-1] - z[-2])
    return dydz

def build_temporal_observables(z, sigma8, fs8):
    G_t = dlog_dz(sigma8, z)
    F_t = dlog_dz(fs8, z)
    return G_t, F_t

def detectability(obs1, obs2, sigma=0.02):
    chi2 = np.sum(((obs1 - obs2)/sigma)**2)
    return np.sqrt(chi2)

def run_analysis(models, z):
    results = {}
    for name, res in models.items():
        G_t, F_t = build_temporal_observables(z, res["sigma8_z"], res["fs8_z"])
        results[name] = {"G_t": G_t, "F_t": F_t}

    b1 = results["DEE_B1"]

    print("\n=== Detectabilidad ===\n")
    for name, r in results.items():
        if name == "DEE_B1":
            continue
        print(name)
        print(" G_t:", detectability(b1["G_t"], r["G_t"]))
        print(" F_t:", detectability(b1["F_t"], r["F_t"]))
        print()

    return results

def plot(z, results, key):
    plt.figure()
    for name, r in results.items():
        plt.plot(z, r[key], label=name)
    plt.legend()
    plt.title(key)
    plt.grid()
    plt.show()
