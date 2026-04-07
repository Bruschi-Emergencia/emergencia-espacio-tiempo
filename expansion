"""
Expansión local diferencial y diagrama de Hubble.
Parte de una configuración final de red_3D o genera una nueva.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

# ============================================================
# PARÁMETROS DE EXPANSIÓN
# ============================================================
N = 120
L_box = 10.0
dt = 0.02
T_exp = 2000
beta = 1.5          # sensibilidad de H a χ
H0_base = 0.08      # tasa base

# ============================================================
# INICIALIZACIÓN: usar una red ya relajada (simulada aquí rápidamente)
# En un caso real, cargar S y coords desde archivo.
# ============================================================
np.random.seed(42)
coords = np.random.rand(N, 3) * L_box
def dist_matrix(coords):
    return np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
D = dist_matrix(coords)
L_corr = 1.5
S = np.exp(-D / L_corr)
np.fill_diagonal(S, 0)
S = S / np.max(S)

# Función χ local (entropía)
def compute_chi_local(S, i):
    probs = S[i, :] / (np.sum(S[i, :]) + 1e-12)
    probs = np.clip(probs, 1e-12, 1)
    return -np.sum(probs * np.log(probs))

chi = np.array([compute_chi_local(S, i) for i in range(N)])
chi = chi / np.max(chi)

# ============================================================
# EVOLUCIÓN DE LA EXPANSIÓN
# ============================================================
history_t = [0.0]
history_L = [L_box]
history_chi_mean = [np.mean(chi)]
history_a = [1.0]
coords_exp = coords.copy()

for step in range(1, T_exp+1):
    t = step * dt
    # Tasa de expansión local
    H_local = H0_base * np.exp(-beta * chi)
    H_global = np.mean(H_local)
    # Factor de escala global
    L_new = history_L[-1] * (1 + H_global * dt)
    history_L.append(L_new)
    # Expansión de coordenadas (respecto al centro de masa)
    center = np.mean(coords_exp, axis=0)
    vec = coords_exp - center
    coords_exp = center + vec * (1 + H_local[:, None] * dt)
    # Actualizar χ (degradación suave por expansión)
    chi = chi * (1 - 0.01 * dt)
    chi = np.clip(chi, 0, 1)
    history_chi_mean.append(np.mean(chi))
    history_t.append(t)
    history_a.append(L_new / L_box)

history_t = np.array(history_t)
history_a = np.array(history_a)
z = 1.0 / history_a - 1.0
H_global_arr = np.gradient(history_L, history_t) / history_L

# ============================================================
# DIAGRAMA DE HUBBLE (última configuración)
# ============================================================
coords_final = coords_exp
center_final = np.mean(coords_final, axis=0)
r_final = np.linalg.norm(coords_final - center_final, axis=1)
velocities = (coords_final - coords) / (T_exp * dt)  # velocidad media aprox
v_r = np.sum(velocities * (coords_final - center_final) / (r_final[:, None] + 1e-12), axis=1)

def hubble_law(r, H0_eff):
    return H0_eff * r
H0_eff, _ = curve_fit(hubble_law, r_final, v_r)
print(f"H0 efectiva medida: {H0_eff[0]:.3f} (unidades adimensionales)")

# Gráficos
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(history_t, H_global_arr)
plt.xlabel("Tiempo")
plt.ylabel("H global")
plt.title("Tasa de expansión global")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(z, H_global_arr)
plt.xlabel("z")
plt.ylabel("H(z)")
plt.title("Evolución de H con el corrimiento al rojo")
plt.grid(True)

plt.subplot(1,3,3)
plt.scatter(r_final, v_r, alpha=0.5, label="Nodos")
r_line = np.linspace(0, max(r_final), 100)
plt.plot(r_line, H0_eff[0]*r_line, 'r-', label=f"H0_eff = {H0_eff[0]:.2f}")
plt.xlabel("Distancia al centro")
plt.ylabel("Velocidad radial")
plt.title("Diagrama de Hubble")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Parámetro de desaceleración q(z)
da_dt = np.gradient(history_a, history_t)
d2a_dt2 = np.gradient(da_dt, history_t)
q = - (history_a * d2a_dt2) / (da_dt**2 + 1e-12)
plt.figure()
plt.plot(z[1:], q[1:])
plt.xlabel("z")
plt.ylabel("q(z)")
plt.title("Parámetro de desaceleración")
plt.grid(True)
plt.show()

print("Simulación expansion.py finalizada.")
