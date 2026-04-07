"""
Simulación de dos núcleos incompresibles, su interacción y ecos gravitacionales.
Incluye potencial de saturación y evolución hiperbólica.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ============================================================
# PARÁMETROS
# ============================================================
N = 200
L_box = 10.0
dt = 0.01
T_sim = 3000
epsilon = 0.12
gamma = 0.1
eta = 0.5
umbral_saturacion = 0.95
penalizacion = 1000.0

# Inicializar dos clusters (núcleos) separados
np.random.seed(42)
coords = np.random.rand(N, 3) * L_box
centro1 = np.array([2.0, L_box/2, L_box/2])
centro2 = np.array([8.0, L_box/2, L_box/2])
for i in range(N//2):
    coords[i] = centro1 + 0.5 * (np.random.rand(3) - 0.5)
for i in range(N//2, N):
    coords[i] = centro2 + 0.5 * (np.random.rand(3) - 0.5)

def dist_matrix(coords):
    return np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)

D = dist_matrix(coords)
L_corr = 1.5
S = np.exp(-D / L_corr)
np.fill_diagonal(S, 0)
S = S / np.max(S)
S_prev = S.copy()

# Cargas bariónicas (balanceadas)
B = np.random.choice([-1,1], size=N)
while np.sum(B) != 0:
    B = np.random.choice([-1,1], size=N)

# ============================================================
# FUNCIONES
# ============================================================
def compute_chi_local(S, i):
    probs = S[i, :] / (np.sum(S[i, :]) + 1e-12)
    probs = np.clip(probs, 1e-12, 1)
    return -np.sum(probs * np.log(probs))

def gradiente_F_local(S, i, j, eps=1e-6):
    def F_total(Smat):
        triu = np.triu_indices_from(Smat, k=1)
        Sij = Smat[triu]
        term1 = -np.sum(Sij)
        term2 = np.sum(Sij**2)
        sobre = np.maximum(0, Sij - umbral_saturacion)
        term4 = penalizacion * np.sum(sobre**4)
        grad = 0.0
        for a in range(N):
            row = Smat[a, :]
            row_mean = np.mean(row)
            grad += np.sum((row - row_mean)**2)
        return term1 + term2 + 0.1 * grad + term4
    S_plus = S.copy()
    S_plus[i, j] += eps; S_plus[j, i] = S_plus[i, j]
    S_minus = S.copy()
    S_minus[i, j] -= eps; S_minus[j, i] = S_minus[i, j]
    return (F_total(S_plus) - F_total(S_minus)) / (2*eps)

# Historiales
hist_chi = []
hist_asym = []
hist_dist = []

for step in range(T_sim):
    # Gradiente y ruido
    grad = np.zeros_like(S)
    for i in range(N):
        for j in range(i+1, N):
            g = gradiente_F_local(S, i, j)
            grad[i,j] = g; grad[j,i] = g
    ruido = gamma * epsilon * np.random.randn(N, N) * np.sqrt(dt)
    ruido = (ruido + ruido.T) / 2
    np.fill_diagonal(ruido, 0)
    fuerza = -grad + ruido
    S_new = 2*S - S_prev + fuerza * dt**2 + eta * (S - S_prev) * dt
    S_new = (S_new + S_new.T) / 2
    S_new = np.clip(S_new, 0, 1)
    np.fill_diagonal(S_new, 0)
    S_prev = S.copy()
    S = S_new.copy()
    
    # χ medio
    chi_vals = [compute_chi_local(S, i) for i in range(N)]
    chi_mean = np.mean(chi_vals)
    hist_chi.append(chi_mean)
    
    # Evolución de cargas bariónicas (Metropolis)
    T_B = 0.5; J_coup = 0.3
    for i in range(N):
        campo = 0.0
        for j in range(N):
            if j != i:
                campo += J_coup * S[i,j] * B[j]
        delta_E = 2 * B[i] * campo
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T_B):
            B[i] = -B[i]
    asym = np.abs(np.sum(B)) / N
    hist_asym.append(asym)
    
    # Distancia entre centros de los dos clusters
    idx1 = np.arange(N//2); idx2 = np.arange(N//2, N)
    c1 = np.mean(coords[idx1], axis=0)
    c2 = np.mean(coords[idx2], axis=0)
    dist = np.linalg.norm(c1 - c2)
    hist_dist.append(dist)
    
    if step % 1000 == 0:
        print(f"Paso {step}: χ={chi_mean:.3f}, asym={asym:.4f}, dist={dist:.2f}")

# ============================================================
# SIMULACIÓN DE ECOS GRAVITACIONALES
# ============================================================
t = np.linspace(0, 10, 5000)
# Señal estándar (Kerr)
h_kerr = np.exp(-t/2) * np.sin(2*np.pi*2*t) * (t<5)
# Señal con ecos (núcleo incompresible)
h_eco = h_kerr.copy()
for eco in range(1, 4):
    t_shift = t - eco*1.2
    eco_signal = np.exp(-t_shift/2) * np.sin(2*np.pi*2*t_shift) * (t_shift>0) * 0.3**eco
    h_eco += eco_signal

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(hist_chi)
plt.title('⟨χ⟩')
plt.subplot(1,3,2)
plt.plot(hist_asym)
plt.title('Asimetría bariónica')
plt.subplot(1,3,3)
plt.plot(hist_dist)
plt.title('Distancia entre núcleos')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, h_kerr, 'b-', label='Kerr (sin estructura)')
plt.plot(t, h_eco, 'r--', label='Núcleo incompresible (con ecos)')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Ecos gravitacionales post-fusión')
plt.legend()
plt.grid(True)
plt.show()

print(f"Asimetría final: {hist_asym[-1]:.5f}")
print("Simulación nucleos_incompresibles.py finalizada.")
