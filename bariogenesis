"""
Estudio de la asimetría bariónica en función de ε y del tamaño del núcleo.
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
T_sim = 2000
gamma = 0.1
eta = 0.5
umbral_saturacion = 0.95
penalizacion = 1000.0
J_coup = 0.3
T_B = 0.5

# Lista de epsilones a probar
epsilons = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
asym_final = []

for epsilon in epsilons:
    print(f"Simulando ε = {epsilon}")
    # Inicializar red con un único núcleo central (para simplificar)
    np.random.seed(42)
    coords = np.random.rand(N, 3) * L_box
    centro = np.array([L_box/2, L_box/2, L_box/2])
    # Forzar un núcleo denso alrededor del centro
    for i in range(N):
        if np.linalg.norm(coords[i] - centro) < 1.5:
            coords[i] = centro + 0.3 * (np.random.rand(3) - 0.5)
    def dist_matrix(coords):
        return np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
    D = dist_matrix(coords)
    L_corr = 1.5
    S = np.exp(-D / L_corr)
    np.fill_diagonal(S, 0)
    S = S / np.max(S)
    S_prev = S.copy()
    
    # Cargas bariónicas balanceadas
    B = np.random.choice([-1,1], size=N)
    while np.sum(B) != 0:
        B = np.random.choice([-1,1], size=N)
    
    # Función χ local
    def compute_chi_local(S, i):
        probs = S[i, :] / (np.sum(S[i, :]) + 1e-12)
        probs = np.clip(probs, 1e-12, 1)
        return -np.sum(probs * np.log(probs))
    
    # Gradiente con saturación (igual que en otros scripts)
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
    
    # Evolución
    for step in range(T_sim):
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
        
        # Evolución de cargas
        for i in range(N):
            campo = 0.0
            for j in range(N):
                if j != i:
                    campo += J_coup * S[i,j] * B[j]
            delta_E = 2 * B[i] * campo
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T_B):
                B[i] = -B[i]
    
    asym = np.abs(np.sum(B)) / N
    asym_final.append(asym)
    print(f"  Asimetría final: {asym:.4f}")

# Gráfico
plt.plot(epsilons, asym_final, 'o-')
plt.xlabel('ε (asimetría del ruido)')
plt.ylabel('Asimetría bariónica final')
plt.title('Dependencia de la bariogénesis con ε')
plt.grid(True)
plt.show()

print("Simulación bariogenesis.py finalizada.")
