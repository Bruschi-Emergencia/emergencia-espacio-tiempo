"""
Creación de un defecto topológico (anillo saturado) y cálculo de su energía (masa).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ============================================================
# PARÁMETROS
# ============================================================
N = 2000
L_box = 10.0
R_anillo = 2.0
grosor = 0.2
penalizacion_saturacion = 1000.0
umbral_saturacion = 0.95

np.random.seed(42)
coords = np.random.rand(N, 3) * L_box
centro = np.array([L_box/2, L_box/2, L_box/2])

# Identificar nodos cerca del anillo (en el plano z=centro_z)
indices_anillo = []
for i, p in enumerate(coords):
    dx = p[0] - centro[0]
    dy = p[1] - centro[1]
    dz = p[2] - centro[2]
    r = np.sqrt(dx*dx + dy*dy)
    if abs(r - R_anillo) < grosor and abs(dz) < grosor:
        indices_anillo.append(i)

print(f"Nodos en el anillo: {len(indices_anillo)}")

# Matriz S inicial aleatoria (correlaciones bajas)
S = np.random.rand(N, N)
S = (S + S.T) / 2
np.fill_diagonal(S, 0)
S = S / np.max(S)

# Forzar correlaciones máximas dentro del anillo (defecto saturado)
for i in indices_anillo:
    for j in indices_anillo:
        if i < j:
            S[i,j] = S[j,i] = 1.0

# Definir funcional de energía F (sin dinámica, solo cálculo)
def F_total(Smat):
    triu = np.triu_indices_from(Smat, k=1)
    Sij = Smat[triu]
    term1 = -np.sum(Sij)
    term2 = np.sum(Sij**2)
    sobre = np.maximum(0, Sij - umbral_saturacion)
    term4 = penalizacion_saturacion * np.sum(sobre**4)
    grad = 0.0
    for a in range(N):
        row = Smat[a, :]
        row_mean = np.mean(row)
        grad += np.sum((row - row_mean)**2)
    return term1 + term2 + 0.1 * grad + term4

# Calcular energía del defecto
energia_con_defecto = F_total(S)

# Para comparar, crear una red sin defecto (misma distribución de nodos pero S aleatoria)
S_sin_defecto = np.random.rand(N, N)
S_sin_defecto = (S_sin_defecto + S_sin_defecto.T) / 2
np.fill_diagonal(S_sin_defecto, 0)
S_sin_defecto = S_sin_defecto / np.max(S_sin_defecto)
energia_sin_defecto = F_total(S_sin_defecto)

energia_defecto = energia_con_defecto - energia_sin_defecto
print(f"Energía del defecto (masa en unidades arbitrarias): {energia_defecto:.3f}")

# Para escalar a masa del electrón, necesitaríamos una calibración.
# Normalmente, se ajustaría para que energía_defecto * escala = m_e c^2.

# Visualización simple de la matriz de correlaciones
plt.imshow(S[:100, :100], cmap='hot', interpolation='nearest')
plt.colorbar(label='S_ij')
plt.title("Submatriz de correlaciones (primeros 100 nodos)")
plt.show()

print("Simulación defecto_topologico.py finalizada.")
