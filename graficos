"""
Funciones auxiliares de visualización reutilizables.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_historial(t, chi, asym, dist):
    """Muestra evolución de χ, asimetría y distancia entre núcleos."""
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(t, chi)
    plt.title('⟨χ⟩')
    plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(t, asym)
    plt.title('Asimetría bariónica')
    plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(t, dist)
    plt.title('Distancia núcleos')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ecos(t, h_kerr, h_eco):
    """Compara señal de ondas gravitacionales con y sin ecos."""
    plt.figure()
    plt.plot(t, h_kerr, 'b-', label='Kerr')
    plt.plot(t, h_eco, 'r--', label='Núcleo incompresible')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title('Ecos gravitacionales')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_matriz_correlaciones(S, titulo="Matriz de correlaciones"):
    plt.figure(figsize=(8,7))
    plt.imshow(S, cmap='hot', interpolation='nearest')
    plt.colorbar(label='S_ij')
    plt.title(titulo)
    plt.show()

def plot_diagrama_hubble(r, v_r, H0_eff):
    plt.scatter(r, v_r, alpha=0.5)
    r_line = np.linspace(0, max(r), 100)
    plt.plot(r_line, H0_eff * r_line, 'r-', label=f'H0_eff = {H0_eff:.2f}')
    plt.xlabel('Distancia')
    plt.ylabel('Velocidad radial')
    plt.title('Diagrama de Hubble')
    plt.legend()
    plt.grid(True)
    plt.show()
