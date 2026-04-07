"""
Ajuste del modelo cosmológico emergente a datos de supernovas (Pantheon+).
Si no se encuentran los datos reales, se generan datos simulados.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import emcee
import corner

# ============================================================
# INTENTAR CARGAR DATOS REALES
# ============================================================
try:
    # Asumimos que los archivos están en la carpeta 'datos/'
    data = np.loadtxt('datos/Pantheon+SH0ES.dat')
    z = data[:,0]
    mu_obs = data[:,1]
    mu_err = data[:,2]
    print("Cargados datos reales de Pantheon+")
except:
    print("No se encontraron datos reales. Generando datos simulados basados en ΛCDM.")
    np.random.seed(42)
    N_sne = 80
    z = np.random.uniform(0.01, 2.3, N_sne)
    # Simular ΛCDM con H0=70, Omega_m=0.315
    def dL_LCDM(z, H0=70, Om=0.315):
        c = 299792.458
        def integrand(zp):
            return 1.0 / np.sqrt(Om*(1+zp)**3 + (1-Om))
        integral = np.array([quad(integrand, 0, zi)[0] for zi in z])
        return c * (1+z) * integral / H0
    mu_true = 5 * np.log10(dL_LCDM(z)) + 25
    mu_obs = mu_true + np.random.normal(0, 0.15, N_sne)
    mu_err = np.full_like(z, 0.15)

# ============================================================
# MODELO EMERGENTE
# ============================================================
def chi_of_z(z, beta, H0):
    # Decaimiento exponencial de la organización con el redshift
    return np.exp(-beta * z)

def dL_emergente(z, beta, H0):
    c = 299792.458
    def integrand(zp):
        chi_val = chi_of_z(zp, beta, H0)
        H_eff = H0 * np.exp(-beta * chi_val)
        return 1.0 / H_eff
    integral = np.array([quad(integrand, 0, zi)[0] for zi in z])
    return c * (1+z) * integral

def log_likelihood(theta, z, mu_obs, mu_err):
    beta, H0 = theta
    if beta < 0 or H0 < 40 or H0 > 100:
        return -np.inf
    mu_theo = 5 * np.log10(dL_emergente(z, beta, H0)) + 25
    residuo = (mu_obs - mu_theo) / mu_err
    chi2 = np.sum(residuo**2)
    return -0.5 * chi2

# ============================================================
# MCMC
# ============================================================
ndim, nwalkers, nsteps = 2, 32, 1000
initial = np.array([1.5, 70.0])
pos = [initial + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(z, mu_obs, mu_err))
print("Ejecutando MCMC...")
sampler.run_mcmc(pos, nsteps, progress=True)

# Extraer muestras (descartar burn-in)
flat_samples = sampler.get_chain(discard=300, thin=10, flat=True)

# Resultados
beta_mcmc = np.percentile(flat_samples[:,0], [16,50,84])
H0_mcmc = np.percentile(flat_samples[:,1], [16,50,84])

print(f"beta = {beta_mcmc[1]:.3f} +{beta_mcmc[2]-beta_mcmc[1]:.3f} -{beta_mcmc[1]-beta_mcmc[0]:.3f}")
print(f"H0   = {H0_mcmc[1]:.1f} +{H0_mcmc[2]-H0_mcmc[1]:.1f} -{H0_mcmc[1]-H0_mcmc[0]:.1f} km/s/Mpc")

# Corner plot
fig = corner.corner(flat_samples, labels=[r'$\beta$', r'$H_0$ (km/s/Mpc)'],
                    truths=[beta_mcmc[1], H0_mcmc[1]])
plt.show()

# Mejor ajuste sobre los datos
beta_best = beta_mcmc[1]
H0_best = H0_mcmc[1]
z_fine = np.linspace(0.01, 2.5, 100)
mu_fit = 5 * np.log10(dL_emergente(z_fine, beta_best, H0_best)) + 25

plt.figure()
plt.errorbar(z, mu_obs, yerr=mu_err, fmt='.', capsize=2, label='Datos')
plt.plot(z_fine, mu_fit, 'r-', label='Mejor ajuste (modelo emergente)')
plt.xlabel('z')
plt.ylabel('μ')
plt.title('Diagrama de Hubble: modelo emergente vs datos')
plt.legend()
plt.grid(True)
plt.show()

print("Simulación mcmc_supernovas.py finalizada.")
