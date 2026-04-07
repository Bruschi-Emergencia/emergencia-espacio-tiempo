# Emergencia del espacio-tiempo desde la organización del entrelazamiento

Este repositorio contiene el código de simulación y análisis para la propuesta "Dinámica de Estructura de Entrelazamiento y Emergencia del Espacio-Tiempo" (Versión 4.0). Los scripts implementan la red de correlaciones 3D, la ecuación de Langevin hiperbólica, la expansión diferencial, la formación de núcleos incompresibles, la bariogénesis y defectos topológicos.

## Requisitos

- Python 3.8 o superior
- Bibliotecas: numpy, scipy, matplotlib, emcee, corner, scikit-learn

Instalación rápida:
```bash
pip install -r requirements.txt

### Orden de ejecución recomendado

red_3D.py – Simulación básica de la red de correlaciones y cálculo de χ, dimensión de correlación, incrustación MDS.
Salida esperada: gráficos de evolución de χ, matriz de correlaciones final, dendrograma, dimensión ~3.

expansion.py – Expansión local diferencial y diagrama de Hubble.
Salida esperada: curva de Hubble, evolución de H(z) y q(z), estimación de H0.

mcmc_supernovas.py – Ajuste del modelo a datos de supernovas (Pantheon+).
Requiere datos reales: descargar Pantheon+SH0ES.dat y Pantheon+SH0ES_STAT+SYS.cov desde https://github.com/PantheonPlusSH0ES/DataRelease y colocarlos en datos/.
Salida esperada: corner plot con β y H0, curva de mejor ajuste.

nucleos_incompresibles.py – Simulación de dos núcleos saturados, colisión y ecos gravitacionales.
Salida esperada: gráficos de evolución de la distancia entre núcleos, asimetría bariónica, y señal sintética de ecos.

bariogenesis.py – Estudio de la asimetría bariónica inducida por el ruido ε y el anclaje en núcleos.
Salida esperada: asimetría final en función de ε, relación con el tamaño del núcleo.

defecto_topologico.py – Creación de un anillo saturado (defecto topológico) y cálculo de su energía (masa).
Salida esperada: valor numérico de la energía de deformación, que puede calibrarse a la masa del electrón.

graficos.py – Funciones auxiliares de visualización (importadas por los demás scripts).

#### Datos externos

Pantheon+ (supernovas): descargar de https://github.com/PantheonPlusSH0ES/DataRelease.

SPARC (curvas de rotación): opcional para comparación, descargar de http://astroweb.case.edu/SPARC/.

Planck CMB (parámetros cosmológicos): usar valores reportados (no se necesita archivo).

##### Resultados esperados

Dimensión de correlación final ≈ 2.8–3.2 (espacio emergente 3D).

Asimetría bariónica final ≈ 0.2–0.3 (escala a 10⁻⁹).

Constante de Hubble H0 ≈ 71.4 ± 1.2 km/s/Mpc.

Parámetro β ≈ 1.5 ± 0.2.

Ecos gravitacionales con periodicidad ≈ 1.2 unidades de tiempo.

Energía del defecto topológico (masa) del orden de la masa del electrón tras escalar.

###### Notas

Los scripts están configurados con parámetros por defecto que funcionan para pruebas rápidas (N pequeño, pocos pasos). Para resultados científicos aumentar N y T_sim.

Si no se tienen datos reales, los scripts de MCMC generan datos simulados para demostración.

Los gráficos se guardan en pantalla (pop-ups) o pueden modificarse para guardar en archivos.

Licencia
MIT. Se agradece cita al autor original (incluir referencia al documento)
