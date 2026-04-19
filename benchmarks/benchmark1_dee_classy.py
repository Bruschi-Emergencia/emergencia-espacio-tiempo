from __future__ import annotations

import numpy as np

from dee_classy_utils import (
    lcdm_params,
    make_output_dir,
    plot_background,
    plot_growth,
    plot_mu_curves_benchmark1,
    plot_pk_ratio,
    run_class_model,
    save_summary_table,
    save_tables,
)

OUT = make_output_dir("benchmark1_outputs")

z_grid = np.linspace(0.0, 2.0, 41)
k_h_grid = np.logspace(-3, 0, 120)

lcdm = lcdm_params()

dee_params_benchmark1 = lcdm_params() | {
    "dee_model": "yes",
    "dee_use_Q_mode": "no",
    "dee_w0": -0.98,
    "dee_wa": 0.05,
    "dee_mu0": 0.08,
    "dee_kc": 0.07,
}

lcdm_res = run_class_model(lcdm, "LCDM", z_grid, k_h_grid)
dee_res = run_class_model(dee_params_benchmark1, "DEE benchmark 1", z_grid, k_h_grid)

save_tables(lcdm_res, OUT / "lcdm")
save_tables(dee_res, OUT / "dee_benchmark1")
save_summary_table([lcdm_res, dee_res], OUT / "summary_table.csv")

plot_background([lcdm_res, dee_res], OUT / "background_comparison.png")
plot_growth([lcdm_res, dee_res], OUT / "growth_comparison.png")
plot_pk_ratio(lcdm_res, dee_res, OUT / "pk_ratio_z0p5.png", z_for_ratio=0.5)
plot_pk_ratio(lcdm_res, dee_res, OUT / "pk_ratio_z1p0.png", z_for_ratio=1.0)
plot_mu_curves_benchmark1(
    mu0=float(dee_params_benchmark1["dee_mu0"]),
    kc=float(dee_params_benchmark1["dee_kc"]),
    out_file=OUT / "mu_curve.png",
)

print(f"Listo. Resultados guardados en: {OUT.resolve()}")
