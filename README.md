# AI Digital Health Platforms for Preventive Care, Healthcare Efficiency and Organizational Innovation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22558363.svg)](https://doi.org/10.5281/zenodo.22558363)

Reproducible analysis accompanying the paper presented at **AI-SMART 2026 – AI for a Smarter Tomorrow**, 2nd International Scientific Multidisciplinary Conference, Belgrade, 23–25 September 2026.

Authors: Vedran Uroš (Veleučilište „Marko Marulić" u Kninu), Martin Birač (StoreDNA), Damir Mihanović (Sveučilište Sjever).

## What the analysis does

Country-level analysis of the EU-27 (2020–2024) on whether digital health maturity is associated with healthcare efficiency and access, plus a probabilistic budget-impact scenario for an AI-enabled preventive platform in Croatia.

1. **Descriptive statistics** for the most recent year, with Croatia's rank (`results/table1_descriptives.csv`).
2. **Spearman correlations** between digital indicators and outcomes (preventable and treatable mortality, unmet medical needs) with 2,000-resample bootstrap CIs, Holm adjustment and partial correlations controlling for GDP per capita (`results/table2_spearman.csv`, `figures/FigureS1_correlation_heatmap.*`).
3. **PCA and Ward hierarchical clustering** of six standardized indicators; silhouette for k = 2–6 and bootstrap Jaccard stability (`results/table3_clusters.csv`, `results/pca_cluster_meta.json`, `figures/Figure2_pca_clusters.*`).
4. **Fixed-effects panel regression** over the three Eurostat e-health waves (2020, 2022, 2024; N = 77) with cluster-robust standard errors (`results/table4_panel.csv`).
5. **Monte Carlo budget-impact scenario for Croatia** (10,000 iterations) with one-way tornado sensitivity (`results/mc_results.json`, `figures/Figure3_tornado.*`).
6. **Conceptual framework figure** (`figures/Figure1_framework.*`).

## Data

All data were retrieved on 6 September 2026 and are stored as downloaded (JSON-stat) in `data/raw/` so the analysis runs offline.

| File | Source | Content |
|---|---|---|
| `data/raw/raw_isoc_iuapr.json`, `raw_isoc_iumapp_ihif.json` | Eurostat `isoc_ci_ac_i` | Individuals accessing health records online (I_IUAPR), booking appointments online (I_IUMAPP), seeking health information (I_IHIF); 2020, 2022, 2024 |
| `data/raw/raw_silc08.json` | Eurostat `hlth_silc_08` | Unmet needs for medical examination (too expensive / too far / waiting list), 16+, 2020–2024 |
| `data/raw/raw_cd_apr.json` | Eurostat `hlth_cd_apr` | Preventable and treatable mortality per 100 000, 2020–2023 |
| `data/raw/raw_sha11_hc.json` | Eurostat `hlth_sha11_hc` | Current health expenditure (PPS per inhabitant) and preventive care share (HC.6, % CHE), 2020–2023 |
| `data/raw/raw_rs_prs2.json` | Eurostat `hlth_rs_prs2` | Practising physicians and nurses per 100 000, 2020–2023 |
| `data/raw/raw_gdp.json` | Eurostat `nama_10_pc` | GDP per capita in PPS, 2020, 2022, 2024 |
| `data/raw/raw_age65.json` | Eurostat `demo_pjanind` | Share of population aged 65+, 2020, 2022, 2024 |
| `data/desi_aehr.csv` | European Commission, Digital Decade DESI platform, indicator `desi_aehr` | Access to e-health records score (0–100), reference years 2022–2025 |
| `data/params_hr.json` | OECD HCQO `DSD_HCQO@DF_PC` (HRV, 2021); Eurostat `demo_pjanbroad`, `hlth_co_disch1`, `hlth_co_disch2`, `hlth_sha11_hc` | Inputs for the Croatian scenario (avoidable admission rates, population 15+, discharges, inpatient expenditure) |

Eurostat data: © European Union, reuse permitted under the Eurostat policy (CC BY 4.0). OECD data: OECD terms and conditions. Digital Decade data: European Commission, reuse under Decision 2011/833/EU.

## How to run

```bash
pip install -r requirements.txt
python src/analysis.py      # tables, results and Figures 2, 3, S1
python src/fig1.py          # Figure 1 (conceptual framework)
```

Python 3.11 was used. Random seed is fixed (`numpy.random.default_rng(2026)`), so bootstrap and Monte Carlo results are reproducible. Figures are written as PNG and 300-dpi LZW-compressed TIFF.

## Repository layout

```
data/        raw API extracts and scenario parameters
src/         analysis.py (main pipeline), jsonstat.py (JSON-stat parser), fig1.py (framework figure)
results/     CSV/JSON outputs and the full console log of the last run
figures/     Figures 1–3 of the paper and the supplementary correlation heatmap
```

## Citation

Uroš, V., Birač, M., & Mihanović, D. (2026). *AI digital health platforms for preventive care, healthcare efficiency and organizational innovation*. Paper presented at AI-SMART 2026, Belgrade. Code and data: https://doi.org/10.5281/zenodo.22558363 (archived on Zenodo) and this repository.

## License

Code: MIT License (see `LICENSE`). Data files remain subject to the licences of their original providers listed above.
