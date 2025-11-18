# Ocean Dynamic Sea Level (ODSL) analysis

## Overview

This repository contains code for analyzing Ocean Dynamic Sea Level (ODSL) using both observational data and CMIP model outputs. The analysis extends on the methodology outlined in Richter et al. (2017).

## Usage

### First run (computes everything)

python odsl_analysis.py

### Subsequent runs (uses cache)

python odsl_analysis.py

### Force recompute

Edit config.py: FORCE_RECOMPUTE = True

then run

python odsl_analysis.py

### Clear cache

rm -rf cache/*

## Features

- **Observational ODSL calculation**: Combines altimetry, geoid, and GIA data
- **CMIP model analysis**: Processes CMIP5 and CMIP6 models (historical + future scenario)
- **Model-observation comparison**: Calculates Pattern Correlation Coefficient (PCC) and Root Mean Square Error (RMSE)
- **Sliding window analysis**: Identifies best-matching periods with a window size of max 31 years (1993-2024)
- **Ensemble statistics**: Creates multi-model ensemble means
- **Visualization**: Generates publication-quality figures

## Data requirements

1. **Altimetry data**
   - Source: DUACS/CMEMS
   - DOI: 10.48670/moi-00148
   - Files: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_*.nc`
   - [Download Link](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description)

2. **Budget Data (Frederikse et al., 2020)**
   - DOI: 10.1038/s41586-020-2591-3
   - Files: `total.nc`
   - [Download Link](https://zenodo.org/records/3862995)

3. **GIA data (Peltier et al., 2015)**
   - Model: ICE-6G_C (VM5a)
   - Files: `drad.1grid_O512.nc`, `dsea.1grid_O512.nc`
   - [Download Link](https://www.atmosp.physics.utoronto.ca/~peltier/data.php)

4. **CMIP data**
   - Preprocessed by Dewi Le Bars (KNMI)
   - Variable: `zos` (sea surface height)
   - Scenarios: historical, RCP2.6, RCP4.5, RCP8.5
   - [Preprocessing Code](https://github.com/dlebars/CMIP_SeaLevel/blob/master/code/PreparePlaneVariables.py)

### Data directory structure

```text
Data/
├── Altimetry/
│   └── cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_*.nc
├── Budget/
│   ├── Frederikse/
│   │   └── total.nc
│   └── GIA/
│       ├── drad.1grid_O512.nc
│       └── dsea.1grid_O512.nc
└── CMIP/
    ├── CMIP5/
    │   ├── cmip5_zos_historical/
    │   ├── cmip5_zos_rcp26/
    │   ├── cmip5_zos_rcp45/
    │   └── cmip5_zos_rcp85/
    └── CMIP6/
        ├── cmip6_zos_historical/
        ├── cmip6_zos_ssp126/
        ├── cmip6_zos_ssp245/
        └── cmip6_zos_ssp585/
```

## Installation

### Prerequisites

- Python 3.8+
- Conda or pip for package management

## Project structure

```text
ODSL/
├── odsl_analysis.py      # Main analysis script
├── data_loader.py        # Data loading functions with caching
├── utils.py              # Utility functions and cache management
├── config.py             # Configuration parameters
├── plotting.py           # Plotting
├── cache/                # Cached intermediate results (auto-created)
├── figures/              # Output directory for figures (auto-created)
├── requirements.txt      # Package requirements
└── README.md             # This file
```

## Cache size

Total cache size: ~5-10 GB
First run: 30-60 minutes
Cached runs: 1-5 minutes

## License

This project is licensed under the MIT License - see below:

```text
MIT License

Copyright (c) 2025 Luc van Dijk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{odsl_analysis_2025,
  author = {Luc van Dijk},
  title = {Ocean Dynamic Sea Level (ODSL) Analysis},
  year = {2025},
  url = {https://github.com/lucius100100/ODSL}
}
```

## Data references

1. **Richter, K., Øie Nilsen, J. E., Raj, R. P., Bethke, I., Johannessen, J. A., Slangen, A. B., & Marzeion, B. (2017)**. Northern North Atlantic sea level in CMIP5 climate models: evaluation of mean state, variability, and trends against altimetric observations. *Journal of Climate*, 30(23), 9383-9398.

2. **Frederikse, T., Landerer, F., Caron, L., et al. (2020)**. The causes of sea-level rise since 1900. *Nature*, 584(7821), 393-397.

3. **Peltier, W. R., Argus, D. F., & Drummond, R. (2015)**. Space geodesy constrains ice age terminal deglaciation: The global ICE‐6G_C (VM5a) model. *Journal of Geophysical Research: Solid Earth*, 120(1), 450-487.

4. **Le Bars, D. (2018)**. Uncertainty in sea level rise projections due to the dependence between contributors. *Earth's Future*, 6(9), 1275-1291.

## Contact

Luc van Dijk - <l.g.vandijk1@students.uu.nl> ; <luc.van.dijk@knmi.nl> ; <luciusvandijk@gmail.com>

Project Link: [https://github.com/lucius100100/ODSL](https://github.com/lucius100100/ODSL)
