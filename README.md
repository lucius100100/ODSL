# Ocean Dynamic Sea Level (ODSL) Analysis

## Overview

This repository contains code for analyzing Ocean Dynamic Sea Level (ODSL) using both observational data and CMIP model outputs. The analysis extends on the methodology outlined in Richter et al. (2017).

## Features

- **Observational ODSL Calculation**: Combines altimetry, geoid, and GIA data
- **CMIP Model Analysis**: Processes CMIP models (historical + RCP4.5)
- **Model-Observation Comparison**: Calculates Pattern Correlation Coefficient (PCC) and Root Mean Square Error (RMSE)
- **Sliding Window Analysis**: Identifies best-matching 20-year periods
- **Ensemble Statistics**: Creates multi-model ensemble means
- **Visualization**: Generates publication-quality figures

## Data Requirements

### Observational Data

1. **Altimetry Data**
   - Source: DUACS/CMEMS
   - DOI: 10.48670/moi-00148
   - Files: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_*.nc`
   - [Download Link](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description)

2. **Budget Data (Frederikse et al., 2020)**
   - DOI: 10.1038/s41586-020-2591-3
   - Files: `total.nc`
   - [Download Link](https://zenodo.org/records/3862995)

3. **GIA Data (Peltier et al., 2015)**
   - Model: ICE-6G_C (VM5a)
   - Files: `drad.1grid_O512.nc`, `dsea.1grid_O512.nc`
   - [Download Link](https://www.atmosp.physics.utoronto.ca/~peltier/data.php)

### Model Data

4. **CMIP Data**
   - Preprocessed by Dewi Le Bars (KNMI)
   - Variable: `zos` (sea surface height)
   - Scenarios: historical, RCP2.6, RCP4.5, RCP8.5
   - [Preprocessing Code](https://github.com/dlebars/CMIP_SeaLevel/blob/master/code/PreparePlaneVariables.py)

### Data Directory Structure

```
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

### Required Packages

```bash
# Core packages
pandas
numpy
xarray
xesmf  # For regridding
cartopy  # For map projections
matplotlib

# Additional dependencies
glob
os
```

## Project Structure

```
odsl-analysis/
├── ODSL_observational_and_modelled.py  # Main analysis script
├── figures/                            # Output directory for figures
└── README.md                           # This file
```

## Methodology

### ODSL Calculation

```
ODSL = MSL - Geoid - GIA
```

Where:
- **MSL**: Mean Sea Level from altimetry
- **Geoid**: Geoid change from Frederikse budget
- **GIA**: Glacial Isostatic Adjustment

### Area-Weighted Statistics

Following Richter et al. (2017), the analysis uses area-weighted statistics accounting for latitude-dependent grid cell areas:

- Weights: `w = cos(latitude)`
- Unbiased variance correction factor
- Area-weighted mean, variance, RMSE, and PCC

### Region of Interest

North Atlantic: 
- Longitude: -65° to 40°
- Latitude: 50° to 80°

## License

This project is licensed under the MIT License - see below:

```
MIT License

Copyright (c) 2024 Luc van Dijk

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
@software{odsl_analysis_2024,
  author = {Luc van Dijk},
  title = {Ocean Dynamic Sea Level (ODSL) Analysis},
  year = {2025},
  url = {https://github.com/lucius100100/ODSL}
}
```

## Data References

1. **Richter, K., Riva, R. E., & Drange, H. (2017)**. Impact of self-attraction and loading effects induced by shelf mass loading on projected regional sea level rise. *Geophysical Research Letters*, 44(3), 1466-1475.

2. **Frederikse, T., Landerer, F., Caron, L., et al. (2020)**. The causes of sea-level rise since 1900. *Nature*, 584(7821), 393-397.

3. **Peltier, W. R., Argus, D. F., & Drummond, R. (2015)**. Space geodesy constrains ice age terminal deglaciation: The global ICE‐6G_C (VM5a) model. *Journal of Geophysical Research: Solid Earth*, 120(1), 450-487.

4. **Le Bars, D. (2018)**. Uncertainty in sea level rise projections due to the dependence between contributors. *Earth's Future*, 6(9), 1275-1291.

## Contact

Luc van Dijk - l.g.vandijk1@students.uu.nl ; luc.van.dijk@knmi.nl

Project Link: [https://github.com/lucius100100/ODSL](https://github.com/lucius100100/ODSL)
