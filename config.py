"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Configuration file for ODSL analysis and plotting.
"""

#parameters
CMIP_VERSION               = 'CMIP6'           #CMIP version to be used: 'CMIP5' or 'CMIP6'
START_YEAR                 = 1993              #observed period
END_YEAR                   = 2024
#EXTENT                     = [-65, 40, 50, 80] #North Atlantic (Richter et al., 2017)
#EXTENT                     = [-100, 20, 8, 80] #North Atlantic
EXTENT                     = [-100, 20, 0, 80] #North Atlantic until equator
USE_CACHE                  = True              #cache settings
FORCE_RECOMPUTE            = False
VARIABILITY_DETREND_DEGREE = 2                 #variability polynomial degree: 1=linear, 2=quadratic, etc...
PLOT_VARIABLE              = 'trend'     #what to plot: 'ODSL', 'trend', 'variability'
#ODSL not implemented fully yet!
CMIP5_FUTURE_SCENARIO      = 'rcp45'           #CMIP future scenario to be used after historical
CMIP6_FUTURE_SCENARIO      = 'ssp245'
EOF_N_MODES                = 4                 #number of EOF modes to compute
APPLY_SPATIAL_SMOOTHING    = False              #Gaussian smoothing altimetry dataset
SPATIAL_SMOOTHING_SIGMA    = 3.0               #Gaussian smoothing sigma
APPLY_NaN_INPAINTING_BUDGET = False            #apply NaN inpainting around margins of budget data

PLOT_CONFIG = {
    'trend': {
        'name': 'Trend',
        'units': 'mm/yr',
        'cmap': 'coolwarm'
    },
    'variability': {
        'name': 'Variability',
        'units': 'cm',
        'cmap': 'viridis'
    }
}

#projection
lon_min, lon_max, lat_min, lat_max = EXTENT
center_lon = (lon_min + lon_max) / 2
center_lat = (lat_min + lat_max) / 2

PROJECTION_PARAMS = {
    'central_longitude': center_lon,
    'central_latitude': center_lat,
    'standard_parallels': (lat_min + 10, lat_max - 10)
}

#CMIP scenarios
CMIP_SCENARIOS = {
    "CMIP5": {
        "historical": "cmip5_zos_historical",
        "rcp26": "cmip5_zos_rcp26",
        "rcp45": "cmip5_zos_rcp45",
        "rcp85": "cmip5_zos_rcp85"
    },
    "CMIP6": {
        "historical": "cmip6_zos_historical",
        "ssp126": "cmip6_zos_ssp126",
        "ssp245": "cmip6_zos_ssp245",
        "ssp585": "cmip6_zos_ssp585"
    }
}

#target models from supplementary material table 1 Richter et al. 2017
# TARGET_CMIP5_MODELS = [
#     "ACCESS1-0",
#     "bcc-csm1-1",
#     "CanESM2",
#     "CCSM4",
#     "CMCC-CMS",
#     "CNRM-CM5",
#     "CSIRO-Mk3-6-0",
#     "GFDL-ESM2G",
#     "GFDL-ESM2M",
#     "HadGEM2-CC",
#     "HadGEM2-ES",
#     "inmcm4",
#     "IPSL-CM5A-MR",
#     "MIROC-ESM",
#     "MPI-ESM-LR",
#     "MRI-CGCM3",
#     "NorESM1-M",
#     "NorESM1-ME"
# ]

#all target models CMIP5 (historical)
TARGET_CMIP5_MODELS = [
    "ACCESS1-0",
    "bcc-csm1-1",
    "bcc-csm1-1-m",
    "CanESM2",
    "CCSM4",
    "CMCC-CESM",
    "CMCC-CM",
    "CMCC-CMS",
    "CNRM-CM5",
    "CSIRO-Mk3-6-0",
    "EC-EARTH",
    "GFDL-CM3",
    "GFDL-ESM2G",
    "GFDL-ESM2M",
    "GISS-E2-R",
    "GISS-E2-R-CC",
    "HadGEM2-CC",
    "HadGEM2-ES",
    "inmcm4",
    "IPSL-CM5A-LR",
    "IPSL-CM5A-MR",
    "IPSL-CM5B-LR",
    "MIROC5",
    "MIROC-ESM",
    "MIROC-ESM-CHEM",
    "MPI-ESM-LR",
    "MPI-ESM-MR",
    "MRI-CGCM3",
    "NorESM1-M",
    "NorESM1-ME"
]

#all target models CMIP6 (historical)
TARGET_CMIP6_MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "BCC-ESM1",
    "CAMS-CSM1-0",
    "CanESM5",
    "CanESM5-CanOE",
    "CAS-ESM2-0",
    "CESM2",
    "CESM2-FV2",
    "CESM2-WACCM",
    "CESM2-WACCM-FV2",
    "CIESM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CNRM-CM6-1",
    "CNRM-ESM2-1",
    "E3SM-1-0",
    "E3SM-1-1",
    "E3SM-1-1-ECA",
    "EC-Earth3",
    "EC-Earth3-AerChem",
    "EC-Earth3-CC",
    "EC-Earth3-Veg",
    "EC-Earth3-Veg-LR",
    "FGOALS-f3-L",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "GISS-E2-1-G-CC",
    "GISS-E2-1-H",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "INM-CM4-8",
    "INM-CM5-0",
    "IPSL-CM6A-LR",
    "MIROC-ES2L",
    "MIROC6",
    "MPI-ESM-1-2-HAM",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "NorCPM1",
    "NorESM2-LM"
]