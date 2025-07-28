#parameters
START_YEAR = 1993               #observed period
END_YEAR = 2012
EXTENT = [-65, 40, 50, 80]      #North Atlantic extent
USE_CACHE = True                #cache settings
FORCE_RECOMPUTE = False         #force recompute cache
VARIABILITY_DETREND_DEGREE = 4  #polynomial degree, 1=linear, 2=quadratic, 3=cubic, 4=quartic, etc...
PLOT_VARIABLE = 'variability'   #what to plot: 'trend', 'variability'

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
PROJECTION_PARAMS = {
    'central_longitude': -15.0,
    'central_latitude': 60.0,
    'standard_parallels': (55.0, 75.0)
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
TARGET_CMIP5_MODELS = [
    "ACCESS1-0",
    "bcc-csm1-1",
    "CanESM2",
    "CCSM4",
    "CMCC-CMS",
    "CNRM-CM5",
    "CSIRO-Mk3-6-0",
    "GFDL-ESM2G",
    "GFDL-ESM2M",
    "HadGEM2-CC",
    "HadGEM2-ES",
    "inmcm4",
    "IPSL-CM5A-MR",
    "MIROC-ESM",
    "MPI-ESM-LR",
    "MRI-CGCM3",
    "NorESM1-M",
    "NorESM1-ME"
]

TARGET_CMIP6_MODELS = [
    
]