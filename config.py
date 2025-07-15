#parameters
START_YEAR = 1993           #observed period
END_YEAR = 2012
EXTENT = [-65, 40, 50, 80]  #North Atlantic

#cache settings
USE_CACHE = True
FORCE_RECOMPUTE = False

#projection
PROJECTION_PARAMS = {
    'central_longitude': -15.0,
    'central_latitude': 60.0,
    'standard_parallels': (55.0, 75.0)
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