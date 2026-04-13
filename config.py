"""
@author: L.G. van Dijk (l.g.vandijk1@students.uu.nl, luc.van.dijk@knmi.nl, luciusvandijk@gmail.com)

Configuration file for ODSL analysis and plotting.
"""

#parameters
CMIP_VERSION                  = 'CMIP5'           #CMIP version to be used: 'CMIP5' or 'CMIP6'
PROCESS_ALL_SCENARIOS         = True             #process all CMIP5/CMIP6 scenarios for comparison figure (slow)
START_YEAR                    = 1993              #observed period
END_YEAR                      = 2024
#EXTENT                       = [-65, 40, 50, 80] #North Atlantic (Richter et al., 2017)
#EXTENT                       = [-100, 20, 8, 80] #North Atlantic
EXTENT                        = [-100, 40, 8, 80] #North Atlantic including Caribbean
USE_CACHE                     = True              #cache settings
FORCE_RECOMPUTE               = False
VARIABILITY_DETREND_DEGREE    = 1                 #variability polynomial degree: 1=linear, 2=quadratic, etc...
#BE CAREFUL THAT FOR 'ODSL' OBSERVED AND MODELLED ARE NOT DIRECTLY COMPARABLE IN TERMS OF MAGNITUDE (CMIP MODELS HAVE 1986-2005, OBSERVED HAVE 1993-2012 AS REFERENCE PERIOD)
PLOT_VARIABLE                 = 'trend'           #what to plot: 'ODSL' (cm), 'trend' (mm/yr), 'variability' (cm)
PLOT_ALL_VARIABLES            = True              #plot all variables
CMIP5_FUTURE_SCENARIO         = 'rcp45'           #CMIP future scenario to be used after historical: 'rcp26', 'rcp45', 'rcp85'
CMIP6_FUTURE_SCENARIO         = 'ssp245'          #CMIP future scenario to be used after historical: 'ssp126', 'ssp245', 'ssp585'
PROCESS_PICONTROL             = False             #include piControl or not 
EOF_N_MODES                   = 5                 #number of EOF modes to plot
APPLY_SPATIAL_SMOOTHING       = True              #Gaussian smoothing altimetry dataset
SPATIAL_SMOOTHING_SIGMA       = 9.0               #Gaussian smoothing sigma
ALPHA                         = 0.05              #significance level
USE_ROTATED_EOF               = False             #use rotated EOFs (only without Monte Carlo significance testing, random noise has no structure to rotate)      
MONTE_CARLO_SIGNIFICANCE_TEST = True              #perform Monte Carlo significance testing
N_REALIZATIONS_MONTE_CARLO    = 1000              #number of Monte Carlo realizations for significance testing

N_MODES_OBSERVED = (END_YEAR - START_YEAR) + 1  #number of modes for observed period (1 mode per data point (per year))

PLOT_CONFIG = {'trend':       {'name': 'Trend',             'units': 'mm/yr', 'cmap': 'coolwarm'},
               'variability': {'name': 'Variability',       'units': 'cm',    'cmap': 'viridis'},
               'ODSL':        {'name': 'Dynamic Sea Level', 'units': 'cm',    'cmap': 'coolwarm'}}

#region definitions for overview trend figure (observed dataset)
KNMI_DUTCH_COAST = [[2.5, 53], [3.3, 51.5], [4.25, 52.25], [4.75, 53.3], [5.5, 53.6], [7, 53.75], [7, 55], [4, 54.5]]
NA_REGIONS       = {'North Atlantic':   {'extent': list(EXTENT),       'color': 'black'},
                    'Subpolar Gyre':    {'extent': [-60, -5, 50, 68],  'color': '#7B2D8E'},
                    'Labrador Current': {'extent': [-80, -45, 48, 75], 'color': "#1A13DA"},
                    'North Sea':        {'extent': [-4, 10, 51, 62],   'color': '#2CA02C'},
                    'Dutch Coast':      {'extent': [-4, 10, 51, 62],   'color': '#17BECF', 'polygon': KNMI_DUTCH_COAST},
                    'Gulf Stream':      {'extent': [-80, -55, 25, 45], 'color': '#D62728'},
                    'Gulf of Mexico':   {'extent': [-98, -80, 18, 30], 'color': '#FF7F0E'},
                    'Subtropical Gyre': {'extent': [-70, -15, 15, 45], 'color': '#BCBD22'}}

#projection
lon_min, lon_max, lat_min, lat_max = EXTENT
center_lon = (lon_min + lon_max) / 2
center_lat = (lat_min + lat_max) / 2

PROJECTION_PARAMS = {'central_longitude': center_lon, 'central_latitude': center_lat, 'standard_parallels': (lat_min + 10, lat_max - 10)}

#CMIP scenarios
CMIP_SCENARIOS = {"CMIP5": {"historical": "cmip5_zos_historical",
                            "rcp26":      "cmip5_zos_rcp26",
                            "rcp45":      "cmip5_zos_rcp45",
                            "rcp85":      "cmip5_zos_rcp85",
                            "piControl":  "cmip5_zos_piControl"},
                  "CMIP6": {"historical": "cmip6_zos_historical",
                            "ssp126":     "cmip6_zos_ssp126",
                            "ssp245":     "cmip6_zos_ssp245",
                            "ssp585":     "cmip6_zos_ssp585",
                            "piControl":  "cmip6_zos_piControl"}}

#target models from supplementary material table 1 Richter et al. 2017
# TARGET_CMIP5_MODELS = ["ACCESS1-0",
#                        "bcc-csm1-1",
#                        "CanESM2",
#                        "CCSM4",
#                        "CMCC-CMS",
#                        "CNRM-CM5",
#                        "CSIRO-Mk3-6-0",
#                        "GFDL-ESM2G",
#                        "GFDL-ESM2M",
#                        "HadGEM2-CC",
#                        "HadGEM2-ES",
#                        "inmcm4",
#                        "IPSL-CM5A-MR",
#                        "MIROC-ESM",
#                        "MPI-ESM-LR",
#                        "MRI-CGCM3",
#                        "NorESM1-M",
#                        "NorESM1-ME"]

#all target models CMIP5 (historical)
TARGET_CMIP5_MODELS = ["ACCESS1-0",
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
                       "NorESM1-ME"]

#all target models CMIP6 (historical)
TARGET_CMIP6_MODELS = ["ACCESS-CM2",
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
                       "NorESM2-LM"]
