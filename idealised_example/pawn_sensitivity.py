# Modified PAWN sensitivity analysis for comparison with VoI analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import safepython.PAWN_pmf as PAWN_pmf
import os

# os.chdir('./idealised_example')
from python_funcs import *

# # Import results of VoI in London
# lon_name = "London"
# lon_ind = 241
# lon_results = np.load(f"./results/voi_results_{lon_name.replace(' ', '_')}_{lon_ind}.npy", allow_pickle=True).item()

# Sample all input values
# ----- Constants and global set-up ------

# Data folder
DATA_DIR = "/home/aw23877/Documents/bda_sensitivity_paper/bda_risk_dec_sensitivity/data/"

# Number of decisions
nd = 3

# Define X
# Define risk inputs
calibration_opts = ["UKCP_raw", "UKCP_BC", "ChangeFactor"]
warming_opts = ["2deg", "4deg"]
ssp_opts = ["1", "2", "5"]
vuln1_opts = ["53.78", "54.5", "55.79"]
vuln2_opts = ["-4.597", "-4.1", "-3.804"]

# X labels
# Not worrying about non-financial yet
X_labels = ['Calibration method',
            'Warming level',
            'SSP',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Theta', # including risk
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

X_e_labels = ['Calibration method',
            'Warming level',
            'SSP',
            'Vulnerability parameter 1',
            'Vulnerability parameter 2',
            'Cost per day of work lost',
            'Annual cost per person of d2',
            'Effectiveness of d2',
            'Annual cost per person of d3',
            'Effectiveness of d3']

# Ranges for annual cost per person for each decision
AC_lows = [0, 150, 500]
AC_highs = [0, 350, 700]
# Ranges for efficacies for each decision
E_lows = [0, 0.3, 0.7]
E_highs = [0, 0.5, 0.9]

# Get land indices
Exp_array = get_Exp(input_data_path = DATA_DIR,
                    ssp = ssp_opts[0],
                    ssp_year = 2041)
ind, lat, lon = get_ind_lat_lon(Exp_array,
                                DATA_DIR,
                                data_source = calibration_opts[0],
                                warming_level = warming_opts[0],
                                ssp = ssp_opts[0],
                                vp1 = vuln1_opts[0],
                                vp2 = vuln2_opts[0])