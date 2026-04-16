import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import cftime
from numba import jit

# Function to plot the location given an index
def plot_index(index, lat, lon):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot all points in grey
    ax.scatter(lon, lat, c="grey", s=12)
    # Subset to desired index
    ax.scatter(lon[index], lat[index], c="red", s=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    return fig

# Function to return the array of estimated annual impact (EAI)
def get_EAI(input_data_path,
            data_source,
            warming_level,
            ssp,
            vp1,
            vp2):
    # load in GAM samples of EAI
    gamsamples_file = input_data_path+data_source+'/GAMsamples_expected_annual_impact_data_'+data_source+'_WL'+warming_level+'_SSP'+ssp+'_vp1='+vp1+'_vp2='+vp2+'.nc'
    gamsamples = Dataset(gamsamples_file)
    EAI = np.array(gamsamples.variables['sim_annual_impact'])

    return EAI

# Function to return the array of number of people in each grid cell
def get_Exp(input_data_path,
            ssp,
            ssp_year):
    # need the number of ppl in each grid cell to calculate the total cost as input is 'cost per person'
    exposure_netcdf = Dataset(input_data_path+'UKSSPs/Employment_SSP'+ssp+'_12km_Physical.nc')
    units = getattr(exposure_netcdf['time'], 'units')
    calendar = getattr(exposure_netcdf['time'], 'calendar')
    dates = cftime.num2date(exposure_netcdf.variables['time'][:], units, calendar)
    year_to_index = {k.timetuple().tm_year: v for v, k in enumerate(dates)}
    # pick out the right year (SSP year)
    index = year_to_index[int(ssp_year)]
    Exp = np.array(exposure_netcdf.variables['employment'][index])

    return Exp

# Function to get indices of land locations and the corresponding arrays of latitude and longitude
def get_ind_lat_lon(Exp,
                    input_data_path,
                    data_source,
                    warming_level,
                    ssp,
                    vp1,
                    vp2):
    # load in GAM samples of EAI
    gamsamples_file = input_data_path+data_source+'/GAMsamples_expected_annual_impact_data_'+data_source+'_WL'+warming_level+'_SSP'+ssp+'_vp1='+vp1+'_vp2='+vp2+'.nc'
    gamsamples = Dataset(gamsamples_file)

    # find indices of land locations
    ind = np.where(Exp < 9e30)
    # only apply in land location
    lon = np.array(gamsamples.variables['exposure_longitude'])[ind[0],ind[1]]
    lat = np.array(gamsamples.variables['exposure_latitude'])[ind[0],ind[1]]

    return (ind, lat, lon)

# Get EAI and exposure for a given location
def get_EAI_Exp_bundle(
        index,
        ind,
        input_data_path,
        calibration_opts,
        warming_level_opts,
        ssp_opts,
        vuln_param_1_opts,
        vuln_param_2_opts
        ):
    EAI_Exp_samples = {}
    for cal in calibration_opts:
        for wl in warming_level_opts:
            for ssp in ssp_opts:
                for vp1 in vuln_param_1_opts:
                    for vp2 in vuln_param_2_opts:
                        EAI = get_EAI(input_data_path,
                                    data_source = cal,
                                    warming_level = wl,
                                    ssp = ssp,
                                    vp1 = vp1,
                                    vp2 = vp2)
                        Exp = get_Exp(input_data_path,
                                      ssp = ssp,
                                      ssp_year = 2041 if wl == "2deg" else 2084)
                        # Extract the risk values for this location
                        xi = EAI[ind[0][index],ind[1][index],:]
                        # If EAI in a region is < 0, we will set it to 0
                        xi[np.where(xi < 0)] = 0
                        # Exponentiate
                        EAI_samples = (10**xi - 1)
                        # Get the exposure for this location
                        ppl = Exp[ind[0][index],ind[1][index]]

                        # Store values
                        key = (cal, wl, ssp, vp1, vp2)
                        EAI_Exp_samples[key] = (EAI_samples, ppl)
    return EAI_Exp_samples


# Jit version of function to calculate Y_e
@jit(nopython=True)
def calc_Ye_jit(
    EAI_Exp,
    decision_inputs
):
    EAI_samples, ppl = EAI_Exp
    # Initialize cost values for 1000 GAM samples
    cost = np.empty(1000)
    for k in range(1000): # loop over GAM samples
        # cost outcome for sample k
        cost[k] = decision_inputs[1]*ppl + decision_inputs[0]*(1-decision_inputs[2])*EAI_samples[k]
    # Average cost over the 1000 GAM samples
    Y_e = np.mean(cost)
    return Y_e


# # Function to calculate Y_e based on all the risk and decision inputs
# def calc_Ye(
#         index,
#         ind,
#         input_data_path,
#         risk_inputs, # length of 5: calibration, warming level, SSP, vuln param 1, vuln param 2
#         decision_inputs # length of 3: cost per day of work, annual cost per person of this decision, efficacy of this decision
# ):
#     # Get EAI
#     EAI = get_EAI(input_data_path,
#             data_source = risk_inputs[0],
#             warming_level = risk_inputs[1],
#             ssp = risk_inputs[2],
#             vp1 = risk_inputs[3],
#             vp2 = risk_inputs[4])

#     # Get exposure
#     # Exposure depends on SSP and SSP year (which comes from warming level)
#     # Get SSP year to use based on warming level
#     if risk_inputs[1] == "2deg":
#         ssp_year = 2041
#     else:
#         ssp_year = 2084
#     Exp = get_Exp(input_data_path,
#                   ssp = risk_inputs[2],
#                   ssp_year = ssp_year)
    
#     # Calculate Y_e
#     Y_e = calc_Ye_jit(index, ind, EAI, Exp, decision_inputs)
#     return Y_e


    