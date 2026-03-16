import numpy as np
import pandas as pd
from netCDF4 import Dataset
import cftime

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

# Function to calculate Y_e based on all the risk and decision inputs
def calc_Ye(
        index,
        ind,
        input_data_path,
        risk_inputs, # length of 5: calibration, warming level, SSP, vuln param 1, vuln param 2
        decision_inputs # length of 3: cost per day of work, annual cost per person of this decision, efficacy of this decision
):
    # Get EAI
    EAI = get_EAI(input_data_path,
            data_source = risk_inputs[0],
            warming_level = risk_inputs[1],
            ssp = risk_inputs[2],
            vp1 = risk_inputs[3],
            vp2 = risk_inputs[4])
    # state of nature (risk)
    xi = EAI[ind[0][index],ind[1][index],:]
    # If EAI in a region is < 0, we will set it to 0
    xi[np.where(xi < 0)] = 0

    # Get exposure
    # Exposure depends on SSP and SSP year (which comes from warming level)
    # Get SSP year to use based on warming level
    if risk_inputs[1] == "2deg":
        ssp_year = 2041
    else:
        ssp_year = 2084
    Exp = get_Exp(input_data_path,
                  ssp = risk_inputs[2],
                  ssp_year = ssp_year)
    # no. of people/jobs in each location
    ppl = Exp[ind[0][index],ind[1][index]]
    # Initialize cost values for 1000 GAM samples
    cost = np.empty(1000)
    for k in range(1000): # loop over GAM samples
        EAI_k = (10**xi[k] - 1)
        # cost outcome for sample k
        cost[k] = decision_inputs[1]*ppl + decision_inputs[0]*(1-decision_inputs[2])*EAI_k
    # Average cost over the 1000 GAM samples
    Y_e = np.mean(cost)
    return Y_e


# Function to find decision in a single cell
def decision_single_cell(ind,
         index,
         EAI,
         Exp,
         nd,
         decision_inputs,
         cost_per_day):
    # state of nature (risk)
    xi = EAI[ind[0][index],ind[1][index],:]
    # If EAI in a region is < 0, we will set it to 0
    xi[np.where(xi < 0)] = 0
    # no. of people/jobs in each location
    ppl = Exp[ind[0][index],ind[1][index]]
    # Initialize the optimal decision as an array with a single element
    opd = np.empty(1, dtype=np.int64)
    # If exposure in a region is 0, the decision should always be "do nothing"
    if ppl <= 0:
        opd[0] = 1
        exp_util = np.full(nd, np.nan)
        cost = np.full((nd, 1000), np.nan)
    # If there is exposure, find the Bayes optimal decision
    else:
        # calculate cost of each decision option for each of the 1000 GAM samples of risk
        cost = np.empty((nd,1000))
        for j in range(nd): # loop over number of decisions [do this in parallel?]
            for k in range(1000): # loop over GAM samples
                EAI_k = (10**xi[k] - 1)
                # cost outcome for decision j and sample k
                cost[j,k] = decision_inputs[j, 0]*ppl + cost_per_day*(1-decision_inputs[j, 1])*EAI_k

        # Calculate the utility of each decision attribute (cost and meeting objectives), i.e. the value of different
        # values of each to the decision maker - here assuming a linear
        # utility but this could be elicited from the decision maker (i.e. how risk averse they are)
        util_cost = -1 * cost

        # find expected (mean) utility
        exp_util = np.empty(nd)
        for j in range(nd):
            exp_util[j] = np.mean(util_cost[j,:])
        #find which decision optimises the expected utility
        opd = np.where(exp_util == max(exp_util))[0] + 1 #(add one because python indexing starts at 0)

    # Return: optimal decision, expected utility, utility scores, cost
    return opd, exp_util, util_cost, cost