"""
One-off spatial setup: load the land-point grid indices and lat/lon
coordinates used to locate an analysis location within the underlying
NetCDF risk data.

This is computed once per run (not per location, and not per sample) and
is independent of which epistemic input samples get drawn afterwards -
sampling.generate_location_samples takes the resulting `ind` array as an
input.

NOTE: get_Exp and get_ind_lat_lon are imported from location_funcs,
treated here as a fixed black box. Not executed against the real data in
this environment - please verify against your actual location_funcs.py.
"""
from dataclasses import dataclass

import numpy as np

import config as cfg
from location_funcs import get_Exp, get_ind_lat_lon


@dataclass
class SpatialGrid:
    """The land-point index/lat/lon arrays used to look up a location.

    ind : tuple of two arrays (row indices, col indices) into the exposure
        raster, one entry per land location. Pass this to
        sampling.generate_location_samples / compute_Ye_matrix.
    lat, lon : np.ndarray
        Latitude/longitude of each land location, in the same order as ind.
        Used for mapping/plotting results, not for the sensitivity
        calculations themselves.
    """
    ind: tuple
    lat: np.ndarray
    lon: np.ndarray


def load_spatial_grid(
    reference_ssp: str = cfg.SSP_OPTS[0],
    reference_calibration: str = cfg.CALIBRATION_OPTS[0],
    reference_warming_level: str = cfg.WARMING_OPTS[0],
    reference_vuln1: str = cfg.VULN1_OPTS[0],
    reference_vuln2: str = cfg.VULN2_OPTS[0],
) -> SpatialGrid:
    """Load the land-point grid once, from an arbitrary valid input combination.

    The set of land locations (`ind`) and their lat/lon coordinates are a
    fixed property of the underlying spatial data - they don't depend on
    *which* calibration/warming/SSP/vulnerability combination is used to
    read them, only on a data file existing for that combination. The
    defaults reproduce the combination used in the original script
    (first calibration method, 2deg warming, SSP1); override if that
    particular file isn't available in your data directory.

    The SSP year is derived from reference_warming_level via
    cfg.WARMING_LEVEL_TO_SSP_YEAR, so it can't drift out of sync with the
    warming level the way a separately-passed year could.
    """
    reference_ssp_year = cfg.WARMING_LEVEL_TO_SSP_YEAR[reference_warming_level]

    Exp_array = get_Exp(
        input_data_path=cfg.DATA_DIR,
        ssp=reference_ssp,
        ssp_year=reference_ssp_year,
    )
    ind, lat, lon = get_ind_lat_lon(
        Exp_array,
        cfg.DATA_DIR,
        data_source=reference_calibration,
        warming_level=reference_warming_level,
        ssp=reference_ssp,
        vp1=reference_vuln1,
        vp2=reference_vuln2,
    )
    return SpatialGrid(ind=ind, lat=lat, lon=lon)