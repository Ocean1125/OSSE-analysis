# from netCDF4 import Dataset
# def loadRomsGrid(grdFile):
#     #grdFile = '/Users/z3533156/OneDrive - UNSW/Python/PUG/EACouter_varres_grd_mergedBLbry_uhroms.nc'
#     print('loading ROMS grid from '+grdFile)
#     grid_file = Dataset(grdFile, 'r')
#     grid_out = {
#         'name': "ROMS grid file"
#     }
#     grid_out["lon_rho"]  = grid_file.variables['lon_rho'][:,:]
#     grid_out["lat_rho"]  = grid_file.variables['lat_rho'][:,:]
#     grid_out["maskr"]  = grid_file.variables['mask_rho'][:,:]
#     try:
#         grid_out["n"]  = grid_file.variables['N'][:]
#     except:
#         print('file has no N variable, filling from size s_rho')
#         grid_out["n"]  = grid_file.variables['s_rho'][:].length()

#     grid_out["h"] = grid_file.variables['h'][:,:]
#     grid_out["dx"] = 1.0/grid_file.variables['pm'][:,:]
#     grid_out["dy"] = 1.0/grid_file.variables['pn'][:,:]

#     return grid_out

import xarray as xr

def loadRomsGrid(grdFile):
     print('loading ROMS grid from '+grdFile)
     grid_out = xr.open_dataset(grdFile)
     grid_out = grid_out.assign(dx=1.0/grid_out["pm"])
     grid_out = grid_out.assign(dy=1.0/grid_out["pn"])

     return grid_out