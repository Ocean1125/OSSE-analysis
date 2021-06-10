# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%

# run my startup script
exec(open("../../src/functions/fun_loadStartup.py").read())
# load my functions
exec(open("../../src/functions/fun_loadMyFunctions.py").read())

# %%
### Load grid
grdFile = '../../data/raw/EACouter_varres_grd_mergedBLbry_uhroms.nc'
grid = loadRomsGrid(grdFile)


# %%
### Set file names
runningLocation = 'gdata'
if  runningLocation== 'local':
    hisFilePath='../../data/raw/'
elif runningLocation == 'gdata':
    hisFilePath='/g/data/fu5/deg581/EAC_2yr_truthRun_obsVerification/output/'
outFilePath='../../data/proc/'


# %%
chunks = {'ocean_time':1}

filelist = glob.glob(hisFilePath+'outer_his_*.nc')

for files in filelist: 
    print(files)
    # filelist.append(files)
ds = xr.open_mfdataset(filelist,chunks, data_vars='minimal',compat='override',coords='minimal',parallel=True, join='right') 


# %%
# load to some variables:
ssh = ds.zeta
sst = ds.temp.isel(s_rho=-1)
temp = ds.temp
salt = ds.salt
u = ds.u
v = ds.v


# %%
## save ds to output netcdf

out = ssh.to_dataset()
out['sst'] = sst
out['temp'] = temp
out['salt'] = salt
out['u'] = u
out['v'] = v

out.to_netcdf(path = outFilePath+'truth.nc')

