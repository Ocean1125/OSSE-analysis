#from dask.distributed import Client, LocalCluster
#cluster = LocalCluster()
#client = Client(cluster)

#cd /g/data/fu5/deg581/OSSE_analysis/src/analyses

### run my startup script
exec(open("../functions/fun_loadStartup.py").read())
### load my functions
exec(open("../functions/fun_loadMyFunctions.py").read())

#%config InlineBackend.figure_format='png'

### Load grid
grdFile = '../../data/raw/EACouter_varres_grd_mergedBLbry_uhroms.nc'
grid = loadRomsGrid(grdFile)


### Other local functions

def loadOSSE(hisFilePath,prefix,filestring='0*.nc',overlapDays=7):
    chunks = {'ocean_time':1}
    filelist = glob.glob(hisFilePath+prefix+filestring)

    def preprocessRemoveLastDay(ds):
        '''remove the last 7 timesteps from each file'''
        return ds.isel(ocean_time = slice(0,-overlapDays))

#     for files in filelist: 
#         print(files)
    outName = xr.open_mfdataset(filelist,chunks, preprocess=preprocessRemoveLastDay, data_vars='minimal', compat='override', coords='minimal', parallel=True, join='right') 
    print('loaded from '+filelist[0]+' to '+filelist[-1])
    return outName

### function to load overlapping OSSE data
def loadOverlappedNetcdf(hisFilePath,prefix,filestring='0*.nc',overlapDays=7):
    chunks = {'ocean_time':1}
    filelist = sorted(glob.glob(hisFilePath+prefix+filestring))

    def preprocessRemoveLastDay(ds):
        '''remove the last 7 timesteps from each file'''
        return ds.isel(ocean_time = slice(0,-overlapDays))

    outName = xr.open_mfdataset(filelist,chunks, preprocess=preprocessRemoveLastDay, data_vars='minimal', compat='override', coords='minimal', parallel=True, join='right') 
    print('loaded from '+filelist[0]+' to '+filelist[-1])
    return outName

def dropDuplicateTimes(inputData):
    _, index = np.unique(inputData['ocean_time'], return_index=True)
    out = inputData.isel(ocean_time=index)
    return out

def calc_rmseSpatial(input1,input2,etaRange,xiRange):
    err2 = (input1-input2)**2
    # err=err2**(1/2)
    mse = indexMeanMetric(err2,etaRange,xiRange) #mean square error MSE
    output = mse**(1/2)
    return mse,output

# define function for calculating spatial mean
def indexMeanMetric(input,etaRange,xiRange):
    ''' iRange and jRange are converted to slices, so they are the start/end values of the range '''
    output = input.isel(eta_rho=slice(etaRange[0],etaRange[1]), xi_rho=slice(xiRange[0],xiRange[1])).mean(dim='eta_rho', skipna=True).mean(dim='xi_rho', skipna=True)
    return output

def simple_TimeSeries(inputDataX, inputDataY, plt_kwargs={}, ax=None):
    #Plotting
    if ax is None:
        ax = plt.gca()   
    hOut = ax.plot(inputDataX, inputDataY, **plt_kwargs)
    ax.grid(color='black', alpha=0.2, linestyle='--')
    return(hOut)

def plot_spatialMapSubplot(toPlotData, ax=None, pcol_kwargs={}, cont_kwargs={}, kde_kwargs={}):
    #Plotting
    if ax is None:
        ax = plt.gca()
    # plt.subplot(projection=ccrs.PlateCarree())
#     ax = fig.add_subplot(gs[ax], projection=ccrs.PlateCarree())
    ax.set_extent([147, 162.5, -42, -25])
    feature = ax.add_feature(Coast, edgecolor='black',facecolor='gray')
    im=toPlotData.plot.pcolormesh('lon_rho','lat_rho',ax=ax, add_colorbar=False, **pcol_kwargs)       
    toPlotData.plot.contour('lon_rho','lat_rho',ax=ax, **cont_kwargs)
    gl = ax.gridlines(draw_labels=True,
                     color='black', alpha=0.2, linestyle='--')
        #gl.xformatter = LONGITUDE_FORMATTER
        #gl.yformatter = LATITUDE_FORMATTER
    gl.right_labels = False
    gl.top_labels = False
    cax = inset_axes(ax,
                 width="5%",  # width = 10% of parent_bbox width
                 height="50%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(.07,.39, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
    fig.colorbar(im, cax=cax)

def plot_spatialMapGSSubplot(toPlotData, ax=None, pcol_kwargs={}, cont_kwargs={}, kde_kwargs={}):
    #Plotting
    if ax is None:
        ax = plt.gca()
    # plt.subplot(projection=ccrs.PlateCarree())
#     ax = fig.add_subplot(gs[ax], projection=ccrs.PlateCarree())
    ax.set_extent([147, 162.5, -42, -25])
    feature = ax.add_feature(Coast, edgecolor='black',facecolor='gray')
    im=toPlotData.plot.pcolormesh('lon_rho','lat_rho',ax=ax, add_colorbar=False, **pcol_kwargs)       
    toPlotData.plot.contour('lon_rho','lat_rho',ax=ax, **cont_kwargs)
    gl = ax.gridlines(draw_labels=True,
                     color='black', alpha=0.2, linestyle='--')
        #gl.xformatter = LONGITUDE_FORMATTER
        #gl.yformatter = LATITUDE_FORMATTER
    gl.right_labels = False
    gl.top_labels = False
    cax = inset_axes(ax,
                 width="5%",  # width = 10% of parent_bbox width
                 height="50%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(.07,.39, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
    fig.colorbar(im, cax=cax)

def calc_boxSpatial(grid,etaRange,xiRange):
    pt00lo = grid['lon_rho'].isel(eta_rho=etaRange[0],xi_rho=xiRange[0]).values
    pt01lo = grid['lon_rho'].isel(eta_rho=etaRange[0],xi_rho=xiRange[1]).values
    pt10lo = grid['lon_rho'].isel(eta_rho=etaRange[1],xi_rho=xiRange[0]).values
    pt11lo = grid['lon_rho'].isel(eta_rho=etaRange[1],xi_rho=xiRange[1]).values
    pt00la = grid['lat_rho'].isel(eta_rho=etaRange[0],xi_rho=xiRange[0]).values
    pt01la = grid['lat_rho'].isel(eta_rho=etaRange[0],xi_rho=xiRange[1]).values
    pt10la = grid['lat_rho'].isel(eta_rho=etaRange[1],xi_rho=xiRange[0]).values
    pt11la = grid['lat_rho'].isel(eta_rho=etaRange[1],xi_rho=xiRange[1]).values
    boxLo = np.array([pt00lo, pt01lo, pt11lo, pt10lo, pt00lo])
    boxLa = np.array([pt00la, pt01la, pt11la, pt10la, pt00la])
    return boxLo,boxLa

def datestring_to_serial_day(datestring,epochY=1990,epochm=1,epochd=1,epochH=0,epochM=0):
    import pandas as pd
    import datetime
    serial_day_timedelta = pd.to_datetime(datestring) - datetime.datetime(epochY,epochm,epochd,epochH,epochM)
    corrected_serial_day_number = serial_day_timedelta.days + serial_day_timedelta.seconds/86400
    return corrected_serial_day_number


def serial_day_to_datestring(day,epochY=1990,epochm=1,epochd=1,epochH=0,epochM=0):
    import datetime
    corrected_date = datetime.datetime(epochY,epochm,epochd,epochH,epochM) + datetime.timedelta(day)
    return corrected_date.strftime("%Y-%m-%d %H:%M")  

### Functions for calculating metrics

def loadOSSEdata(hisOSSEFilePath,prefixForecast,prefixAnalysis,dates):
    filenameForecast=hisOSSEFilePath+prefixForecast+'0'+str(dates)+'.nc'
    filenameAnalysis=hisOSSEFilePath+prefixAnalysis+'0'+str(dates)+'.nc'
    sshForecast=xr.open_dataset(filenameForecast).zeta.load()
    sshAnalysis=xr.open_dataset(filenameAnalysis).zeta.load()
    sstForecast=xr.open_dataset(filenameForecast).temp.isel(s_rho=-1).load()
    sstAnalysis=xr.open_dataset(filenameAnalysis).temp.isel(s_rho=-1).load()
    return sshForecast, sshAnalysis, sstForecast, sstAnalysis

def loadTruthdata(hisTruthFilePath,prefixTruth,dates):
    filenameTruth   =hisTruthFilePath+prefixTruth+'0'+str(dates)+'.nc'
    sshTruth   =xr.open_dataset(filenameTruth).zeta.load()
    sstTruth=xr.open_dataset(filenameTruth).temp.isel(s_rho=-1).load()
    return sshTruth, sstTruth

def compileOSSETimeMetricSSH(hisOSSEFilePath,prefixForecast,prefixAnalysis,etaRangeMetric,xiRangeMetric,datelist):
    for dates in datelist:
        sshForecast,sshAnalysis,sstForecast,sstAnalysis = loadOSSEdata(hisOSSEFilePath,prefixForecast,prefixAnalysis,dates)       
        if dates == datelist[0]:
            ssh_metricForecast = indexMeanMetric(sshForecast, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            ssh_metricAnalysis = indexMeanMetric(sshAnalysis, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
        elif dates != datelist[0]:
            temp = indexMeanMetric(sshForecast, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            ssh_metricForecast = xr.merge([ssh_metricForecast, temp])
            temp = indexMeanMetric(sshAnalysis, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            ssh_metricAnalysis = xr.merge([ssh_metricAnalysis, temp])
    return ssh_metricForecast,ssh_metricAnalysis

def compileOSSETimeMetricSST(hisOSSEFilePath,prefixForecast,prefixAnalysis,etaRangeMetric,xiRangeMetric,datelist):
    for dates in datelist:
        sshForecast,sshAnalysis,sstForecast,sstAnalysis = loadOSSEdata(hisOSSEFilePath,prefixForecast,prefixAnalysis,dates)       
        if dates == datelist[0]:
            sst_metricForecast = indexMeanMetric(sstForecast, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            sst_metricAnalysis = indexMeanMetric(sstAnalysis, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
        elif dates != datelist[0]:
            temp = indexMeanMetric(sstForecast, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            sst_metricForecast = xr.merge([sst_metricForecast, temp])
            temp = indexMeanMetric(sstAnalysis, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            sst_metricAnalysis = xr.merge([sst_metricAnalysis, temp])
    return sst_metricForecast,sst_metricAnalysis

def compileTruthTimeMetric(hisTruthFilePath,prefixTruth,etaRangeMetric,xiRangeMetric,datelist):
    for dates in datelist:
        sshTruth,sstTruth = loadTruthdata(hisTruthFilePath,prefixTruth,dates)
        if dates == datelist[0]:
            ssh_metricTruth = indexMeanMetric(sshTruth, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            sst_metricTruth = indexMeanMetric(sstTruth, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
        elif dates != datelist[0]:
            temp = indexMeanMetric(sshTruth, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            ssh_metricTruth = xr.merge([ssh_metricTruth, temp])
            temp = indexMeanMetric(sstTruth, etaRange=etaRangeMetric, xiRange=xiRangeMetric).to_dataset(name = str(dates))
            sst_metricTruth = xr.merge([sst_metricTruth, temp])
    return ssh_metricTruth, sst_metricTruth

### Functions for loading and processing ROMS data nicely... ROMS FUNCTIONS

from xgcm import Grid

def processROMSGrid(ds):
    ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho', 'xi_psi': 'xi_u', 'eta_psi': 'eta_v'})

    coords={'X':{'center':'xi_rho', 'inner':'xi_u'}, 
        'Y':{'center':'eta_rho', 'inner':'eta_v'}, 
        'Z':{'center':'s_rho', 'outer':'s_w'}}

    grid = Grid(ds, coords=coords, periodic=[])

    if ds.Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho/ds.h)
        Zo_w = ds.hc * (ds.s_w - ds.Cs_w) + ds.Cs_w * ds.h
        z_w = Zo_w + ds.zeta * (1 + Zo_w/ds.h)
    elif ds.Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho
        Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
        z_w = Zo_w * (ds.zeta + ds.h) + ds.zeta

    ds.coords['z_w'] = z_w.where(ds.mask_rho, 0).transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho')
    ds.coords['z_rho'] = z_rho.where(ds.mask_rho, 0).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
    # Other Option is to transpose arrays and fill NaNs with a minimal depth
    # ds['z_rho'] = z_rho.transpose(*('time', 's_rho','yh','xh'),transpose_coords=False).fillna(hmin)
    # ds['z_w'] = z_w.transpose(*('time', 's_w','yh','xh'),transpose_coords=False).fillna(hmin)
    ds.coords['z_rho0'] = z_rho.mean(dim='ocean_time')

     # interpolate depth of levels at U and V points
    ds['z_u'] = grid.interp(ds['z_rho'], 'X', boundary='fill')
    ds['z_v'] = grid.interp(ds['z_rho'], 'Y', boundary='fill')

    ds['pm_v'] = grid.interp(ds.pm, 'Y')
    ds['pn_u'] = grid.interp(ds.pn, 'X')
    ds['pm_u'] = grid.interp(ds.pm, 'X')
    ds['pn_v'] = grid.interp(ds.pn, 'Y')
    ds['pm_psi'] = grid.interp(grid.interp(ds.pm, 'Y'),  'X') # at psi points (eta_v, xi_u) 
    ds['pn_psi'] = grid.interp(grid.interp(ds.pn, 'X'),  'Y') # at psi points (eta_v, xi_u)

    ds['dx'] = 1/ds.pm
    ds['dx_u'] = 1/ds.pm_u
    ds['dx_v'] = 1/ds.pm_v
    ds['dx_psi'] = 1/ds.pm_psi

    ds['dy'] = 1/ds.pn
    ds['dy_u'] = 1/ds.pn_u
    ds['dy_v'] = 1/ds.pn_v
    ds['dy_psi'] = 1/ds.pn_psi

    ds['dz'] = grid.diff(ds.z_w, 'Z', boundary='fill')
    ds['dz_w'] = grid.diff(ds.z_rho, 'Z', boundary='fill')
    ds['dz_u'] = grid.interp(ds.dz, 'X')
    ds['dz_w_u'] = grid.interp(ds.dz_w, 'X')
    ds['dz_v'] = grid.interp(ds.dz, 'Y')
    ds['dz_w_v'] = grid.interp(ds.dz_w, 'Y')

    ds['dA'] = ds.dx * ds.dy

    metrics = {
        ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'], # X distances
        ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
        ('Z',): ['dz', 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v'], # Z distances
        ('X', 'Y'): ['dA'] # Areas
    }
    grid = Grid(ds, coords=coords, metrics=metrics, periodic=[])

    return ds

def makeROMSGridObject(gridIn):
    gridOut = Grid(gridIn, 
    coords={'X':{'center':'xi_rho', 'inner':'xi_u'}, 
    'Y':{'center':'eta_rho', 'inner':'eta_v'}, 
    'Z':{'center':'s_rho', 'outer':'s_w'}},
    periodic=False)
    return gridOut

def horizontalSectionROMS(grid,inputVal,targetDepth):
    # v2
    output = grid.transform(inputVal, 'Z', targetDepth,
                                    target_data=inputVal['z_rho'],
                                    method='linear').squeeze()
    return output


def loadOSSEFileList(hisFilePath,prefix,filelist,overlapDays=7):
    chunks = {'ocean_time':1}
    def preprocessRemoveLastDay(ds):
        '''remove the last 7 timesteps from each file'''
        return ds.isel(ocean_time = slice(0,-overlapDays))

    outName = xr.open_mfdataset(filelist,chunks, preprocess=preprocessRemoveLastDay, data_vars='minimal', compat='override', coords='minimal', parallel=True, join='right') 
    print('loaded from '+filelist[0]+' to '+filelist[-1])
    return outName

def loadOverlappedNetcdfFileList(hisFilePath,prefix,filelist,overlapDays=7):
    chunks = {'ocean_time':1}
    def preprocessRemoveLastDay(ds):
        '''remove the last 7 timesteps from each file'''
        return ds.isel(ocean_time = slice(0,-overlapDays))

    outName = xr.open_mfdataset(filelist,chunks, preprocess=preprocessRemoveLastDay, data_vars='minimal', compat='override', coords='minimal', parallel=True, join='right') 
    print('loaded from '+filelist[0]+' to '+filelist[-1])
    return outName

def loadNetcdfFileListAverages(hisFilePath,prefix,filelist):

    outName = xr.open_mfdataset(filelist, data_vars='minimal', compat='override', coords='minimal', parallel=True, join='right') 
    print('loaded from '+filelist[0]+' to '+filelist[-1])
    return outName


def generateFileList(hisFilePath,prefix,datelist):
    filelist=[hisFilePath+prefix+'0'+str(datelist[0])+'.nc']
    for dates in datelist[1:]:
        filenameToAppend=hisFilePath+prefix+'0'+str(dates)+'.nc'
        filelist.append(filenameToAppend)

#         print(filelist)
    return filelist

def calc_dailyDownsample(ds):
    ds_withtime = ds.drop([ var for var in ds.variables if not 'ocean_time' in ds[var].dims ])
    ds_timeless = ds.drop([ var for var in ds.variables if     'ocean_time' in ds[var].dims ])
    ds_workaround = xr.merge([ds_timeless, ds_withtime.resample(ocean_time='5D').mean('ocean_time')])
    return ds_workaround

def calc_timeMeanEKE(input):
    u_eastward_top = input.u_eastward.isel(s_rho=-1)
    v_northward_top = input.v_northward.isel(s_rho=-1)

    u_bar = u_eastward_top.mean("ocean_time")
    v_bar = v_northward_top.mean("ocean_time")

    mke = 0.5*(u_bar**2 + v_bar**2)

    u_prime = u_eastward_top - u_bar
    v_prime = v_northward_top - v_bar

    eke = 0.5*(u_prime**2 + v_prime**2)

    eketimemean = (eke*input.dA).sum(dim=['eta_rho','xi_rho'],skipna=True)/(input.dA).sum(dim=['eta_rho','xi_rho'],skipna=True)
    return eketimemean

def calc_MKEandEKElevel(input,level):
    u_eastward = input.u_eastward
    v_northward = input.v_northward

    u_bar = u_eastward.isel(s_rho=level).mean("ocean_time")
    v_bar = v_northward.isel(s_rho=level).mean("ocean_time")

    mke = 0.5*(u_bar**2 + v_bar**2)

    u_prime = u_eastward.isel(s_rho=level) - u_bar
    v_prime = v_northward.isel(s_rho=level) - v_bar

    eke = 0.5*(u_prime**2 + v_prime**2)
    
    input['mke']=mke
    input['eke']=eke
    return input

def calc_MKEandEKEall(input):
    u_eastward = input.u_eastward
    v_northward = input.v_northward

    u_bar = u_eastward.mean("ocean_time")
    v_bar = v_northward.mean("ocean_time")

    mke = 0.5*(u_bar**2 + v_bar**2)

    u_prime = u_eastward - u_bar
    v_prime = v_northward - v_bar

    eke = 0.5*(u_prime**2 + v_prime**2)
    
    input['mke']=mke
    input['eke']=eke
    return input

def horizontalSectionROMS(grid,inputVal,targetDepth):
    # v2
    output = grid.transform(inputVal, 'Z', targetDepth,
                                    target_data=inputVal['z_rho0'],
                                    method='linear').squeeze()
    return output

def calc_areaAverageROMS(input,grid):
    output = (input*grid.dA).sum(dim=['eta_rho','xi_rho'],skipna=True)/(grid.dA).sum(dim=['eta_rho','xi_rho'],skipna=True)
    return output

#### NOW START

hisTruthFilePath='/g/data/fu5/deg581/EAC_2yr_truthRun_obsVerification/output/' # Truth file settings

hisOSSEFilePath1='/g/data/fu5/eac/OSSEs/OSSE_ssh_sst_is4dvar/output/'
modOSSEFilePath1='/g/data/fu5/eac/OSSEs/OSSE_ssh_sst_is4dvar/output/'
obsOSSEFilePath1='/g/data/fu5/eac/OSSEs/OSSE_ssh_sst_is4dvar/output/'

hisOSSEFilePath2='/g/data/fu5/deg581/OSSE_SSHSST_XBT/output/'
modOSSEFilePath2='/g/data/fu5/deg581/OSSE_SSHSST_XBT/output/'
obsOSSEFilePath2='/g/data/fu5/deg581/OSSE_SSHSST_XBT/output/'

hisOSSEFilePath3='/g/data/fu5/deg581/OSSE_SSHSST_XBT_NORTH/output/'
modOSSEFilePath3='/g/data/fu5/deg581/OSSE_SSHSST_XBT_NORTH/output/'
obsOSSEFilePath3='/g/data/fu5/deg581/OSSE_SSHSST_XBT_NORTH/output/'

hisOSSEFilePath4='/g/data/fu5/deg581/OSSE_SSHSST_XBT_SOUTH/output/'
modOSSEFilePath4='/g/data/fu5/deg581/OSSE_SSHSST_XBT_SOUTH/output/'
obsOSSEFilePath4='/g/data/fu5/deg581/OSSE_SSHSST_XBT_SOUTH/output/'

prefixForecast='roms_fwd_outer0_'
prefixAnalysis='roms_fwd_outer1_'
prefixAverage='roms_avg_outer1_'
prefixTruth='outer_his_'
prefixTruthAverage='outer_avg_'

prefixObs='obs_0'
prefixMod='eac_mod_'
prefixPert='outer_his_'

outFigurePath='../cache/out/'

## SET TIME RANGE

timeRange = [8005, 8201]
datelist = np.array(range(timeRange[0],timeRange[1],4))

## LOAD DATA


filelistTruth=generateFileList(hisTruthFilePath,prefixTruth,datelist)
filelist_SSHSST=generateFileList(hisOSSEFilePath1,prefixAnalysis,datelist)
filelist_SSHSST_XBT=generateFileList(hisOSSEFilePath2,prefixAnalysis,datelist)
filelist_SSHSST_XBT_N=generateFileList(hisOSSEFilePath3,prefixAnalysis,datelist)
filelist_SSHSST_XBT_S=generateFileList(hisOSSEFilePath4,prefixAnalysis,datelist)

truth=loadOverlappedNetcdfFileList(hisTruthFilePath,prefixTruth,filelist=filelistTruth,overlapDays=7)
OSSE_SSHSST=loadOSSEFileList(hisOSSEFilePath1,prefixAnalysis, filelist=filelist_SSHSST,overlapDays=7)
OSSE_SSHSST_XBT=loadOSSEFileList(hisOSSEFilePath2,prefixAnalysis, filelist=filelist_SSHSST_XBT,overlapDays=7)
OSSE_SSHSST_XBT_N=loadOSSEFileList(hisOSSEFilePath3,prefixAnalysis, filelist=filelist_SSHSST_XBT_N,overlapDays=7)
OSSE_SSHSST_XBT_S=loadOSSEFileList(hisOSSEFilePath4,prefixAnalysis, filelist=filelist_SSHSST_XBT_S,overlapDays=7)

## DOWNSAMPLE DATA
truth = calc_dailyDownsample(truth)
OSSE_SSHSST = calc_dailyDownsample(OSSE_SSHSST)
OSSE_SSHSST_XBT = calc_dailyDownsample(OSSE_SSHSST_XBT)
OSSE_SSHSST_XBT_N = calc_dailyDownsample(OSSE_SSHSST_XBT_N)
OSSE_SSHSST_XBT_S = calc_dailyDownsample(OSSE_SSHSST_XBT_S)

## process into more useful format
truth = processROMSGrid(truth)
OSSE_SSHSST=processROMSGrid(OSSE_SSHSST)
OSSE_SSHSST_XBT=processROMSGrid(OSSE_SSHSST_XBT)
OSSE_SSHSST_XBT_N=processROMSGrid(OSSE_SSHSST_XBT_N)
OSSE_SSHSST_XBT_S=processROMSGrid(OSSE_SSHSST_XBT_S)

# MAKE XGCM GRID OBJECT
grid = makeROMSGridObject(truth)


## BEGIN

# calculate mke/eke statistics

truth = calc_MKEandEKEall(truth)
OSSE_SSHSST=calc_MKEandEKEall(OSSE_SSHSST)
OSSE_SSHSST_XBT=calc_MKEandEKEall(OSSE_SSHSST_XBT)
OSSE_SSHSST_XBT_N=calc_MKEandEKEall(OSSE_SSHSST_XBT_N)
OSSE_SSHSST_XBT_S=calc_MKEandEKEall(OSSE_SSHSST_XBT_S)

# 500m cross-sections

truth_mke_500 = horizontalSectionROMS(grid, truth.mke, np.array([-500]))
truth_eke_500 = horizontalSectionROMS(grid, truth.eke, np.array([-500])), print('done')

OSSE_SSHSST_mke_500 = horizontalSectionROMS(grid, OSSE_SSHSST.mke, np.array([-500]))
OSSE_SSHSST_eke_500 = horizontalSectionROMS(grid, OSSE_SSHSST.eke, np.array([-500])), print('done')

OSSE_SSHSST_XBT_N_mke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT_N.mke, np.array([-500]))
OSSE_SSHSST_XBT_N_eke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT_N.eke, np.array([-500])), print('done')

OSSE_SSHSST_XBT_S_mke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT_S.mke, np.array([-500]))
OSSE_SSHSST_XBT_S_eke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT_S.eke, np.array([-500])), print('done')

OSSE_SSHSST_XBT_mke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT.mke, np.array([-500]))
OSSE_SSHSST_XBT_eke_500 = horizontalSectionROMS(grid, OSSE_SSHSST_XBT.eke, np.array([-500])), print('done')



truth_mke_500 = truth_mke_500
truth_eke_500 = truth_eke_500[0]
OSSE_SSHSST_mke_500 = OSSE_SSHSST_mke_500
OSSE_SSHSST_eke_500 = OSSE_SSHSST_eke_500[0]
OSSE_SSHSST_XBT_N_mke_500 =OSSE_SSHSST_XBT_N_mke_500
OSSE_SSHSST_XBT_N_eke_500 =OSSE_SSHSST_XBT_N_eke_500[0]
OSSE_SSHSST_XBT_S_mke_500 =OSSE_SSHSST_XBT_S_mke_500
OSSE_SSHSST_XBT_S_eke_500 =OSSE_SSHSST_XBT_S_eke_500[0]
OSSE_SSHSST_XBT_mke_500 =OSSE_SSHSST_XBT_mke_500
OSSE_SSHSST_XBT_eke_500 =OSSE_SSHSST_XBT_eke_500[0]

truth_mke_500.load(), print('done')
OSSE_SSHSST_mke_500.load(), print('done')
OSSE_SSHSST_XBT_N_mke_500.load(), print('done')
OSSE_SSHSST_XBT_S_mke_500.load(), print('done')
OSSE_SSHSST_XBT_mke_500.load(), print('done')

truth_eke_500.load(), print('done')
OSSE_SSHSST_eke_500.load(), print('done')
OSSE_SSHSST_XBT_N_eke_500.load(), print('done')
OSSE_SSHSST_XBT_S_eke_500.load(), print('done')
OSSE_SSHSST_XBT_eke_500.load(), print('done')

# surface eke stats

truth_mke_0 = truth.mke.isel(s_rho=-1)
truth_eke_0 = truth.eke.isel(s_rho=-1), print('done')

OSSE_SSHSST_mke_0 = OSSE_SSHSST.mke.isel(s_rho=-1)
OSSE_SSHSST_eke_0 = OSSE_SSHSST.eke.isel(s_rho=-1), print('done')

OSSE_SSHSST_XBT_N_mke_0 = OSSE_SSHSST_XBT_N.mke.isel(s_rho=-1)
OSSE_SSHSST_XBT_N_eke_0 = OSSE_SSHSST_XBT_N.eke.isel(s_rho=-1), print('done')

OSSE_SSHSST_XBT_S_mke_0 = OSSE_SSHSST_XBT_S.mke.isel(s_rho=-1)
OSSE_SSHSST_XBT_S_eke_0 = OSSE_SSHSST_XBT_S.eke.isel(s_rho=-1), print('done')

OSSE_SSHSST_XBT_mke_0 = OSSE_SSHSST_XBT.mke.isel(s_rho=-1)
OSSE_SSHSST_XBT_eke_0 = OSSE_SSHSST_XBT.eke.isel(s_rho=-1), print('done')



truth_eke_0 = truth_eke_0[0]
OSSE_SSHSST_eke_0 = OSSE_SSHSST_eke_0[0]
OSSE_SSHSST_XBT_N_eke_0 = OSSE_SSHSST_XBT_N_eke_0[0]
OSSE_SSHSST_XBT_S_eke_0 = OSSE_SSHSST_XBT_S_eke_0[0]
OSSE_SSHSST_XBT_eke_0 = OSSE_SSHSST_XBT_eke_0[0]

truth_mke_0.load()
truth_eke_0.load(), print('done')

OSSE_SSHSST_mke_0.load()
OSSE_SSHSST_eke_0.load(), print('done')

OSSE_SSHSST_XBT_N_mke_0.load()
OSSE_SSHSST_XBT_N_eke_0.load(), print('done')

OSSE_SSHSST_XBT_S_mke_0.load()
OSSE_SSHSST_XBT_S_eke_0.load(), print('done')

OSSE_SSHSST_XBT_mke_0.load()
OSSE_SSHSST_XBT_eke_0.load(), print('done')

# first test plots

#plt.figure()
#((truth_mke_500)).plot(vmin=0,vmax=0.03)#/truth_mke_500).plot(vmin=-100,vmax=100)
#plt.figure()
#((OSSE_SSHSST_mke_500)).plot(vmin=0,vmax=0.03)#/truth_mke_500).plot(vmin=-100,vmax=100)
#plt.figure()
#((OSSE_SSHSST_mke_500-truth_mke_500)).plot()#/truth_mke_500).plot(vmin=-100,vmax=100)
#plt.figure()
#((OSSE_SSHSST_mke_500/truth_mke_500)).plot(vmin=-2,vmax=2,cmap='cmo.balance')
#plt.figure()
#((OSSE_SSHSST_mke_500-truth_mke_500)/truth_mke_500).plot(vmin=-2,vmax=2,cmap='cmo.balance')
#
#plt.figure()
#((((OSSE_SSHSST_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)).plot()
#plt.figure()
#(truth_eke_0.std(dim='ocean_time')).plot()
#plt.figure()
#((((OSSE_SSHSST_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)/(truth_eke_0.std(dim='ocean_time'))).plot(vmax=2)

## define
def addSubplot_spatialMap_contourf(input,levels,gs,nrow,ncol, labelText=None, pcol_kwargs={}, cont_kwargs={}, kde_kwargs={}):
    ax = fig.add_subplot(gs[nrow,ncol], projection=ccrs.PlateCarree())
    ax.set_extent([148, 161, -42, -25])
    feature = ax.add_feature(Coast, edgecolor='black',facecolor='gray')
    im = ax.contourf(input.lon_rho,input.lat_rho,input,levels,**pcol_kwargs)
    gl = ax.gridlines(draw_labels=True,
                     color='black', alpha=0.2, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False
    gl.bottom_labels = False
    cax = inset_axes(ax,
                 width="5%",  # width = 10% of parent_bbox width
                 height="50%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(.07,.39, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
    fig.colorbar(im, cax=cax)
    ax.text(0.01, 0.99, labelText, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    ax.set_title('')
    return ax


def addSubplot_spatialMap_pcolor(input,gs,nrow,ncol, labelText=None, pcol_kwargs={}, cont_kwargs={}, kde_kwargs={}):
    ax = fig.add_subplot(gs[nrow,ncol], projection=ccrs.PlateCarree())
    ax.set_extent([148, 161, -42, -25])
    feature = ax.add_feature(Coast, edgecolor='black',facecolor='gray')
    im = ax.pcolormesh(input.lon_rho,input.lat_rho,input,**pcol_kwargs)
    gl = ax.gridlines(draw_labels=True,
                     color='black', alpha=0.2, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False
    gl.bottom_labels = False
    cax = inset_axes(ax,
                 width="5%",  # width = 10% of parent_bbox width
                 height="50%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(.07,.39, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0,
                 )
    fig.colorbar(im, cax=cax)
    ax.text(0.01, 0.99, labelText, transform=ax.transAxes,fontsize=22, fontweight='bold', va='top')
    ax.set_title('')
    return ax


# gs to make a 4 row, 7 col plot
gs = gridspec.GridSpec(nrows=5,ncols=5)
plt.cla()
plt.clf()
fig = plt.figure(figsize=[12,17.5])
ax = None


# # add plots
# # top row

###
ax7 = addSubplot_spatialMap_pcolor(truth.zeta.mean(dim='ocean_time')  ,gs,0,0,'a',pcol_kwargs={'cmap':'cmo.curl','vmin':-.3,'vmax':0.3})
ax7.text(0.5, 1.1, 'SSH',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((truthSurf.mke),truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST.zeta-truth.zeta)**2).mean(dim='ocean_time'))*(1/2)   ,gs,1,0,'b',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_N.zeta-truth.zeta)**2).mean(dim='ocean_time'))*(1/2)   ,gs,2,0,'c',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-N',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_NSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_S.zeta-truth.zeta)**2).mean(dim='ocean_time'))*(1/2)   ,gs,3,0,'d',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_SSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax8 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT.zeta-truth.zeta)**2).mean(dim='ocean_time'))*(1/2)  ,gs,4,0,'e',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax8.text(0.5, 1.1, 'SSH+SST+XBT-N+S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax8.transAxes)
# ax8.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax8.transAxes)

###

ax7 = addSubplot_spatialMap_pcolor(truth_mke_0  ,gs,0,1,'a',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
ax7.text(0.5, 1.1, 'MKE',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((truthSurf.mke),truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( OSSE_SSHSST_mke_0   ,gs,1,1,'b',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_N_mke_0   ,gs,2,1,'c',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-N',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_NSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_S_mke_0   ,gs,3,1,'d',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_SSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax8 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_mke_0  ,gs,4,1,'e',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax8.text(0.5, 1.1, 'SSH+SST+XBT-N+S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax8.transAxes)
# ax8.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax8.transAxes)

# (((OSSE_SSHSST_mke_500-truth_mke_500)**2).mean(dim='ocean_time'))**(1/2)/truth_mke_500.std(dim='ocean_time)

###
ax7 = addSubplot_spatialMap_pcolor(truth_mke_500  ,gs,0,2,'a',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
ax7.text(0.5, 1.1, 'MKE (500m)',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((truthSurf.mke),truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( OSSE_SSHSST_mke_500   ,gs,1,2,'b',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_N_mke_500   ,gs,2,2,'c',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-N',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_NSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_S_mke_500   ,gs,3,2,'d',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_SSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax8 = addSubplot_spatialMap_pcolor(OSSE_SSHSST_XBT_mke_500  ,gs,4,2,'e',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax8.text(0.5, 1.1, 'SSH+SST+XBT-N+S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax8.transAxes)
# ax8.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax8.transAxes)

###
ax7 = addSubplot_spatialMap_pcolor(truth_eke_0.mean(dim='ocean_time')  ,gs,0,3,'a',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
ax7.text(0.5, 1.1, 'EKE',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((truthSurf.mke),truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,1,3,'b',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_N_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,2,3,'c',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-N',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_NSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_S_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,3,3,'d',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_SSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax8 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_eke_0-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)  ,gs,4,3,'e',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax8.text(0.5, 1.1, 'SSH+SST+XBT-N+S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax8.transAxes)
# ax8.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax8.transAxes)


###
ax7 = addSubplot_spatialMap_pcolor(truth_eke_500.mean(dim='ocean_time')  ,gs,0,4,'a',pcol_kwargs={'cmap':'cmo.amp','vmin':0,'vmax':0.03})
ax7.text(0.5, 1.1, 'EKE (500m)',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((truthSurf.mke),truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_eke_500-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,1,4,'b',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_N_eke_500-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,2,4,'c',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-N',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_NSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax7 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_S_eke_500-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)   ,gs,3,4,'d',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax7.text(0.5, 1.1, 'SSH+SST+XBT-S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax7.transAxes)
# ax7.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBT_SSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax7.transAxes)

ax8 = addSubplot_spatialMap_pcolor( (((OSSE_SSHSST_XBT_eke_500-truth_eke_0)**2).mean(dim='ocean_time'))**(1/2)  ,gs,4,4,'e',pcol_kwargs={})#'cmap':'cmo.amp','vmin':0,'vmax':0.03})
# ax8.text(0.5, 1.1, 'SSH+SST+XBT-N+S',fontsize=14, fontweight='bold', va='top', ha='center', transform=ax8.transAxes)
# ax8.text(.9, .05, str(np.round(calc_areaAverageROMS((OSSE_SSHSST_XBTSurf.mke-truthSurf.mke)/truthSurf.mke*100,truth).values,4))+' $m^2/s^2$',fontsize=10, va='center', ha='center', transform=ax8.transAxes)

