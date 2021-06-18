modPath = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/truth/eac_mod_08005.nc'

obsprov = ncread(modPath,'obs_provenance');
obsvalue = ncread(modPath,'obs_value');
obstime = ncread(modPath,'obs_time');
NLmodelvalue = ncread(modPath,'NLmodel_value');


figure, plot(obstime,obsvalue,'.'), hold on, plot(obstime,NLmodelvalue,'.')
xlim([8005 8006])
legend('obs value','NLmodel_value')

