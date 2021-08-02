modPath = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/truth/eac_mod_08005.nc'
grid = grid_read('/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/EACouter_varres_grd_mergedBLbry_uhroms.nc'); % load grid
obsPath = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/verification/eac_obs_8005.nc'

%% Mod file plots
mod.obsprov = ncread(modPath,'obs_provenance');
mod.obsvalue = ncread(modPath,'obs_value');
mod.obstime = ncread(modPath,'obs_time');
mod.NLmodelfinal = ncread(modPath,'NLmodel_final');

ind = (mod.obsprov==405);
figure, plot(mod.obstime(ind),mod.obsvalue(ind),'.'), hold on, plot(mod.obstime(ind),mod.NLmodelvalue(ind),'.')
xlim([8005 8006])
legend('obs value','NLmodel_value')


ind = (mod.obsprov==340);
figure, plot(mod.obstime(ind),mod.obsvalue(ind),'.'), hold on, plot(mod.obstime(ind),mod.NLmodelvalue(ind),'.')
xlim([8005 8006])
legend('obs value','NLmodel_value')


%% obs file plots
obs.obsprov = ncread(obsPath,'obs_provenance');
obs.obsvalue = ncread(obsPath,'obs_value');
obs.obstime = ncread(obsPath,'obs_time');
obs.Xgrid = ncread(obsPath,'obs_Xgrid');
obs.Ygrid = ncread(obsPath,'obs_Ygrid');
obs.lon = ncread(obsPath,'obs_lon');
obs.lat = ncread(obsPath,'obs_lat');
obs.Zgrid = ncread(obsPath,'obs_Zgrid');

ind = (obs.obsprov==340);
ind2 = (obs.obstime>8008 & obs.obstime<8009);


xin = obs.Xgrid(ind & ind2);
yin = obs.Ygrid(ind & ind2);
for ii = 1:numel(xin)
lonp(ii) = grid.lonr(floor(yin(ii))+1,floor(xin(ii))+1);
latp(ii) = grid.latr(floor(yin(ii))+1,floor(xin(ii))+1);
end

figure


figure,
scatter(lonp,  latp,'o')
hold on
scatter(obs.lon(ind & ind2),obs.lat(ind & ind2),5,'.')

