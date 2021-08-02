addpath(genpath('../src/ext/matlab'))



modPath = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/dupedSST/eac_mod_08009.nc'
grid = grid_read('/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/EACouter_varres_grd_mergedBLbry_uhroms.nc'); % load grid
forePath = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/dupedSST/roms_fwd_outer1_08009.nc'

%% Mod file plots
mod.obsprov = ncread(modPath,'obs_provenance');
mod.obsvalue = ncread(modPath,'obs_value');
mod.obstime = ncread(modPath,'obs_time');
mod.obsXgrid = ncread(modPath,'obs_Xgrid');
mod.obsYgrid = ncread(modPath,'obs_Ygrid');
mod.NLmodel_value = ncread(modPath,'NLmodel_value');
mod.NLmodel_initial = ncread(modPath,'NLmodel_initial');
mod.NLmodel_final = ncread(modPath,'NLmodel_final');

ind = (mod.obsprov==405);
figure, plot(mod.obstime(ind),mod.obsvalue(ind),'.'), 
hold on, 
plot(mod.obstime(ind),mod.NLmodel_value(ind),'.')
plot(mod.obstime(ind),mod.NLmodel_initial(ind),'.')
plot(mod.obstime(ind),mod.NLmodel_final(ind),'.')
% xlim([8005 8006])
legend('obs value','NLmodel_value','NLmodel_initial','NLmodel_final')

ind = (mod.obsprov==405);
figure,subaxis(4,1,1)
 scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.obsvalue(ind),'filled'),caxis([-.2 1.1]),ntitle('obs value')
subaxis(4,1,2)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_value(ind),'filled'),caxis([-.2 1.1]),ntitle('NLmodel val')
subaxis(4,1,3)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_initial(ind),'filled'),caxis([-.2 1.1]),ntitle('NLmodel ini')
subaxis(4,1,4)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_final(ind),'filled'),caxis([-.2 1.1]),ntitle('NLmodel fin')
% xlim([8005 8006])
%legend('obs value','NLmodel_value','NLmodel_initial','NLmodel_final')

ind = (mod.obsprov==405);
figure,subaxis(4,1,1)
 scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.obsvalue(ind),'filled'),caxis([-.2 1.2]),ntitle('obs value')
subaxis(4,1,2)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_value(ind)-mod.obsvalue(ind),'filled'),caxis([-.2 .2]),ntitle('NLmodel val')
subaxis(4,1,3)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_initial(ind)-mod.obsvalue(ind),'filled'),caxis([-.2 .2]),ntitle('NLmodel ini')
subaxis(4,1,4)
scatter(mod.obstime(ind),mod.obsXgrid(ind),15,mod.NLmodel_final(ind)-mod.obsvalue(ind),'filled'),caxis([-.2 .2]),ntitle('NLmodel fin')
% xlim([8005 8006])
%legend('obs value','NLmodel_value','NLmodel_initial','NLmodel_final')


% 
% 
% ind = (mod.obsprov==340);
% figure, plot(mod.obstime(ind),mod.obsvalue(ind),'.'), 
% hold on, 
% plot(mod.obstime(ind),mod.NLmodel_value(ind),'.')
% plot(mod.obstime(ind),mod.NLmodel_initial(ind),'.')
% plot(mod.obstime(ind),mod.NLmodel_final(ind),'.')
% % xlim([8005 8006])
% legend('obs value','NLmodel_value','NLmodel_initial','NLmodel_final')





% %% out file plots
% osse.obsprov = ncread(obsPath,'obs_provenance');
% osse.obsvalue = ncread(obsPath,'obs_value');
% osse.obstime = ncread(obsPath,'obs_time');
% osse.Xgrid = ncread(obsPath,'obs_Xgrid');
% osse.Ygrid = ncread(obsPath,'obs_Ygrid');
% osse.lon = ncread(obsPath,'obs_lon');
% osse.lat = ncread(obsPath,'obs_lat');
% osse.Zgrid = ncread(obsPath,'obs_Zgrid');

% ind = (obs.obsprov==340);
% ind2 = (obs.obstime>8008 & obs.obstime<8009);


% xin = obs.Xgrid(ind & ind2);
% yin = obs.Ygrid(ind & ind2);
% for ii = 1:numel(xin)
% lonp(ii) = grid.lonr(floor(yin(ii))+1,floor(xin(ii))+1);
% latp(ii) = grid.latr(floor(yin(ii))+1,floor(xin(ii))+1);
% end

% figure


% figure,
% scatter(lonp,  latp,'o')
% hold on
% scatter(obs.lon(ind & ind2),obs.lat(ind & ind2),5,'.')



%% out file plots

anaPref = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/dupedSST/roms_fwd_outer1_0';
forePref = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/dupedSST/roms_fwd_outer0_0';
% anaPref = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/OSSE_SSHSST/roms_fwd_outer1_0';
% forePref = '/Users/dave/Documents/dave/Projects/OSSE-analysis/data/raw/OSSE_SSHSST/roms_fwd_outer0_0';
figure(801)
figure(802)


etaRangeMetric = [270, 310]+1
xiRangeMetric = [75, 175] +1
N=30;

files = [8005:4:8021];

for nn =1:numel(files)
fileNo = files(nn);
forePath=[forePref,num2str(fileNo),'.nc']
anaPath=[anaPref,num2str(fileNo),'.nc']

fore.zeta = 	ncread(forePath,'zeta');
fore.temp = 	ncread(forePath,'temp');
fore.time = 	ncread(forePath,'ocean_time')/86400;
ana.zeta = 	ncread(anaPath,'zeta');
ana.temp = 	ncread(anaPath,'temp');
ana.time = 	ncread(anaPath,'ocean_time')/86400;


fore.zeta_mean = squeeze(mean(mean(fore.zeta(xiRangeMetric(1):xiRangeMetric(2),etaRangeMetric(1):etaRangeMetric(2),:),1,'omitnan'),2,'omitnan'));
fore.temp_mean = squeeze(mean(mean(fore.temp(xiRangeMetric(1):xiRangeMetric(2),etaRangeMetric(1):etaRangeMetric(2),N,:),1,'omitnan'),2,'omitnan'));
ana.zeta_mean = squeeze(mean(mean(ana.zeta(xiRangeMetric(1):xiRangeMetric(2),etaRangeMetric(1):etaRangeMetric(2),:),1,'omitnan'),2,'omitnan'));
ana.temp_mean = squeeze(mean(mean(ana.temp(xiRangeMetric(1):xiRangeMetric(2),etaRangeMetric(1):etaRangeMetric(2),N,:),1,'omitnan'),2,'omitnan'));

figure(801), hold on,
plot(fore.time,fore.zeta_mean,'color',[0    0.4470    0.7410]), hold on
plot(ana.time,ana.zeta_mean,'color',[0.8500    0.3250    0.0980]), hold on

figure(802), hold on,
plot(fore.time,fore.temp_mean,'color',[0    0.4470    0.7410]), hold on
plot(ana.time,ana.temp_mean,'color',[0.8500    0.3250    0.0980]), hold on

end