clear all;close all

addpath(genpath('../src/ext/matlab/'))

truth='../data/raw/outer_avg_08005.nc';
fore='../data/raw/roms_fwd_outer0_08005.nc';
anal='../data/raw/roms_fwd_outer1_08005.nc';
g=grid_read( '../data/raw/EACouter_varres_grd_mergedBLbry_uhroms.nc' );


timet=nc_varget(truth,'ocean_time')./86400+datenum(1990,1,1);
time=nc_varget(fore,'ocean_time')./86400+datenum(1990,1,1);



tr=nc_varget(truth,'zeta');
itruth=find(timet>=time(1)&timet<=time(end));
tr=tr(itruth,:,:);timet=timet(itruth);

it=[7:6:31];
analzeta=nc_varget(anal,'zeta');analzeta=analzeta(it,:,:);
forezeta=nc_varget(fore,'zeta');forezeta=forezeta(it,:,:);
time=time(it);

%%
 figure;
 set(gcf,'position',[1           7        1164         698],'color', 'w')
 subplot1(3,5,'Gap',[0.03 0.03],'YTickL','Margin','XTickL','Margin') 
 
 % truth
 for i=1:5
 subplot1(i)
 flat_pcolor(g.lonr,g.latr,squeeze(tr(i,:,:)));caxis([-0.8,0.8]);
 title(datestr(timet(i)));colormap(jet)
 end
 
 % forecast
  for i=1:5
 subplot1(5+i)
 flat_pcolor(g.lonr,g.latr,squeeze(forezeta(i,:,:)));caxis([-0.8,0.8]);
 title(datestr(time(i)));colormap(jet)
  end
 
  % analysis
   for i=1:5
 subplot1(10+i)
 flat_pcolor(g.lonr,g.latr,squeeze(analzeta(i,:,:)));caxis([-0.8,0.8]);
 title(datestr(time(i)));colormap(jet)
   end

 %%
 
  figure;
 set(gcf,'position',[108         546        1517         361],'color', 'w')
 subplot1(2,5,'Gap',[0.03 0.03],'YTickL','Margin','XTickL','Margin') 
 

 
 % forecast
  for i=1:5
 subplot1(i)
 flat_pcolor(g.lonr,g.latr,squeeze(forezeta(i,:,:))-squeeze(tr(i,:,:)));caxis([-0.4,0.4]);
 title(datestr(time(i)));cmocean('balance','pivot',0)
  end
 
  % analysis
   for i=1:5
 subplot1(5+i)
 flat_pcolor(g.lonr,g.latr,squeeze(analzeta(i,:,:))-squeeze(tr(i,:,:)));caxis([-0.4,0.4]);
 title(datestr(time(i)));cmocean('balance','pivot',0)
   end
   
   %% look at analysis more often
   

time=nc_varget(fore,'ocean_time')./86400+datenum(1990,1,1);


it=[1:2:31];
analzeta=nc_varget(anal,'zeta');analzeta=analzeta(it,:,:);
forezeta=nc_varget(fore,'zeta');forezeta=forezeta(it,:,:);
time=time(it);

 figure;
 set(gcf,'position',[-1633           7        1164         698],'color', 'w')
 subplot1(2,length(it),'Gap',[0.003 0.003],'YTickL','Margin','XTickL','Margin') 

 
 % forecast
  for i=1:length(it)
 subplot1(i)
 flat_pcolor(g.lonr,g.latr,squeeze(forezeta(i,:,:)));;caxis([-0.8,0.8]);
 %title(datestr(time(i)));colormap(jet)
  end
 
  % analysis
   for i=1:length(it)
 subplot1(length(it)+i)
 flat_pcolor(g.lonr,g.latr,squeeze(analzeta(i,:,:)));;caxis([-0.8,0.8]);
 %title(datestr(time(i)));colormap(jet)
   end
   
   colormap(jet)
