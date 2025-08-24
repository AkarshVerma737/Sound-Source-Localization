%FIG_CARDIOID_DIRECTIVITY generates Figure 4.4, 
% illustrating Cardioid directivity.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

TH=linspace(0,2*pi,500);
y=0.5*(1+cos(TH));

h=polarplot(TH,y);
set(h,'LineWidth',3,'Color',[0 0 0.5]);
set(gca,'FontSize',16);

% Print figure in png
% print -dpng ../../../figures/chapter4/fig_cardioid.png

% Print figure in eps
% print -depsc -loose ../../../figures/chapter4/fig_cardioid.eps
