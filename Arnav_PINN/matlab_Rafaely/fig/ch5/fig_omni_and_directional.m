%FIG_OMINI_AND_DIRECTIONAL generates Figure 5.4, 
% illustrating omni-directional and directional directivities.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=14;

Theta=linspace(0,2*pi,2014);
z=cos(Theta);

y0=(1/(4*pi))*ones(size(Theta));
y1=(1/(4*pi))*(3*z+1);
y2=(1/(4*pi))*(3/2)*(5*z.^2+2*z-1);

figure;
subplot(121);
h2=polarplot(Theta,abs(y0),'-');
set(h2,'LineWidth',2,'Color',[0 0 0.5]);
title('Omni-directional','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
rlim([0,0.09]);

subplot(122);
h2=polarplot(Theta,abs(y2),'-');
set(h2,'LineWidth',2,'Color',[0 0 0.5]);
title('Directional','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
rlim([0,0.8]);

% Print figure in png
% print -dpng ../../../figures/chapter5/fig_directivity.png

% Print figure in eps
% print -depsc -loose ../../../figures/chapter5/fig_directivity.eps
