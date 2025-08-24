%FIG_PLANEWAVE_FREEFIELD_SPHERE generates Figure 2.7, 
% illustrating a spherical-harmonics composition of a plane wave
% in free-field, over the sphere. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

% orders
N1=20;
N2=10;
N3=5;

% Wave arrival direction
thk=pi/4;
phk=-pi/4;

kr=10;

B=BnMat(N1,kr,kr,0);
Yk=conj(sh2(N1,thk,phk));

pnm1=diag(B)*Yk;
pnm2=diag(B(:,1:(N2+1)^2))*Yk(1:(N2+1)^2);
pnm3=diag(B(:,1:(N3+1)^2))*Yk(1:(N3+1)^2);


figure;
plot_sphere(pnm1);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
zlabel('$z\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N1)),'Interp','Latex');
set(gca,'FontSize',AxisFontSize);

figure;
plot_sphere(pnm2);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
zlabel('$z\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N2)),'Interp','Latex');
set(gca,'FontSize',AxisFontSize);

figure;
plot_sphere(pnm3);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
zlabel('$z\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N3)),'Interp','Latex');
set(gca,'FontSize',AxisFontSize);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter2/fig_pressureNr1.png
% figure(2); print -dpng ../../../figures/chapter2/fig_pressureNr2.png
% figure(3); print -dpng ../../../figures/chapter2/fig_pressureNr3.png

% Print figures in eps
% figure(1); print -depsc -loose -r300 ../../../figures/chapter2/fig_pressureNr1.eps
% figure(2); print -depsc -loose -r300 ../../../figures/chapter2/fig_pressureNr2.eps
% figure(3); print -depsc -loose -r300 ../../../figures/chapter2/fig_pressureNr3.eps
