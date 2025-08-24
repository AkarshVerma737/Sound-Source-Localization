%FIG_PLANEWAVE_RIGID_SPHERE generates figure 2.12, 
% illustrating a spherical-harmonics composition 
% of a plane wave around a rigid sphere, presented
% over the sphere. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

N=32;
thk=pi/4;
phk=-pi/4;
kr=10;

B=BnMat(N,kr,kr,1);
Yk=conj(sh2(N,thk,phk));
pnm=diag(B)*Yk;

figure;
plot_sphere(pnm);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
zlabel('$z\,$ (m)','Interp','Latex');
set(gca,'FontSize',AxisFontSize);

% Print figure in png
% figure(1); print -dpng ../../../figures/chapter2/fig_pressureNr_rigid.png

% Print figure in eps
% figure(1); print -depsc -loose -r300 ../../../figures/chapter2/fig_pressureNr_rigid.eps

