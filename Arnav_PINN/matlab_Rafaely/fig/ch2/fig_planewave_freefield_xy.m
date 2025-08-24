%FIG_PLANEWAVE_FREEFIELD_XY generates Figure 2.5, 
% illustrating a spherical-harmonics composition of a plane wave
% in free-field, over the xy plane. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

% orders
N1=32; 
N2=16;
N3=8;

% unit circle
z=linspace(0,2*pi,300);

% Sampling grid
x=linspace(-20,20,100); 
y=linspace(-20,20,100); 
[X,Y]=meshgrid(x,y);
X1=reshape(X,length(x)*length(y),1);
Y1=reshape(Y,length(x)*length(y),1);
ph=atan2(Y1,X1);
th=pi/2*ones(size(ph));
r=sqrt(X1.^2+Y1.^2);

% Wave arrival direction
thk=pi/2;
phk=pi/9;

% Compute matrices
k=1; % wave number
B=BnMat(N1,k*r',k*r',0); % Radial function matrix
Yk=conj(sh2(N1,thk,phk)); % Spherical harmonics vector
Y=(sh2(N1,th,ph)).'; % Spherical harmonics matrix

% Compute p = SUM_n SUM_m bn(kr) Ynm(thk,phk)^* Ynm(th,ph)
p1=B.*Y*Yk;
p2=B(:,1:(N2+1)^2).*Y(:,1:(N2+1)^2)*Yk(1:(N2+1)^2);
p3=B(:,1:(N3+1)^2).*Y(:,1:(N3+1)^2)*Yk(1:(N3+1)^2);

p1a=reshape(p1,length(x),length(y));
p2a=reshape(p2,length(x),length(y));
p3a=reshape(p3,length(x),length(y));

figure(1);
[c,h]=contourf(x,y,real(p1a),'LineStyle','none'); 
axis square
colormap(jet);
colorbar;
set(gca,'FontSize',AxisFontSize);
set(colorbar,'FontSize',AxisFontSize);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N1)),'Interp','Latex');

figure(2);
[c,h]=contourf(x,y,real(p2a),'LineStyle','none'); 
axis square
colormap(jet);
colorbar;
set(gca,'FontSize',AxisFontSize);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N2)),'Interp','Latex');
hold on;
plot(N2*cos(z),N2*sin(z),'w-','LineWidth',2);

figure(3);
[c,h]=contourf(x,y,real(p3a),'LineStyle','none'); 
axis square
colormap(jet);
colorbar;
set(gca,'FontSize',AxisFontSize);
xlabel('$x\,$ (m)','Interp','Latex');
ylabel('$y\,$ (m)','Interp','Latex');
title(strcat('$N=\,\,$',num2str(N3)),'Interp','Latex');
hold on;
plot(N3*cos(z),N3*sin(z),'w-','LineWidth',2);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter2/fig_pressureNxy1.png
% figure(2); print -dpng ../../../figures/chapter2/fig_pressureNxy2.png
% figure(3); print -dpng ../../../figures/chapter2/fig_pressureNxy3.png

% Print figures in eps
% figure(1); print -depsc -loose ../../../figures/chapter2/fig_pressureNxy1.eps
% figure(2); print -depsc -loose ../../../figures/chapter2/fig_pressureNxy2.eps
% figure(3); print -depsc -loose ../../../figures/chapter2/fig_pressureNxy3.eps
