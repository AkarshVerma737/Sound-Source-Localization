%FIG_CHEBYSHEV_POLYNOMIAL generates Figures 6.5, 6.6, 
% illustrating the Chebyshev polynomial.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

x=linspace(0,1.07,512);
y8=128*x.^8-256*x.^6+160*x.^4-32*x.^2+1;

M=8;
th0=pi/4;
x0=cos(pi/(2*M))/cos(th0/2);
R=cosh(M*acosh(x0));
th=linspace(-pi,pi,512);
z=x0*cos(th/2);
b8=(1/R)*(128*z.^8-256*z.^6+160*z.^4-32*z.^2+1);

% print values
x0,
R,

figure;
plot(x,y8,'-','LineWidth',2,'Color',[0 0 0.5]); hold on
plot(x,ones(size(x)),'k--','LineWidth',1);
plot(x,-ones(size(x)),'k--','LineWidth',1);
plot(x0,R,'ko','LineWidth',2);
text(x0-0.15,R,'$(x_0,R)$','FontSize',AxisFontSize,'interp','Latex');
axis([min(x),max(x)+0.1,-2,10]);
set(gca,'FontSize',AxisFontSize);
xlabel('$x$','interp','Latex');

figure;
plot(th*180/pi,b8,'-','LineWidth',2,'Color',[0 0 0.5]); hold on
plot(th*180/pi,(1/R)*ones(size(th)),'k--','LineWidth',1);
plot(th*180/pi,-(1/R)*ones(size(th)),'k--','LineWidth',1);
axis([min(th*180/pi),max(th*180/pi),-0.2,1.2]);
set(gca,'FontSize',AxisFontSize);
xlabel('$\theta$ (degrees)','interp','Latex');

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter6/fig_dolph_poly1.png
% figure(2); print -dpng ../../../figures/chapter6/fig_dolph_poly2.png

% Print figure in eps
% figure(1); print -depsc -loose ../../../figures/chapter6/fig_dolph_poly1.eps
% figure(2); print -depsc -loose ../../../figures/chapter6/fig_dolph_poly2.eps
