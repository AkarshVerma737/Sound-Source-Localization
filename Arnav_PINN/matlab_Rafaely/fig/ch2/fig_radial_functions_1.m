%FIG_RADIAL_FUNCTIONS_1 generates Figures 2.1-2.4, 2.9, 
% illustrating the spherical Bessel and Hankel functions, 
% and the rigid-sphere radial function. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;
N=6;

% besseljs zeros to enhance notches on graph
xz=[3.1415 4.493 5.764 6.283 6.988 7.725 8.182 9.095 9.356 9.424];

x=linspace(0.01,10,400);
x1=linspace(0.001,1,400);

clear J H B J1 H1 B1;

for n=0:N,    
    J(n+1,:)=besseljs(n,sort([x xz]));
    H(n+1,:)=besselhs(n,x);
    B(n+1,:)=Bn(n,x,x,1);
    J1(n+1,:)=besseljs(n,x1);
    H1(n+1,:)=besselhs(n,x1);
    B1(n+1,:)=Bn(n,x1,x1,1);
end;

figure(1);
h=plot(sort([x xz]),20*log10(abs(J)),'-','LineWidth',1.5,'Color',[0 0 0.5]);
text(0.2,25-22,'0','FontSize',AxisFontSize);
text(0.6,12.2-22,'1','FontSize',AxisFontSize);
text(0.9,3.1-22,'2','FontSize',AxisFontSize);
text(1.2,-7.0-22,'3','FontSize',AxisFontSize);
text(1.6,-15.7-22,'4','FontSize',AxisFontSize);
text(1.9,-24.5-22,'5','FontSize',AxisFontSize);
text(2.2,-33.0-22,'6','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
xlabel('$x$','Interp','Latex');
ylabel('Magnitude (dB)','Interp','Latex');
axis([min(x) max(x) -80 10]);

figure(2);
plot(x,20*log10(abs(H)),'-','LineWidth',1.5,'Color',[0 0 0.5]);
text(0.7,-1,'0','FontSize',AxisFontSize);
text(1,4,'1','FontSize',AxisFontSize);
text(1.3,8,'2','FontSize',AxisFontSize);
text(1.5,13,'3','FontSize',AxisFontSize);
text(1.8,19,'4','FontSize',AxisFontSize);
text(2.1,25,'5','FontSize',AxisFontSize);
text(2.5,30,'6','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
xlabel('$x$','Interp','Latex');
ylabel('Magnitude (dB)','Interp','Latex');
axis([min(x) max(x) -20 40]);

figure(3);
plot(x,20*log10(abs(B/(4*pi))),'-','LineWidth',1.5,'Color',[0 0 0.5]);
text(0.2,25-22,'0','FontSize',AxisFontSize);
text(0.6-0.2,12.2-22+2,'1','FontSize',AxisFontSize);
text(0.9-0.2,3.1-22+2,'2','FontSize',AxisFontSize);
text(1.2-0.2,-7.0-22+2,'3','FontSize',AxisFontSize);
text(1.6-0.2,-15.7-22+2,'4','FontSize',AxisFontSize);
text(1.9-0.2,-24.5-22+2,'5','FontSize',AxisFontSize);
text(2.2-0.2,-33.0-22+2,'6','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
xlabel('$kr$','Interp','Latex');
ylabel('Magnitude (dB)','Interp','Latex');
axis([min(x) max(x) -80 10]);


figure(4);
semilogx(x1,20*log10(abs(J1)),'-','LineWidth',1.5,'Color',[0 0 0.5]);
text(0.002,7,'0','FontSize',AxisFontSize);
text(0.01,-40,'1','FontSize',AxisFontSize);
text(0.03,-75,'2','FontSize',AxisFontSize);
text(0.07,-100,'3','FontSize',AxisFontSize);
text(0.12,-120,'4','FontSize',AxisFontSize);
text(0.18,-140,'5','FontSize',AxisFontSize);
text(0.25,-160,'6','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
xlabel('$x$','Interp','Latex');
ylabel('Magnitude (dB)','Interp','Latex');
axis([min(x1) max(x1) -200 20]);

figure(5);
semilogx(x1,20*log10(abs(H1)),'-','LineWidth',1.5,'Color',[0 0 0.5]);
text(0.002,70,'0','FontSize',AxisFontSize);
text(0.0025,120,'1','FontSize',AxisFontSize);
text(0.003,180,'2','FontSize',AxisFontSize);
text(0.004,235,'3','FontSize',AxisFontSize);
text(0.005,290,'4','FontSize',AxisFontSize);
text(0.007,335,'5','FontSize',AxisFontSize);
text(0.01,380,'6','FontSize',AxisFontSize);
set(gca,'FontSize',AxisFontSize);
xlabel('$x$','Interp','Latex');
ylabel('Magnitude (dB)','Interp','Latex');
axis([min(x1) max(x1) 0 500]);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter2/fig_rad_fun_jn.png
% figure(2); print -dpng ../../../figures/chapter2/fig_rad_fun_hn.png
% figure(3); print -dpng ../../../figures/chapter2/fig_rad_fun_bn.png
% figure(4); print -dpng ../../../figures/chapter2/fig_rad_fun_jn1.png
% figure(5); print -dpng ../../../figures/chapter2/fig_rad_fun_hn1.png

% Print figures in eps
% figure(1); print -depsc -loose ../../../figures/chapter2/fig_rad_fun_jn.eps
% figure(2); print -depsc -loose ../../../figures/chapter2/fig_rad_fun_hn.eps
% figure(3); print -depsc -loose ../../../figures/chapter2/fig_rad_fun_bn.eps
% figure(4); print -depsc -loose ../../../figures/chapter2/fig_rad_fun_jn1.eps
% figure(5); print -depsc -loose ../../../figures/chapter2/fig_rad_fun_hn1.eps
