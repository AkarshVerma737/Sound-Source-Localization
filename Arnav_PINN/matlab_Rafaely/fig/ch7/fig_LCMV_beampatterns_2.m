%FIG_LCMV_BEAMPATTERNS_2 generates Figures 7.8-7.11, 
% showing beam patterns of LCMV beamformers 
% with added derivative constraints. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

% array parameter setting
N=4;
Q=36; 
kr=N;
bn=BnMat(N,kr,kr,1); % rigid sphere
B=diag(bn);

% desired signal
th0=60*pi/180; ph0=36*pi/180; % look direction
Y0=sh2(N,th0,ph0);
sigma2s=1;
vnm0=B*conj(Y0);

% disturbance / null
th1=60*pi/180; ph1=90*pi/180; % disturbance direction
Y1=sh2(N,th1,ph1);
vnm1=B*conj(Y1);

% Sxx and sensor noise
sigma2n=0.1; 
Sxx = sigma2s*vnm0*vnm0' + sigma2n*(4*pi/Q)*eye((N+1)^2);


% LCMV with one null
Vnm=[vnm0,vnm1];
c=[1;0]
wnm_conj = c' * inv(Vnm'*inv(Sxx)*Vnm) * Vnm' * inv(Sxx);
wnm=wnm_conj';
wnmB=conj(B)*wnm; 
% Note: this leads to wnmB*Ynm=conj(y).
% Then plot abs(conj(y))=abs(y).
% This way plot_contour and plot_balloon can be used directly

figure;
plot_contour(wnmB);
hold on; 
plot((180/pi)*ph0,(180/pi)*th0,'w+','MarkerSize',14,'LineWidth',3.0);
plot((180/pi)*ph1,(180/pi)*th1,'k+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

figure;
plot_balloon(wnmB,[40,30],0.1);
axis on; axis tight; axis equal;
xlabel('$x$','Interp','Latex');
ylabel('$y$','Interp','Latex');
zlabel('$z$','Interp','Latex');
set(gca,'FontSize',AxisFontSize);

% LCMV with null derivative constraints over phi
vnm_ph1=derivative_ph(vnm1);
Vnm=[vnm0,vnm1,vnm_ph1];
c=[1;0;0]
wnm_conj = c' * inv(Vnm'*inv(Sxx)*Vnm) * Vnm' * inv(Sxx);
wnm=wnm_conj';
wnmB=conj(B)*wnm; 
% Note: this computes conj(y) and not y, see above

figure;
plot_contour(wnmB);
hold on; 
plot((180/pi)*ph0,(180/pi)*th0,'w+','MarkerSize',14,'LineWidth',3.0);
plot((180/pi)*ph1,(180/pi)*th1,'k+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

figure;
plot_balloon(wnmB,[40,30],0.1);
axis on; axis tight; axis equal;
xlabel('$x$','Interp','Latex');
ylabel('$y$','Interp','Latex');
zlabel('$z$','Interp','Latex');
set(gca,'FontSize',AxisFontSize);


% LCMV with null derivative constraints over phi and theta
vnm_th1=derivative_th(vnm1,th1,ph1);
Vnm=[vnm0,vnm1,vnm_ph1,vnm_th1];
c=[1;0;0;0]
wnm_conj = c' * inv(Vnm'*inv(Sxx)*Vnm) * Vnm' * inv(Sxx);
wnm=wnm_conj';
wnmB=conj(B)*wnm; 
% Note: this computes conj(y) and noy y, see above

figure;
plot_contour(wnmB);
hold on; 
plot((180/pi)*ph0,(180/pi)*th0,'w+','MarkerSize',14,'LineWidth',3.0);
plot((180/pi)*ph1,(180/pi)*th1,'k+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

figure;
plot_balloon(wnmB,[40,30],0.1);
axis on; axis tight; axis equal;
xlabel('$x$','Interp','Latex');
ylabel('$y$','Interp','Latex');
zlabel('$z$','Interp','Latex');
set(gca,'FontSize',AxisFontSize);


% LCMV with look-direction derivative constraints
vnm_ph0=derivative_ph(vnm0);
vnm_th0=derivative_th(vnm0,th0,ph0);
Vnm=[vnm0,vnm1,vnm_ph1,vnm_th1,vnm_ph0,vnm_th0];
c=[1;0;0;0;0;0]
wnm_conj = c' * inv(Vnm'*inv(Sxx)*Vnm) * Vnm' * inv(Sxx);
wnm=wnm_conj';
wnmB=conj(B)*wnm; 
% Note: this computes conj(y) and not, see above

figure;
plot_contour(wnmB);
hold on; 
plot((180/pi)*ph0,(180/pi)*th0,'w+','MarkerSize',14,'LineWidth',3.0);
plot((180/pi)*ph1,(180/pi)*th1,'k+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

figure;
plot_balloon(wnmB,[40,30],0.1);
axis on; axis tight; axis equal;
xlabel('$x$','Interp','Latex');
ylabel('$y$','Interp','Latex');
zlabel('$z$','Interp','Latex');
set(gca,'FontSize',AxisFontSize);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter7/fig_lcmv2_contour1.png
% figure(2); print -dpng ../../../figures/chapter7/fig_lcmv2_balloon1.png
% figure(3); print -dpng ../../../figures/chapter7/fig_lcmv2_contour2.png
% figure(4); print -dpng ../../../figures/chapter7/fig_lcmv2_balloon2.png
% figure(5); print -dpng ../../../figures/chapter7/fig_lcmv2_contour3.png
% figure(6); print -dpng ../../../figures/chapter7/fig_lcmv2_balloon3.png
% figure(7); print -dpng ../../../figures/chapter7/fig_lcmv2_contour4.png
% figure(8); print -dpng ../../../figures/chapter7/fig_lcmv2_balloon4.png

% Print figures in eps
% figure(1); print -depsc -loose ../../../figures/chapter7/fig_lcmv2_contour1.eps
% figure(2); print -depsc -loose -r300 ../../../figures/chapter7/fig_lcmv2_balloon1.eps
% figure(3); print -depsc -loose ../../../figures/chapter7/fig_lcmv2_contour2.eps
% figure(4); print -depsc -loose -r300 ../../../figures/chapter7/fig_lcmv2_balloon2.eps
% figure(5); print -depsc -loose ../../../figures/chapter7/fig_lcmv2_contour3.eps
% figure(6); print -depsc -loose -r300 ../../../figures/chapter7/fig_lcmv2_balloon3.eps
% figure(7); print -depsc -loose ../../../figures/chapter7/fig_lcmv2_contour4.eps
% figure(8); print -depsc -loose -r300 ../../../figures/chapter7/fig_lcmv2_balloon4.eps
