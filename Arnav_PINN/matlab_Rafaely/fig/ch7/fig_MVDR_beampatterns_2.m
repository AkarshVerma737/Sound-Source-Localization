%FIG_MVDR_BEAMPATTERNS_2 generates Figures 7.3, 7.4, 
% showing beam pattern of MVDR beamformenrs 
% with added correlated disturbance.
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

sigma0=1; % desired signal 
th0=60*pi/180; ph0=36*pi/180; % look direction
Y0=sh2(N,th0,ph0);
vnm0=B*conj(Y0);

A=0.8*exp(-j*pi/3); % S1=A*S0, disturbance is a reflection of direct sound
sigma1=abs(A^2)*sigma0; % 
th1=60*pi/180; ph1=320*pi/180; % disturbance direction
Y1=sh2(N,th1,ph1);
vnm1=B*conj(Y1);

sigman=0.1; % sensor noise

% MVDR with correlated disturbance Snn minimization
Snn = sigma1*vnm1*vnm1' + sigman*(4*pi/Q)*eye((N+1)^2);
wnm = (vnm0'*inv(Snn)/(vnm0'*inv(Snn)*vnm0))';
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


% MVDR with correlated disturbance Sxx minimization
Sxx = sigma0*vnm0*vnm0' + sigma1*vnm1*vnm1' + sigma0*vnm0*(A*vnm1)' + sigma0*(A*vnm1)*vnm0' + sigman*(4*pi/Q)*eye((N+1)^2);
wnm = (vnm0'*inv(Sxx)/(vnm0'*inv(Sxx)*vnm0))';
wnmB=conj(B)*wnm; 
% Note: this computes conj(y) and not y, see above

figure;
plot_contour(wnmB);
hold on; 
plot((180/pi)*ph0,(180/pi)*th0,'w+','MarkerSize',14,'LineWidth',3.0);
plot((180/pi)*ph1,(180/pi)*th1,'w+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

figure;
plot_balloon(wnmB,[40,70],0.1); 
axis on; axis tight; axis equal;
xlabel('$x$','Interp','Latex');
ylabel('$y$','Interp','Latex');
zlabel('$z$','Interp','Latex');
set(gca,'FontSize',AxisFontSize);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter7/fig_mvdr2_contour1.png
% figure(2); print -dpng ../../../figures/chapter7/fig_mvdr2_balloon1.png
% figure(3); print -dpng ../../../figures/chapter7/fig_mvdr2_contour2.png
% figure(4); print -dpng ../../../figures/chapter7/fig_mvdr2_balloon2.png

% Print figures in eps
% figure(1); print -depsc -loose ../../../figures/chapter7/fig_mvdr2_contour1.eps
% figure(2); print -depsc -loose -r300 ../../../figures/chapter7/fig_mvdr2_balloon1.eps
% figure(3); print -depsc -loose ../../../figures/chapter7/fig_mvdr2_contour2.eps
% figure(4); print -depsc -loose -r300 ../../../figures/chapter7/fig_mvdr2_balloon2.eps
