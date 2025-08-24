%FIG_BEAMFORMING_EXAMPLE generates Figures 5.6-5.8, 
% showing plane-wave decomposition, illustrating stages 
% of spherical array processing.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

j=sqrt(-1);
N=6; 
INFTY=10; % simulates N=infinity

% Spatial sampling
[a,th,ph]=uniform_sampling(N);
Q=length(a);

% sphere
sphere=1; % rigid

% plane waves
amp_s=[1 0.7*exp(j*pi/3) 0.4*j]; % wave amplitude
th_s=[0.5*pi 0.65*pi 0.25*pi ];  % direction of arrival
ph_s=[0.25*pi 0.5*pi 1.5*pi];    % direction of arrival


for kr=[6,10],

% compute pressure at the microphones
Yinf=sh2(INFTY,th,ph); 
Ys=conj(sh2(INFTY,th_s,ph_s)).';
Binf=BnMat(INFTY,kr,kr,sphere);
Pnm=amp_s*Ys*diag(Binf);
P=Pnm*Yinf;

% compute PWD from microphones
Y=sh2(N,th,ph); 
B=BnMat(N,kr,kr,sphere);
PPnm=(4*pi/Q)*P*Y';
Wnm=PPnm./B;
Wnm=Wnm.';

% Compute PWD directly from sound field
Wnm0=amp_s*Ys(:,1:(N+1)^2);
Wnm0=Wnm0.';

% plot W
figure;
plot_contour(Wnm,1);
xlabel('$\phi_l\,$ (degrees)','interp','Latex');
ylabel('$\theta_l\,$ (degrees)','interp','Latex');
hold on;
plot((180/pi)*ph_s,(180/pi)*th_s,'w+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

end;


% plot W0
figure;
plot_contour(Wnm0,1);
xlabel('$\phi_l\,$ (degrees)','interp','Latex');
ylabel('$\theta_l\,$ (degrees)','interp','Latex');
hold on;
plot((180/pi)*ph_s,(180/pi)*th_s,'w+','MarkerSize',14,'LineWidth',3.0);
set(gca,'FontSize',AxisFontSize);

% Print figures in png
% figure(1); print -dpng ../../../figures/chapter5/fig_beamforming_example1.png % kr=6
% figure(2); print -dpng ../../../figures/chapter5/fig_beamforming_example3.png % kr=10;
% figure(3); print -dpng ../../../figures/chapter5/fig_beamforming_example2.png % Direct

% Print figure in eps
% figure(1); print -depsc -loose ../../../figures/chapter5/fig_beamforming_example1.eps % kr=6
% figure(2); print -depsc -loose ../../../figures/chapter5/fig_beamforming_example3.eps % kr=10;
% figure(3); print -depsc -loose ../../../figures/chapter5/fig_beamforming_example2.eps % Direct
