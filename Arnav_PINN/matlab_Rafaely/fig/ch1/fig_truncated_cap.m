%FIG_TRUNCATED_CAP generates Figure 1.14, showing balloon plots of spherical 
% cap functions, illustrating the Gibbs phenomenon.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=14;

% Spherical cap
N=40;
alpha=(pi/180)*30;
fnm=zeros((N+1)^2,1);
fnm(1)=sqrt(pi)*(1-cos(alpha));
fnm(1)=fnm(1)+10; % add DC
for n=1:N,
    fn0=sqrt(pi/(2*n+1))*(legendreP(n-1,cos(alpha))-legendreP(n+1,cos(alpha)));
    fnm(n^2+1:n^2+2*n+1)=[zeros(n,1);fn0;zeros(n,1)];
end;

figure(1);
subplot(221);
N=4;
plot_balloon(fnm(1:(N+1)^2));
title('$N=4$','Interp','Latex','FontSize',AxisFontSize);
axis tight;
camzoom(1.2);

subplot(222);
N=10;
plot_balloon(fnm(1:(N+1)^2));
title('$N=10$','Interp','Latex','FontSize',AxisFontSize);
axis tight;
camzoom(1.2);

subplot(223);
N=20;
plot_balloon(fnm(1:(N+1)^2));
title('$N=20$','Interp','Latex','FontSize',AxisFontSize);
axis tight;
camzoom(1.2);

subplot(224);
N=40;
plot_balloon(fnm(1:(N+1)^2));
title('$N=40$','Interp','Latex','FontSize',AxisFontSize);
axis tight;
camzoom(1.2);

% Print figures in png
% print -dpng ../../../figures/chapter1/fig_Gibbs.png

% Print figures in eps
% print -depsc -loose -r300 ../../../figures/chapter1/fig_Gibbs.eps


