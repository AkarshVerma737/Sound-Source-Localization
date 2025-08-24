%FIG_ALIASING_MATRIX generates Figure 3.6, illustrating the aliasing
% matrix from three sampling schemes.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

AxisFontSize=16;

N=3; % array order
NN=9; % aliasing analysis order

% Equal-angle sampling
[a1,th1,ph1]=equiangle_sampling(N);
Y1=sh2(N,th1,ph1).';
YY1=sh2(NN,th1,ph1).';
E1=abs(Y1'*diag(a1)*YY1);

% Gaussian sampling
[a2,th2,ph2]=gaussian_sampling(N);
Y2=sh2(N,th2,ph2).';
YY2=sh2(NN,th2,ph2).';
E2=abs(Y2'*diag(a2)*YY2);

% uniform sampling
[a3,th3,ph3]=uniform_sampling(N);
Y3=sh2(N,th3,ph3).';
YY3=sh2(NN,th3,ph3).';
E3=abs(Y3'*diag(a3)*YY3);

figure;

subplot(311);
plot_aliasing(E1);
title('(a)','FontSize',AxisFontSize);
caxis([-51,0]);

subplot(312);
plot_aliasing(E2);
title('(b)','FontSize',AxisFontSize);
caxis([-51,0]);

subplot(313);
plot_aliasing(E3);
title('(c)','FontSize',AxisFontSize);
caxis([-51,0]);

% Print figure in png
% print -dpng ../../../figures/chapter3/fig_aliasing_matrix.png

% Print figure in eps
% print -depsc -loose ../../../figures/chapter3/fig_aliasing_matrix.eps
