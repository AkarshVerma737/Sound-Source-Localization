% FIG_SAMPLING_SCHEMES generates Figures 3.1, 3.2, 3.4, 3.5, illustrating
% Equal-angle, Gaussian and uniform sampling schemes. 
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

path(path,'../../math');
path(path,'../../plot');

% Equal-angle sampling
N=5;
[a,th,ph]=equiangle_sampling(N);
plot_sampling(th,ph);

% Gaussian sampling
N=7;
[a,th,ph]=gaussian_sampling(N);
plot_sampling(th,ph);

% Uniform sampling
[V,F]=platonic_solid(5,1); % Dedecahedron
[th,ph,a]=c2s(V(:,1),V(:,2),V(:,3));
ph=ph+pi;
plot_sampling(th,ph);


% Nearly-uniform sampling
N=8;
[a,th,ph]=uniform_sampling(N);
ph=ph+pi;
plot_sampling(th,ph);


% Print figures in png
% figure(1); print -dpng ../../../figures/chapter3/fig_sampling_equal_angle_sphere.png
% figure(2); print -dpng ../../../figures/chapter3/fig_sampling_equal_angle_grid.png
% figure(3); print -dpng ../../../figures/chapter3/fig_sampling_gaussian_sphere.png
% figure(4); print -dpng ../../../figures/chapter3/fig_sampling_gaussian_grid.png
% figure(5); print -dpng ../../../figures/chapter3/fig_sampling_uniform_sphere.png
% figure(6); print -dpng ../../../figures/chapter3/fig_sampling_uniform_grid.png
% figure(7); print -dpng ../../../figures/chapter3/fig_sampling_nearly-uniform_sphere.png
% figure(8); print -dpng ../../../figures/chapter3/fig_sampling_nearly-uniform_grid.png

% Print figures in eps
% figure(1); print -depsc -loose -r300 ../../../figures/chapter3/fig_sampling_equal_angle_sphere.eps
% figure(2); print -depsc -loose ../../../figures/chapter3/fig_sampling_equal_angle_grid.eps
% figure(3); print -depsc -loose -r300 ../../../figures/chapter3/fig_sampling_gaussian_sphere.eps
% figure(4); print -depsc -loose ../../../figures/chapter3/fig_sampling_gaussian_grid.eps
% figure(5); print -depsc -loose -r300 ../../../figures/chapter3/fig_sampling_uniform_sphere.eps
% figure(6); print -depsc -loose ../../../figures/chapter3/fig_sampling_uniform_grid.eps
% figure(7); print -depsc -loose -r300 ../../../figures/chapter3/fig_sampling_nearly-uniform_sphere.eps
% figure(8); print -depsc -loose ../../../figures/chapter3/fig_sampling_nearly-uniform_grid.eps
