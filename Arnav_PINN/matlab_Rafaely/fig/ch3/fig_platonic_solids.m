%FIG_PLATONIC_SOLIDS generates Figures 3.3, illustrating the five
% Platonic solids. 
%
% Remark: uses the function platonic_solid by Kevin Moerman from
% the Matlab File Exchange.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

clear all; 
close all; 

clc;

r=1;
zoom=1.5;
subf=[[2,3];[4,5];[7,8];[11,12];[9,10]]; % corrected to swap last two shapes


figure;

for i=1:5,
    subplot(2,6,[subf(i,:)]);
    [V,F]=platonic_solid(i,r);
    patch('Faces',F,'Vertices',V,'FaceColor','c','FaceAlpha',0.8,'EdgeColor','k','LineWidth',1.5); 
    axis([-1,1,-1,1,-1,1]);
    view(3); 
    axis off;
    camzoom(zoom);
end;

% Print figure in png
% print -dpng ../../../figures/chapter3/fig_platonic_solids.png

% Print figure in eps
% print -depsc -loose -r300 ../../../figures/chapter3/fig_platonic_solids.eps
