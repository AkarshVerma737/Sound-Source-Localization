function [b_n_kr]=FFModeStrength(n,kr)

% returns far field mode strength
% ref sec 2.1 in sachins localization paper

i=sqrt(-1);
jr=sph_bessel(1,n,kr);hr=sph_hankel(1,n,kr);
jm_1=sph_bessel(1,n-1,kr);hm_1=sph_hankel(1,n-1,kr);
dja=jm_1-((n+1)/(kr))*jr;dha=hm_1-((n+1)/(kr))*hr;
b_n_kr=(i^n)*(jr-(dja/dha)*hr);
end