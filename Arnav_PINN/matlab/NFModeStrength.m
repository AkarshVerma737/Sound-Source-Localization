function [b_n_kr_krd]=NFModeStrength(n,k,r,rd)

% returns near field mode strength
% ref Eq : 9 in NF localization

i=sqrt(-1);
res = k * i^(-1*(n-1));
res = res * FFModeStrength(n,k*r);
res = res * sph_hankel2(n,k*rd);
b_n_kr_krd = res;
end