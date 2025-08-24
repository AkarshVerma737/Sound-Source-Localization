function [hh]=sph_hankel2(n,krd)
hh=((pi/(2*krd))^.5)*besselj(n+.5,krd)-1i*((pi/(2*krd))^.5)*bessely(n+.5,krd);
end