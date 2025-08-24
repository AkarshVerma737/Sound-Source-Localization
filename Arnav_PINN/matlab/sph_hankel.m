function [h]=sphhankel(k,n,r)
%SPH HANKEL ? Ca l c u l a t e s t h e s p h e r i c a l h a n k e l f u n t i o n s ( f rom t h e c i l y n d r i c a l one s ) o f
%f i r s t k i n d f o r a
%g i v e n ”n” v a l u e .
%
% S y n t a x : [ h ] = s p h h a n k e l ( k , n , r )
%
% I n p u t s :
% k ? wave number
% n ? v a l u e f o r t h e c u r r e n t l y ”n” p o s i t i o n i n t h e loop
% r ? r a d i o u s
%
% Ou t p u t s :
% h ? s p e h r i c a l h a n k e l o f f i r s t k i n d f o r a g i v e n ”n”
%
%
% Other m? f i l e s r e q u i r e d : none
% S u b f u n c t i o n s : none
% MAT? f i l e s r e q u i r e d : none
%
% Author : Gu i l l e rmo Moreno
% A p r i l 2008;



%????????????? BEGIN CODE ??????????????
H=besselh(n+(1/2),(k*r)); %C y l i n d r i c a l Hankel
h=sqrt(pi/(2*k*r)).*H;%S p h e r i c a l h a n k e l
%????????????? END CODE ??????????????