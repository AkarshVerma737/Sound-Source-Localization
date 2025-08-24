function [Y]=spherical_harmonics(n,m,theta,phi)

Y=0;Yp=0;Yn=0; 
Pnm=legendre(n,cos(theta));
if(m>=0)
    sigma=sqrt(((2*n+1)*factorial(n-m))/(4*pi*factorial(n+m)));
    Yp=sigma.*exp(i*m*phi)*Pnm(m+1,:);
    Y=Yp; 
end
if(m<=-1)
    mm=abs(m);
    sigma=sqrt(((2*n+1)*factorial(n-mm))/(4*pi*factorial(n+mm)));
    Yn=((-1)^mm)*sigma*exp(-i*mm*phi)*Pnm(mm+1,:); 
    Y=Yn ;
end