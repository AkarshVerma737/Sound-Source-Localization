%FIG_PNM generates plots of the associated Legendre functions 
% in Figure 1.10.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

close all;
clear all;

AxisFontSize=12;

x=linspace(-1,1,256);

figure(1);
for n=0:4,
    P=legendre(n,x);
    for m=0:n,
        subplot(5,5,n*5+m+1);
        PP=P(m+1,:);
        h=plot(x,PP,'k-','LineWidth',1.5,'Color',[0 0 0.5]);
        if n==4, xlabel('$x$','FontSize',AxisFontSize,'Interp','Latex'), end;
        set(gca,'FontSize',AxisFontSize);
        grid on;
        h=title(strcat('(',num2str(n),',',num2str(m),')'),'FontSize',AxisFontSize);
        if (n==4)&(m==4) axis([-1 1 0 150]); end;
    end;
end;

% Print figure in png
% figure(1); print -dpng ../../../figures/chapter1/fig_Pnm.png

% Print figure in eps
% figure(1); print -depsc -loose ../../../figures/chapter1/fig_Pnm.eps

