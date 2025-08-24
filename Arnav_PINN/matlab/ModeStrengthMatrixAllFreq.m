function [B] = ModeStrengthMatrixAllFreq(N,faxis,r_a,r_s)
B = zeros(length(faxis),(N+1)^2,(N+1)^2);
c = 343;
for i=1:length(faxis)
    k = 2*pi*faxis(i)/c;
    b=zeros((N+1)^2,1);
    index = 1;
    for n=0:N
        tmp = NFModeStrength(n,k,r_a,r_s);
        for m=-n:n
            b(index) = tmp;
            index = index + 1;
        end
    end
    B(i,:,:) = diag(b);
end
end
