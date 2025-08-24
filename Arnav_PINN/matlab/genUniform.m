function [res] = genUniform(length, lower, upper)
    % generate random number from a unifrom distribution 
    % with the limits lo and up
    range = upper - lower; 
    res = lower + range * rand(length,1);
end
