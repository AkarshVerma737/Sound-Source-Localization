function [result] = PosCheck(L,pos)
if((sum(sum(pos < 1)) >0 ) || (sum(sum((L - pos) < 1)) >0 ))
	result = false;
else
	result = true;
end
end
