function op = per(op)
if size(size(op),2)==4
op = permute(op,[3,4,2,1]);

else
op = permute(op,[2,3,1]);
end
end