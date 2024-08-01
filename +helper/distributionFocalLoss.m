function boxes = distributionFocalLoss(boxesInp)
% Distribution Focal Loss Module

sz = size(boxesInp);
c1=16; % pre-specified

% compute batch. channel and anchors
if size(sz,2)>2
    batch=sz(3);
else
    batch = 1;
end
ch = sz(1);
anchors = sz(2);

% Reshape Operation
boxesInp = permute(boxesInp,[2,1,3,4]); 
boxesReshaped = reshape(boxesInp,anchors,c1,4,batch);
boxesMapped = permute(boxesReshaped,[2,1,3,4]);

% Transpose Operation
boxesTrans = permute(boxesMapped,[3,2,1,4]);
boxesTrans = extractdata(boxesTrans);

% softmax along the channel dimension
boxesMax = softmax(dlarray(boxesTrans,'SSC')); 
boxesMax = extractdata(boxesMax);

% 1-d conv operation
% Define weights
weights = [0:c1-1];
m = size(boxesMax,1);
n = size(boxesMax,2);
weights = reshape(repmat(weights,m*n,1),m,n,[]);
% Conv operation
boxesConv = boxesMax .* weights;
boxesTotal = sum(boxesConv,3); 

% Reshape Operation
boxesTmp = permute(boxesTotal,[2,1,3,4]);
boxesTmpReshaped = reshape(boxesTmp,anchors,4,batch);
boxes = permute(boxesTmpReshaped,[2,1,3,4]);

end
