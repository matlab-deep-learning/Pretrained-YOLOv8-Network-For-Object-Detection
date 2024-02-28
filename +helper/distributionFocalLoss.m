function boxes = distributionFocalLoss(boxes)
% Distribution Focal Loss Module

sz = size(boxes);
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
boxes = permute(boxes,[2,1,3,4]); % 1 d qual like python
boxes = reshape(boxes,anchors,c1,4,batch);
boxes = permute(boxes,[2,1,3,4]);

% Transpose Operation
boxes = permute(boxes,[3,2,1,4]);

% softmax along the channel dimension
boxes = softmax(dlarray(boxes,'SSC'));  % produces a diff of 10^-3
boxes = extractdata(boxes);

% 1-d conv operation
% Define weights
weights = [0:c1-1];
m = size(boxes,1);
n = size(boxes,2);
weights = reshape(repmat(weights,m*n,1),m,n,[]);
% Conv operation
boxes = boxes .* weights;
boxes = sum(boxes,3);   % diff of 10^-2 because of above diff

% Reshape Operation
boxes = permute(boxes,[2,1,3,4]);
boxes = reshape(boxes,anchors,4,batch);
boxes = permute(boxes,[2,1,3,4]);

end













