
function newCoords = postProcessYOLOv8(img1_shape,coords,img0_shape)

% Rescale coords (xyxy) from img1_shape to img0_shape
gain = min(img1_shape(1) / img0_shape(1), img1_shape(2) / img0_shape(2));  
pad = coder.nullcopy(zeros(1,2));
pad(1) = (img1_shape(2) - img0_shape(2) * gain)./ 2;
pad(2) = (img1_shape(1) - img0_shape(1) * gain)./ 2; 

newCoords = coords - repmat(pad,size(coords,1),2);
newCoords = newCoords./gain;
newCoords(newCoords<0) = 0.1;

newCoords(:,1) = min(newCoords(:,1),img0_shape(2));
newCoords(:,2) = min(newCoords(:,2),img0_shape(1));
newCoords(:,3) = min(newCoords(:,3),img0_shape(2));
newCoords(:,4) = min(newCoords(:,4),img0_shape(1));

end