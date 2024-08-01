function output = preprocess(img,targetSize)
% This function preprocesses the input image.

% Copyright 2024 The MathWorks, Inc.

imgSize = size(img,1:2);

% Compute scaling ratio
scaleRatio = min(targetSize(1)./ imgSize(1), targetSize(2) / imgSize(2));
padSize = [round(imgSize(1) * scaleRatio), round(imgSize(2) * scaleRatio)];

% Compute padding values
dw = targetSize(1) - padSize(2);  % w padding
dw = mod(dw, 32);                 % w padding
dw = dw./2;                       % divide padding into 2 sides

dh = targetSize(2) - padSize(1);  % h padding
dh = mod(dh, 32);                 % h padding
dh = dh./2;                       % divide padding into 2 sides

% Resize input image
I = imresize(img, padSize,'bilinear',Antialiasing = false);

% Pad values to the image
top     = round(dh - 0.1);
bottom  = round(dh + 0.1);
left    = round(dw - 0.1);
right   = round(dw + 0.1);

I = padarray(I,[top left],114,'pre');
I = padarray(I,[bottom,right],114,'post');

% Convert uint8 to single
I = single(I);

% Rescale image pixes to [0,1]
output = I./255;
end
