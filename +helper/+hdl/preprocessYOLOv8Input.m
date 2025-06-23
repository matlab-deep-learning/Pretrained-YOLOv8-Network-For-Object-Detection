function output = preprocessYOLOv8Input(net, image)
    % Get the input size of the network.
    inputSize = net.Layers(1).InputSize;
    % Apply Preprocessing on the input image.
    Ibgr = image(:,:,[3,2,1]); % convert image to bgr
    img = helper.preprocess(Ibgr, inputSize);
    output = img(:,:,[3,2,1]); % convert image to rgb
end