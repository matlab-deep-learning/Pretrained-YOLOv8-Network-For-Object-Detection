function net = preprocessNetwork(input)

lg = layerGraph(input);
for i=1:length(input.Layers)
    layer = input.Layers(i);
    % Replace concatenationLayer in the network with depthConcatenationLayer
    if isa(layer,'nnet.cnn.layer.ConcatenationLayer')
        dcLayer = depthConcatenationLayer(layer.NumInputs, "Name", layer.Name);
        lg = lg.replaceLayer(layer.Name, dcLayer);
    end
    % Replace custom splitLayer in the network with dlhdl.layer.splitLayer
    if isa(layer,'splitLayer')
        spLayer = dlhdl.layer.splitLayer(layer.Name, layer.NumOutputs);
        lg = lg.replaceLayer(layer.Name, spLayer);
    end
end
net = dlnetwork(lg);
% Replace dlhdl.layer.splitLayer with dlhdl.layer.sliceLayer
net = dnnfpga.compiler.replaceSplitLayers(net);
net = dnnfpga.compiler.insertSwishLayers(net);

end