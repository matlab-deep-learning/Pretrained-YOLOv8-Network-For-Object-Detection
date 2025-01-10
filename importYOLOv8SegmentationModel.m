function networkImported = importYOLOv8SegmentationModel(fileName)

% Import YOLO v8 pretrained model.
networkImported = importNetworkFromONNX(fileName);

% Remove output layer. Typically these are the output layer names for the
% imported networks.
outLayers = {"x_model_22_dfl_Sof_1","x_model_22_dfl_con_1","output0Output","output1Output"};
networkImported = removeLayers(networkImported,outLayers);

% Obtain Layer Names.
info = analyzeNetwork(networkImported,Plots="none");
layerNames = info.LayerInfo.Name;

% Replace split layers with custom splitLayer.
splitLayerIdx = find(contains(layerNames,'SplitLayer'));

for splIdx = 1:numel(splitLayerIdx)
    % Create custom split layer.
    layer1 = splitLayer(['splitLayer',num2str(splIdx)],2);

    % Replace ONNX Split Layer with custom split layer.
    networkImported = replaceLayer(networkImported, networkImported.Layers(splitLayerIdx(splIdx)).Name, layer1, ReconnectBy='order');
end

% Remove reshape layers. This operation is performed during post
% processing.
reshapeLayerIdx = find(contains(layerNames,'Reshape'));
for resahpeIdx = 1 : numel(reshapeLayerIdx)
    networkImported = removeLayers(networkImported,layerNames{reshapeLayerIdx(resahpeIdx),1});
end

% Remove batchSizeVerifier layer.
batchVerifierIdx = find(contains(layerNames,'BatchSizeVerifier'));
networkImported = removeLayers(networkImported,layerNames{batchVerifierIdx,1});

% Connect disconnected layers.
firstLayer = batchVerifierIdx - 1;
firstConvLayer = batchVerifierIdx + 1;
networkImported = connectLayers(networkImported,layerNames{firstLayer,1},layerNames{firstConvLayer,1});

% Replace sigmoid + multiplication layer with Swish layer.
networkImported = helper.mapToSwishLayer(networkImported);

% Initialize Network.
networkImported = initialize(networkImported);
end
