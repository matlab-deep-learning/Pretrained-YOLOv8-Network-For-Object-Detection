function yolov8Net = importYOLOv8Model(modelToImport)
networkImported = importNetworkFromONNX(modelToImport);

% Obtain Layer Names
info = analyzeNetwork(networkImported,Plots="none");
layerNames = info.LayerInfo.Name;

% Find index of first non-essential layer that should be
% removed
remLayersIdx = find(contains(layerNames,'Reshape_To_Transpose'));

% Remove other non-essential layers
idx = 1;
numLayersToRemove = size(layerNames,1)-remLayersIdx+1;
layersToBeRemoved = cell(1, numLayersToRemove);
for i = remLayersIdx:size(layerNames,1)
    layersToBeRemoved{1,idx} = layerNames{i,1};
    idx = idx + 1;
end

networkImported = removeLayers(networkImported,layersToBeRemoved);

% Find index of indices of split layers that should be replaced
% with custom splitLayer
splitLayerIdx = find(contains(layerNames,'SplitLayer'));

for i = 1:numel(splitLayerIdx)
    % Create custom split layer
    layer1 = splitLayer(['splitLayer',num2str(i)'],2);

    % Replace ONNX Split Layer with custom split layer
    networkImported = replaceLayer(networkImported,networkImported.Layers(splitLayerIdx(i)).Name,layer1,ReconnectBy='order');
end

% Remove batchSizeVerifier layer
networkImported = removeLayers(networkImported,'images_BatchSizeVerifier');
yolov8Net = connectLayers(networkImported,'images','x_model_0_conv_Conv');
end
