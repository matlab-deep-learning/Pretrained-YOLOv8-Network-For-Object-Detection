function output = mapToSwishLayer(net)
lgraph = layerGraph(net);
dg = extractPrivateDirectedGraph(lgraph);

% Obtain layer table
layerTable = dg.Nodes;

% Extract the layer names into a cell array
allLayerNames = {layerTable.Layers.Name}';
lgraphRef = lgraph;

% Obtain digraph nodes
digNodes = dg.Edges.EndNodes;

% for i = 1:numel(multiplicationLayerIdx)
for layerType = lgraphRef.Layers'
    if isa(layerType,'nnet.cnn.layer.MultiplicationLayer')

        % Find the index of the layer of interest: layerName
        layerIndex = find(strcmp(allLayerNames,layerType.Name));

        % Get the layer indices that drive multiplication layer
        drivingIndices = digNodes(digNodes(:,2) == layerIndex, 1);

        % Get the driving layer names based on the indices
        drivingLayers = lgraphRef.Layers(drivingIndices);
        sigmoidIdx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.SigmoidLayer'), drivingLayers));

        if any(sigmoidIdx)
            allLayerSigmoidIdx = drivingIndices(sigmoidIdx);
            drivingIndices(sigmoidIdx) = [];
            layersToBeRemoved(1,1) = allLayerNames(allLayerSigmoidIdx);
            
            % Layer name is not cell array hence indexing is changed
            layersToBeRemoved{1,2} = layerType.Name;

            % Find the indices in the Source column matching layerName
            % Then get the names in these indices from the Destination column.
            succeedingLayers = lgraphRef.Connections.Destination(strcmp(lgraphRef.Connections.Source,layerType.Name));

            % Remove multiplication and sigmoid layers
            lgraph = removeLayers(lgraph,layersToBeRemoved);

            % Create new Swish Layer
            newLayer = swishLayer("Name",layerType.Name);

            % Add swish layer to layer graph
            lgraph = addLayers(lgraph, newLayer);
            lgraph = connectLayers(lgraph, allLayerNames{drivingIndices,1}, layerType.Name);
            for idx = 1:numel(succeedingLayers)
                lgraph = connectLayers(lgraph, layerType.Name, succeedingLayers{idx});
            end
        end
    elseif isa(layerType,'nnet.cnn.layer.ConcatenationLayer')
        dcLayer = depthConcatenationLayer(layerType.NumInputs,"Name",layerType.Name);
        lgraph = replaceLayer(lgraph,layerType.Name, dcLayer);
    end
end

output = dlnetwork(lgraph);
end