classdef splitLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ... 
        & nnet.layer.Acceleratable
    %#codegen

    methods
        function layer = splitLayer(layerName,numOutputs)
            layer.Name = layerName;
            layer.NumOutputs = numOutputs;
        end

        function [Z1,Z2] = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.

            channelSplit = size(X,3)./layer.NumOutputs;
            Z2 = X(:,:,1:channelSplit,:);
            Z1 = X(:,:,channelSplit+1:end,:);
        end
    end
end
