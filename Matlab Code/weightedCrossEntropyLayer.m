% weightedCrossEntropyLayer(classNames,classWeights) creates a weighted
% cross entropy loss layer for deep learning. classNames is a cell array of
% class names, and classWeights is a vector of the corresponding weights.

classdef weightedCrossEntropyLayer < nnet.layer.ClassificationLayer
    
    properties
        ClassWeights
    end
    
    methods
        function this = weightedCrossEntropyLayer(classNames,classWeights)          
            this.ClassWeights = classWeights;
            this.ClassNames = classNames;         
        end
        
        function loss = forwardLoss(this,Y,T)
            numObs = size(Y,4);
            W = shiftdim(repmat(this.ClassWeights,[1 numObs]),-2);
            loss = -sum( sum( sum( W.*T.*log(Y) ) ) )./numObs;
        end
        
        function dLdX = backwardLoss(this,Y,T)
            numObs = size(Y,4);
            W = shiftdim(repmat(this.ClassWeights,[1 numObs]),-2);
            dLdX = -W.*T./Y;
            dLdX = dLdX./numObs;
        end
    end
end

