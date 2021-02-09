clear all, close all, clc

%% Process dataset

%{
load the dataset previously generated, and then divide the dataset into
train, validation and test sets
%}

load('Bangla_sign_language_dataset_2.mat'); %load dataset
% dataset variables: 'image_data' 'labels' 'total_samples'

indices = randperm(total_samples); % generate random index
frac_valid = 0.2;      % fraction of validation data
frac_test = 0.1;            % fraction of test data


% number of train, test and validation samples
len_test = round(total_samples * frac_test);
len_valid = round(total_samples * frac_valid);
len_train = total_samples - len_test - len_valid;

% indices for train, test and validation sets
idx_test = indices(1 : len_test);
idx_valid = indices(len_test+1 : len_test+len_valid);
idx_train = indices(len_test+len_valid+1 : total_samples);

% seperate the train, test and validation sets
XTrain = image_data(:,:,1,idx_train);
XTest = image_data(:,:,1,idx_test);
XValidation = image_data(:,:,1,idx_valid);

YTrain = labels(idx_train);
YTest = labels(idx_test);
YValidation = labels(idx_valid);

clear image_data;
disp('Dataset processing complete');


%% Add Data Augmentation

%{
Create an augmented image datastore for automatic augmentation and resizing
 of the spectrograms. Translate the spectrogram randomly up to 10 frames (100 ms)
 forwards or backwards in time, and scale the spectrograms along the 
time axis up or down by 20 percent. Augmenting the data somewhat increases 
the effective size of the training data and helps prevent the network from 
overfitting. The augmented image datastore creates augmented images in 
real time and inputs these to the network. No augmented spectrograms are 
saved in memory.
%}

epsil = 1e-6;

sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter(...
    'RandXTranslation',[-10 10],...
    'RandXScale',[0.8 1.2],...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain,...
    'DataAugmentation',augmenter,...
    'OutputSizeMode','randcrop');

disp('Data augmentation complete');


%% Define neural network architecture

%{
Create a simple network architecture as an array of layers. Use convolutional
 and batch normalization layers, and downsample the feature maps "spatially" 
(that is, in time and frequency) using max pooling layers. Add a final max 
pooling layer that pools the input feature map globally over time. This 
enforces (approximate) time-translation variance in the input spectrograms, 
which seems reasonable if we expect the network to perform the same 
classification independent of the exact position of the speech in time. 
This global pooling also significantly reduces the number of parameters of 
the final fully connected layer. To reduce the chance of the network 
memorizing specific features of the training data, add a small amount of 
dropout to the inputs to the layers with the largest number of parameters. 
These layers are the convolutional layers with the largest number of filters. 
The final convolutional layers have 64*64*3*3 = 36864 weights each (plus biases). 
The final fully connected layer has 12*5*64 = 3840 weights.

Use a weighted cross entropy classification loss. 
weightedCrossEntropyLayer(classNames,classWeights) creates a custom layer 
that calculates the weighted cross entropy loss for the classes in classNames 
using the weights in classWeights. To give each class equal weight in the loss, 
use class weights that are inversely proportional to the number of training 
examples of each class. When using the Adam optimizer to train the network, 
training should be independent of the overall normalization of the class weights.
%}

classNames = categories(YTrain);
classWeights = 1./countcats(YTrain);
classWeights = classWeights/mean(classWeights);
numClasses = numel(classNames);

dropoutProb = 0.2;
layers = [
    imageInputLayer(imageSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2,'Padding',[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2,'Padding',[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([1 11])

    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedCrossEntropyLayer(classNames,classWeights)];

disp('Neural network model created');

%% Train neural network model

%{
Specify the training options. Use the Adam optimizer with a mini-batch size 
of 128 and a learning rate of 5e-4. Train for 25 epochs and reduce the 
learning rate by a factor of 10 after 20 epochs.
%}

doTraining = true;

miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',5e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',Inf, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

%{
Train the network. If you do not have a GPU, then training the network can 
take some time. To load a pretrained network instead of training a network 
from scratch, set doTraining to false.
%}

if doTraining
    trainedNet = trainNetwork(augimdsTrain,layers,options);
    save('trained2.mat','trainedNet','imageSize');
else
    load('trained2.mat');
end

disp('Neural network training complete');

%% Test Network

%{
Calculate the final accuracy on the training set (without data augmentation) 
and validation set. Plot the confusion matrix. The network is very accurate 
on this data set. However, the training, validation, and test data all come 
from similar distributions that do not necessarily reflect real-world 
environments. This applies in particular to the unknown category which 
contains utterances of a small number of words only
%}

YValPred = classify(trainedNet,XTest);
testError = mean(YValPred ~= YTest);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Test error: " + testError*100 + "%")

figure
plotconfusion(YTest,YValPred,'Test Data')

disp('Network Testing Complete');

%% CPU Check

%{
In applications with constrained hardware resources, such as mobile 
applications, it is important to respect limitations on available memory 
and computational resources. Compute the total size of the network in 
kilobytes, and test its prediction speed when using the CPU. The prediction 
time is the time for classifying a single input image. If you input multiple 
images to the network, these can be classified simultaneously, leading to 
shorter prediction times per image. For this application, however, 
the single-image prediction time is the most relevant
%}

info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i=1:100
    x = randn(imageSize);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end

disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")

%{
REFERENCES
[1] Warden P. "Speech Commands: A public dataset for single-word speech recognition", 2017. 
Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz. 
Copyright Google 2017. The Speech Commands Dataset is licensed under the 
Creative Commons Attribution 4.0 license, available here: 
https://creativecommons.org/licenses/by/4.0/legalcode.
%}

