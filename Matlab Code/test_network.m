clear all, close all, clc

load('trained2.mat');

cam = webcam;
cam.Resolution = '160x120';

labels = trainedNet.Layers(end).ClassNames;


%%
figure(1);

for sample_no = 1:400
    
    % capture and preprocess image
    img = snapshot(cam);
    imshow(flipdim(img ,2));
    %img = imresize(img,0.5);
    img = im_process(img);
    
    [YPredicted,probs] = classify(trainedNet,img,'ExecutionEnvironment','cpu');
    
    maxProb = max(probs);
    probThreshold = 0.7;
    if YPredicted == "Background" || maxProb < probThreshold
        title(" ")
    else
        title(char(YPredicted),'FontSize',15)
    end
    
    drawnow
    
    pause(0.04);
end

close 1;

