%{
This code is used to read color images from specified folders (classes and
calling the im_process.m function to process images. The processed images
are output to another folder as black and white inverted images, and
stacked together in a large array which is saved in a .mat file
%}

clear all, close all, clc


directory = '../dataset/signs1/colour'; % source directory containing the raw rgb images
output_dir = '../dataset/bw';           % output directory where processed images will be placed
signs = ["0","1","2","3","4","5","6","7","8","9","Background"]; % subdirectory/classes
dataset_name = 'Bangla_sign_language_dataset_2.mat';    % output .mat file containing processed images

len = 0;

% find the total available data samples
for idx = 1:length(signs)
    path = [directory,'\',char(signs(idx))];
    files = dir([path,'\*.jpg']);
    len = len + length (files);
end

disp('Initializing dataset generator');

image_data = zeros(120,160,1,len);  % blank dataset
labels = categorical(zeros(len,1)); % blank dataset labels
total_samples = 0;                  % total samples included in the dataset

% add data to dataset
for idx = 1:length(signs)
    
    path = [directory,'\',char(signs(idx))];
    files = dir([path,'\*.jpg']);
    
    L = length (files);
    
    for sample_NO = 1:L              % number of samples in current label
        
        total_samples = total_samples + 1;
        image_sample = imread([path,'\',files(sample_NO).name]); % an image has been read
        write_path = [output_dir,'\',char(signs(idx)),'\',char(string(sample_NO)),'.jpg'];
        
        % the im_process.m function has been hand engineered to best
        % separate the human hand from white background
        % this line can be skipped if you are directly reading BW images
        % or change the preprocess function as you desire
        image_sample = im_process(image_sample); % preprocessing on image
        %imwrite(image_sample,write_path);   % saving processed image to file

        image_data(:,:,1,total_samples) = image_sample; % append processed image to .mat file
        labels(total_samples) = categorical(signs(idx));
        
    end
end

% save dataset as .mat file
save(dataset_name,'image_data','labels','total_samples');
disp('Dataset generation complete');
