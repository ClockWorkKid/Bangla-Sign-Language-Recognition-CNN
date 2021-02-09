%{
The first script used in the project for acquiring data. Prior to using
this function, the MATLAB webcam support package has to be installed, which
has also been included in the repository.

The function will initialize the camera and save the sign images to
corresponding folders. After acquiring images, you need to use the
dataset_generator.m file to process images and create a .mat file as a
dataset.
%}

clear all, close all, clc

cam = webcam;
cam.Resolution = '160x120';

% destination directory
directory = 'New_Images';

% create these subfolders prior to using the script
signs = ["0","1","2","3","4","5","ThumbsUp","Pinky","CallMe","Background"];

% prefix used for image names
name = input('Enter your name:  ','s');

for idx = 1:length(signs)
    
    % the loop will iterate through the classes and save images to
    % corresponding sub-directories
    disp(['Show ',char(signs(idx))]);
    input('Press enter to continue');
    figure(1);
    
    for sample_no = 1:80
        
        path = [directory,'\',char(signs(idx)),'\',name,int2str(sample_no),'.jpg'];
        
        img = snapshot(cam);
        imshow(flipdim(img ,2));
        img = rgb2gray(img);
        [im_h, im_w] = size(img);
        w_start = floor((im_w - im_h)/2);
        img = img(:,w_start+1:w_start+im_h);
        imwrite(img,path);
        pause(0.04);
    end
    close 1;
end

clear cam;