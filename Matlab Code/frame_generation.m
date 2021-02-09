%{
This is a sample script that was used to acquire frames from a
prerecorded video containing data sample for sign language.
%}

clear all, close all, clc


vid=VideoReader('signs.mp4');

numFrames = vid.NumberOfFrames;
n=numFrames;
  
%%

for i = 1:n
  frame = rgb2gray(imresize(read(vid,i),0.25));
  [height,width] = size(frame);
  width_start = floor((width-height)/2);
  frame = frame(:,width_start+1:width_start+height);
  %imshow(frame)
  
  imwrite(frame,['im',int2str(i),'.jpg']);
end 
