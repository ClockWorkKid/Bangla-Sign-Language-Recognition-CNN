clear all
close all
clc

im1 = imread('im (1).jpg');
im2 = imread('im (2).jpg');
im3 = imread('im (3).jpg');
im4 = imread('im (4).jpg');
im5 = imread('im (5).jpg');
im6 = imread('im (6).jpg');
im7 = imread('im (7).jpg');
im8 = imread('im (8).jpg');
im9 = imread('im (9).jpg');
im10 = imread('im (10).jpg');

%%

im_rgb = im10;
im_hsv = rgb2hsv(im_rgb);
clc

Mu = -2;
v = 1;

sat = im_hsv(:,:,2);
sat_expand = real(compand(sat,Mu,v,'mu/expander'));
sat_proc = real(sat_expand)>0.1 ;

blue = double(im_rgb(:,:,3))/255;
[blue_mag, ~] = imgradient(blue,'prewitt');
blue_mag = real(compand(blue_mag/(max(max(blue_mag))),Mu,v,'mu/expander'));


figure(1)

subplot(331), imshow(blue);
subplot(332), imshow(blue_mag);
%subplot(333), imshow();

subplot(334), imshow(sat);
subplot(335), imshow(sat_expand);
%subplot(336), imshow();

subplot(337), imshow(im_rgb);
subplot(338), imshow(blue_mag+sat_expand);
%subplot(339), imshow(blue_proc);