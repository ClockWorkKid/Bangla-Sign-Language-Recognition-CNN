function processed = im_process(image)

im_hsv = rgb2hsv(image);

Mu = 2;
v = 1;

sat = im_hsv(:,:,2);
sat_expand = real(compand(sat,Mu,v,'mu/expander'));

blue = double(image(:,:,3))/255;
[blue_mag, ~] = imgradient(blue,'prewitt');
blue_mag = real(compand(blue_mag/(max(max(blue_mag))),Mu,v,'mu/expander'));

processed = sat_expand+blue_mag;
processed = processed/max(max(processed));
end