clear;close all;
%% settings
folder = 'Train/T91';
savepath = 'Train/T91_2x.h5';
size_input = 33;
size_label = 33;
stride = 14;

%% initialization
data = zeros(1, size_input, size_input, 1);
label = zeros(1, size_label, size_label, 1);
count = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];
% filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
    
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
        
    if size(image, 3) == 3
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));

        [hei,wid] = size(image);

        for scale = 2 : 2
            im_label = image(1:hei - mod(hei, scale), 1:wid - mod(wid, scale));

            [cropped_hei, cropped_wid] = size(im_label);

            im_input = imresize(imresize(im_label, 1/scale, 'bicubic'), [cropped_hei, cropped_wid], 'bicubic');

            for x = 1 : stride : cropped_hei-size_input+1
                for y = 1 :stride : cropped_wid-size_input+1

                    subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
                    subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                    
                    count=count+1;
                    data(1, :, :, count) = subim_input;
                    label(1, :, :,count) = subim_label;
                end
            end
        end
    end
end

order = randperm(count);
data = data(1,:,:,order);
label = label(1,:,:,order); 

%% my writing to HDF5
h5create(savepath, '/data', [1 size_input size_input count]); % width, height, channels, number 
h5create(savepath, '/label', [1 size_label size_label count]); % width, height, channels, number 
h5write(savepath, '/data', data);
h5write(savepath, '/label', label);
h5disp(savepath);
