clc; clear; close all;
% Load an image (replace 'path_to_image.jpg' with your actual image path)
% original_image = imread('TC_3A_Train.jpg');

% % Specify the rectangle's top-left corner and size in pixels
% x = 300;  % x-coordinate of top-left corner
% y = 100;  % y-coordinate of top-left corner
% width = 1320;  % Width of the rectangle
% height = 600;  % Height of the rectangle
% 
% % Crop the image
% cropped_image = original_image(y:y+height-1, x:x+width-1, :);
% 
% % Save the cropped image (replace 'path_to_cropped_image.jpg' with your desired save path)
% imwrite(cropped_image, 'TC_3B_cropped_image.jpg');
% 
% % Display the original and cropped images for visual verification
% figure;
% subplot(1, 2, 1);
% imshow(original_image);
% title('Original Image');
% 
% subplot(1, 2, 2);
% imshow(cropped_image);
% title('Cropped Image');


original_image = imread('Texas_Segmentation_2.jpg');
% Specify the rectangle's top-left corner and size in pixels
x = 200;  % x-coordinate of top-left corner
y = 100;  % y-coordinate of top-left corner
width = 1520;  % Width of the rectangle
height = 700;  % Height of the rectangle

% Crop the image
cropped_image = original_image(y:y+height-1, x:x+width-1, :);

% Save the cropped image (replace 'path_to_cropped_image.jpg' with your desired save path)
imwrite(cropped_image, 'Cropped_image.jpg');

% Display the original and cropped images for visual verification
figure;
subplot(1, 2, 1);
imshow(original_image);
title('Original Image');

subplot(1, 2, 2);
imshow(cropped_image);
title('Cropped Image');

