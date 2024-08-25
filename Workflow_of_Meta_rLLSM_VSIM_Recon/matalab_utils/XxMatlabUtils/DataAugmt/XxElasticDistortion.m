function img_out = XxElasticDistortion(img_in, alpha, sigma, alpha_affine, mode, show)

addpath(genpath('/home/zkyd/Documents/XxMatlabUtils'));

if nargin < 6, show = 0; end
if nargin < 5, mode = 1; end
if nargin < 4, alpha_affine = 0; end
if nargin < 3, sigma = 5; end
if nargin < 2, alpha = 5; end
if nargin < 1
    img_in = imread('Test.tif'); 
    img_in = img_in(300:500, 300:500);
end

% perform affien transformation
shape = size(img_in);
shape = shape(1:2);
n_img = size(img_in, 3);
pt_c = round(shape / 2);
size_t = round(min(shape) / 3);
pts1 = [pt_c + size_t;
        pt_c(1) + size_t, pt_c(2) - size_t;
        pt_c - size_t];
pts2 = pts1 + round(alpha_affine * (rand(size(pts1)) * 2 - 1));
tform1 = fitgeotrans(pts1, pts2, 'affine');

img_affine = zeros(size(imwarp(img_in(:,:,1), tform1)));
for i = 1:n_img
    img_affine(:,:,i) = imwarp(img_in(:,:,i), tform1);
end

% perform elastic distorsions
if mode == 1
    dx = rand(shape) * 2 - 1;
    dy = rand(shape) * 2 - 1;
    W = fspecial('gaussian', [size_t, size_t], sigma);
    dx = alpha * imfilter(dx, W, 'replicate');
    dy = alpha * imfilter(dy, W, 'replicate');
else
    shape_sacled = round(shape / sigma);
    dx = alpha * (rand(shape_sacled) * 2 - 1);
    dy = alpha * (rand(shape_sacled) * 2 - 1);
    dx = imresize(dx, shape);
    dy = imresize(dy, shape);
end

x = 1:shape(2);
y = 1:shape(1);
[X,Y] = meshgrid(x, y);

img_out = zeros(size(img_affine));
for i = 1:n_img
    img_out(:,:,i) = interp2(X, Y, double(img_affine(:,:,i)), X + dx, Y + dy);
end
% img_out = img_out(alpha + 1:end-alpha, alpha + 1:end-alpha, :);
img_out(isnan(img_out)) = 0;

if show
    figure(1);
    subplot(2,2,1), imagesc(dx); axis equal; axis off;
    subplot(2,2,2), imagesc(dy); axis equal; axis off;
    subplot(2,2,3), imshow(img_in(:,:,1), []);
    subplot(2,2,4), imshow(img_out(:,:,1), []);
    
    figure(2), imshow(img_in(:,:,1), []);
    figure(3), imshow(img_out(:,:,1), []);
end

end