function [gt_seg,data_seg,rl_seg] = XxDataSeg_ForTrain_3D_v2(gt, data, num_seg, Nx, Ny, RotFlag, depth, half_side_flag)

% ------------------------------------------------------------------------
% Randomly crop and rotate image patch pairs from data and gt of same size
% rather than gt with larger size
% 
% usage:  [DATA_SEG, GT_SEG] = XxDataSeg_ForTrain(data, gt, num_seg, Nx, 
%                              Ny, RotFlag)
% where,
%    DATA_SEG    -- image patches of raw SIM data (input of networks)
%    GT_SEG      -- image patches of SIM SR data (ground truth)
%    data        -- raw SIM data before segmentation
%    gt          -- SIM SR data before segmentation
%    num_seg     -- number of patches to crop
%    Nx, Ny      -- patch size to crop
%    RotFlag     -- >=1 for random angle rotation, 0 for no rotatioin
%
% ATTENTION: rl seg was meant for R-L, which consists of the same content
% as gt, but later was used for generating input for upsampling(S4). But
% this code can be used for both S3&S4
% ------------------------------------------------------------------------

addpath(genpath('./XxUtils'));

[r,l,~] = size(data(:,:,1));

if RotFlag == 0
    new_r = Nx;
    new_l = Ny;
else
    new_r = ceil(Nx * 1.5);
    new_l = ceil(Ny * 1.5);
    if new_r > r || new_l > l
        new_r = r;
        new_l = l;
        RotFlag = 0;
    end
end

% calculate foreground mask
thresh_mask = 0.7*max(gt(:));
ksize = 3;
mask = XxCalMask(data(:,:,ceil(depth/2)),ksize,thresh_mask);
while sum(mask(:)) < 1e3
    thresh_mask = thresh_mask * 0.9;
    mask = XxCalMask(data(:,:,ceil(depth/2)),ksize,thresh_mask);
end

% figure(1);
% subplot(1,2,1), imshow(gt(:,:,ceil(depth/2)),[]);
% subplot(1,2,2), imshow(mask,[]);

Y = 1:l;
X = 1:r;
[X,Y] = meshgrid(Y,X);
point_list = zeros(sum(mask(:)),2);
point_list(:,1) = Y(mask(:));
point_list(:,2) = X(mask(:));
l_list = size(point_list,1);

halfx = round(new_r / 2);
halfy = round(new_l / 2);

data_seg = [];
gt_seg = [];
rl_seg = [];

%% Crop patches
thresh_ar = 0.5;
thresh_ar_gt = 0.5;
thresh_sum = 10;
count = 0;

for i = 1:num_seg
    
    p = randi(l_list,1);
    y1 = point_list(p, 1) - halfy + 1;
    y2 = point_list(p, 1) + halfy;
    x1 = point_list(p, 2) - halfx + 1;
    x2 = point_list(p, 2) + halfx;
    count_rand = 0;
    while (y1<2 || y2>l || x1<2 || x2>r)
        p = randi(l_list,1);
        y1 = point_list(p, 1) - halfy + 1;
        y2 = point_list(p, 1) + halfy;
        x1 = point_list(p, 2) - halfx + 1;
        x2 = point_list(p, 2) + halfx;
        count_rand = count_rand + 1;
        if count_rand > 1e3, break; end
    end
    if count_rand > 1e3, break; end
    
    if RotFlag >= 1 % if random rotate
        degree = randi(360, 1);
        patch_gt = imrotate(sum(gt(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
        tx_gt = new_r-Nx;
        ty_gt = new_l-Ny;
        patch_gt = patch_gt(tx_gt+1:tx_gt+Nx,ty_gt+1:ty_gt+Ny);
        active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1e-4);
        
        patch = imrotate(sum(data(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
        tx = round(new_r/2)-round(Nx/2);
        ty = round(new_l/2)-round(Ny/2);
        patch = patch(tx+1:tx+Nx,ty+1:ty+Ny);
        sum_patch = sum(patch(:));
        active_range = double(prctile(patch(:),99.9)) / double(prctile(patch(:),0.1)+1e-4);
        
    else % not random rotate
        patch_gt = sum(gt(x1:x2,y1:y2,:), 3);
        active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1e-3);
        
        patch = sum(data(x1:x2,y1:y2,:), 3);
        sum_patch = sum(patch(:));
        active_range = double(prctile(patch(:),99.9)) / double(prctile(patch(:),0.1)+1e-3);
    end
    
    while active_range_gt < thresh_ar_gt || sum_patch < thresh_sum || active_range < thresh_ar
        x1 = randi(r - new_r + 1, 1);
        x2 = x1 + new_r - 1;
        y1 = randi(l - new_l + 1, 1);
        y2 = y1 + new_l - 1;
        
        if RotFlag >= 1 % if random rotate
            degree = randi(360, 1);
            patch_gt = imrotate(sum(gt(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
            tx_gt = new_r-Nx;
            ty_gt = new_l-Ny;
            patch_gt = patch_gt(tx_gt+1:tx_gt+Nx,ty_gt+1:ty_gt+Ny);
            active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1);
        
            patch = imrotate(sum(data(x1:x2,y1:y2,:),3),degree,'bilinear','crop');
            tx = round(new_r/2)-round(Nx/2);
            ty = round(new_l/2)-round(Ny/2);
            patch = patch(tx+1:tx+Nx,ty+1:ty+Ny);
            sum_patch = sum(patch(:));
            active_range = double(prctile(patch(:),99.9)) / double(prctile(patch(:),0.1)+1);
        else % not random rotate
            patch_gt = sum(gt(x1:x2,y1:y2,:), 3);
            active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1);
        
            patch = sum(data(x1:x2,y1:y2,:),3);
            sum_patch = sum(patch(:));
            active_range = double(prctile(patch(:),99.9)) / double(prctile(patch(:),0.1)+1);
        end
        
        count = count + 1;
        if count > 1e2
            thresh_ar_gt = thresh_ar_gt * 0.9;
            thresh_sum = thresh_sum * 0.9;
            thresh_ar = thresh_ar * 0.9;
            count = 0;
        end
        
%         fprintf('ar_gt    = %.2f\n',active_range_gt);
%         fprintf('sum_gt   = %.2f\n',sum_patch_gt);
%         fprintf('ar_data  = %.2f\n',active_range);
%         figure(1);
%         subplot(1,2,1), imagesc(patch(:,:,1)); axis image, axis off;
%         subplot(1,2,2), imagesc(patch_gt(:,:,1)); axis image, axis off;
    end
    
    if RotFlag >= 1
        tgt = imrotate(gt(x1:x2,y1:y2,:),degree,'bilinear','crop');
        tdata = imrotate(data(x1:x2,y1:y2,:),degree,'bilinear','crop');
        if isempty(gt_seg)
            gt_seg = tgt(tx_gt+1:tx_gt+Nx,ty_gt+1:ty_gt+Ny,:);
        else
            h_gt = size(tgt, 3);
            gt_seg(:,:,end+1:end+h_gt) = tgt(tx_gt+1:tx_gt+Nx,ty_gt+1:ty_gt+Ny,:);
        end
        if isempty(data_seg)
            if half_side_flag
                data_seg = tdata(tx+1:tx+Nx,ty+1:ty+Ny,1);
            else
                data_seg = tdata(tx+1:tx+Nx,ty+1:ty+Ny,ceil(depth/2));
            end
        else
            if half_side_flag
                data_seg(:,:,end+1) = tdata(tx+1:tx+Nx,ty+1:ty+Ny,1);
            else
                data_seg(:,:,end+1) = tdata(tx+1:tx+Nx,ty+1:ty+Ny,ceil(depth/2));
            end
        end
        if isempty(rl_seg)
            rl_seg = tdata(tx+1:tx+Nx,ty+1:ty+Ny,:);
        else
            h = size(tdata, 3);
            rl_seg(:,:,end+1:end+h) = tdata(tx+1:tx+Nx,ty+1:ty+Ny,:);
        end
    else
        if isempty(data_seg)
            data_seg = data(x1:x2,y1:y2,1);
        else
            data_seg(:,:,end+1) = data(x1:x2,y1:y2,1);
        end
        if isempty(gt_seg)
            gt_seg = gt(x1:x2,y1:y2,:);
        else
            h_gt = size(gt, 3);
            gt_seg(:,:,end+1:end+h_gt) = gt(x1:x2,y1:y2,:);
        end
        if isempty(rl_seg)
            rl_seg = data(x1:x2,y1:y2,:);
        else
            h = size(data, 3);
            rl_seg(:,:,end+1:end+h) = data(x1:x2,y1:y2,:);
        end
    end
end