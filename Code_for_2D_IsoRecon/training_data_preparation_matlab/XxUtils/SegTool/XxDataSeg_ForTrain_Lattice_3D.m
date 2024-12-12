function [data_seg,gt_seg] = XxDataSeg_ForTrain_Lattice_3D(data, gt, num_seg, Nx, Ny, Nz)

%% make point candidate
[r,~,~] = size(data);
[r_h,l_h,nslice] = size(gt);
ratio = round(r_h/r,1);

halfx_h = round(Nx*ratio / 2);
halfy_h = round(Ny*ratio / 2);
halfz = round(Nz / 2);

halfx = round(Nx/2);
halfy = round(Ny/2);

% calculate mask
thresh_mask = 5e-2;
mask = XxCalMask(XxNorm(gt),10,thresh_mask);
% mask = XxCalMask(gt,10,thresh_mask);

prof = sum(sum(gt,3),2);
ind = find(prof > max(gt(:))*1e-3);

upper = min(ind);
lower = max(ind);
mask(1:upper,:,:) = 0;
mask(lower:end,:,:) = 0;
mask(:,:,1:halfz) = 0;
mask(:,:,end-halfz+1:end) = 0;
mask(1:halfy_h,:,:) = 0;
mask(end-halfy_h+1:end,:,:) = 0;
mask(:,1:halfx_h,:) = 0;
mask(:,end-halfx_h+1:end,:) = 0;

ntry = 0;
while sum(mask(:)) < 1e3
    thresh_mask = thresh_mask * 0.8;
    mask = XxCalMask(gt,10,thresh_mask);
    mask(1:upper,:,:) = 0;
    mask(lower:end,:,:) = 0;
    mask(:,:,1:halfz) = 0;
    mask(:,:,end-halfz+1:end) = 0;
    mask(1:halfy_h,:,:) = 0;
    mask(end-halfy_h+1:end,:,:) = 0;
    mask(:,1:halfx_h,:) = 0;
    mask(:,end-halfx_h+1:end,:) = 0;
    ntry = ntry + 1;
    if ntry > 1e3, break; end
end

Y = 1:l_h;
X = 1:r_h;
Z = 1:nslice;
[X,Y,Z] = meshgrid(Y,X,Z);
point_list = zeros(sum(mask(:)),3);
point_list(:,1) = Y(mask(:));
point_list(:,2) = X(mask(:));
point_list(:,3) = Z(mask(:));
l_list = size(point_list,1);

% figure(1);
% subplot(1,2,1), imshow(max(gt,[],3),[]);
% subplot(1,2,2), imshow(max(mask,[],3),[]);


%% segmentation
data_seg = zeros(Ny, Nx, Nz, num_seg);
gt_seg = zeros(round(Ny*ratio), round(Nx*ratio), Nz, num_seg);

% -------------Latttice SIM-------------
% B23 sum 1e8 ar_gt:10 sum_gt 8e7
thresh_sum = 0;
thresh_ar_gt = 0;
thresh_sum_gt = 0;
count = 0;

for i = 1:num_seg
    
    %     fprintf('Segmentation: %d/%d\n',i,num_seg);
    p = randi(l_list,1);
    y1_h = point_list(p, 1) - halfy_h + 1;
    y2_h = point_list(p, 1) + halfy_h;
    x1_h = point_list(p, 2) - halfx_h + 1;
    x2_h = point_list(p, 2) + halfx_h;
    
    y1 = round(point_list(p,1)/ratio) - halfy + 1;
    y2 = round(point_list(p,1)/ratio) + halfy;
    x1 = round(point_list(p,2)/ratio) - halfx + 1;
    x2 = round(point_list(p,2)/ratio) + halfx;
    
    z1 = point_list(p, 3) - halfz + 1;
    z2 = point_list(p, 3) + halfz;
    
    patch = data(y1:y2,x1:x2,z1:z2);
    sum_patch = sum(patch(:));
    
    patch_gt = gt(y1_h:y2_h,x1_h:x2_h,z1:z2);
    sum_patch_gt = sum(patch_gt(:));
    active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1e-6);
    
    while active_range_gt < thresh_ar_gt || sum_patch_gt < thresh_sum_gt || sum_patch < thresh_sum
        p = randi(l_list,1);
        y1_h = point_list(p, 1) - halfy_h + 1;
        y2_h = point_list(p, 1) + halfy_h;
        x1_h = point_list(p, 2) - halfx_h + 1;
        x2_h = point_list(p, 2) + halfx_h;
        
        y1 = round(point_list(p,1)/ratio) - halfy + 1;
        y2 = round(point_list(p,1)/ratio) + halfy;
        x1 = round(point_list(p,2)/ratio) - halfx + 1;
        x2 = round(point_list(p,2)/ratio) + halfx;
        
        z1 = point_list(p, 3) - halfz + 1;
        z2 = point_list(p, 3) + halfz;
        
        patch = data(y1:y2,x1:x2,z1:z2);
        sum_patch = sum(patch(:));
        patch_gt = gt(y1_h:y2_h,x1_h:x2_h,z1:z2);
        sum_patch_gt = sum(patch_gt(:));
        active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1);
        
        count = count + 1;
        if count > 1e2
            thresh_ar_gt = thresh_ar_gt * 0.9;
            thresh_sum_gt = thresh_sum_gt * 0.9;
            thresh_sum = thresh_sum * 0.9;
            count = 0;
        end
        
%         fprintf('ar_gt    = %.2f\n',active_range_gt);
%         fprintf('sum_gt   = %.2f\n',sum_patch_gt);
%         fprintf('sum_data = %.2f\n',sum_patch);
%         figure(1);
%         subplot(1,2,1), imagesc(max(patch,[],3)); axis image, axis off;
%         subplot(1,2,2), imagesc(max(patch_gt,[],3)); axis image, axis off;
    end
    
    curdata = data(y1:y2,x1:x2,z1:z2);
    curgt = gt(y1_h:y2_h,x1_h:x2_h,z1:z2);
    
    data_seg(:,:,:,i) = curdata;
    gt_seg(:,:,:,i) = curgt;
end

end