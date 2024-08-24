function seg_data = DataAugm_SingleStack(data, SegX, SegZ, n_per_stack) 
SegY = SegX;
halfx = round(SegX/2);
halfy = round(SegY/2);
thresh_mask = 5e-2;

[Ny,Nx,Nz] = size(data);
mask = XxCalMask(data,10,thresh_mask);
ntry = 0;
while sum(mask(:)) < 1e2
    thresh_mask = thresh_mask * 0.8;
    mask = XxCalMask(data,10,thresh_mask);
    ntry = ntry + 1;
    if ntry > 1e3, break; end
end

Y = 1:Nx;
X = 1:Ny;
Z = 1:Nz;
[X,Y,Z] = meshgrid(Y,X,Z);
point_list = zeros(sum(mask(:)),3);
point_list(:,1) = Y(mask(:));
point_list(:,2) = X(mask(:));
point_list(:,3) = Z(mask(:));
l_list = size(point_list,1);

n_left = n_per_stack;
ntry = 0;
n_total = 0;
while n_left >= 1
    p = randi(l_list,1);
    y1 = point_list(p, 1) - halfy + 1;
    y2 = point_list(p, 1) + halfy;
    x1 = point_list(p, 2) - halfx + 1;
    x2 = point_list(p, 2) + halfx;
    z1 = point_list(p, 3) - SegZ + 1;
    z2 = point_list(p, 3) + SegZ;
    ntry = ntry + 1;
    if ntry > 1e3, break; end
    if y1 < 1 || x1 < 1 || z1 < 1, continue; end
    if y2 > Ny || x2 > Nx || z2 > Nz, continue; end
    
    cur_data = data(y1:y2,x1:x2,z1:z2);
    n_total = n_total + 1;
    seg_data(:,:,:,n_total) = cur_data;
    n_left = n_left - 1;
end