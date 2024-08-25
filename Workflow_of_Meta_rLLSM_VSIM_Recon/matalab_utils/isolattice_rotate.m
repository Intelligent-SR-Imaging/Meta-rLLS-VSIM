function [viewA, viewB, filename_rot] = isolattice_rotate(data, param, filename, loadFile)
    if nargin == 4 && loadFile > 0
        disp('Load Cached Rotation Result');        
        [saveFolder, savename, ~] = fileparts(filename);
        fileA = XxDir(saveFolder, [savename, '_Rot_*_A.tif']);
        fileA = fileA{1};
        fileB = XxDir(saveFolder, [savename, '_Rot_*_B.tif']);
        fileB = fileB{1};
        filename_rot = fileA(1:end-6);        
        viewA = XxReadTiff(fileA);
        viewB = XxReadTiff(fileB);
        return;
    end

    [Nx, Ny, Nz] = size(data);
    Nx = round(Nx * param.scale_x);
    Ny = round(Ny * param.scale_y); 
    Nz = round(Nz * param.scale_z); 
    
    RotAngle_y0 = param.RotAngle_y0;
    rot_y = param.rot_y;
    rot_x = param.rot_x;
    rangeZ = param.rangeZ;
    dxy = param.dxy;
    dz = param.dz;
    cutZ = param.cutZ;
    
    % Step1. fine-tune rotation x
    data_roty = XxRotate3D(data, -RotAngle_y0, 0, dz/dxy);
    % crop shape
    cutShape = round((size(data_roty) - [Nx, Ny, Nz])/2);
    data_roty = data_roty(cutShape(1)+ 1 : cutShape(1)+Nx,...
                         cutShape(2)+ 1 : cutShape(2)+Ny,...
                         cutShape(3)+ 1 : cutShape(3)+Nz);
%     data_mipyz = squeeze(max(data_roty, [], 1));
    data_mipyz = squeeze(max(data_roty(1+100:end-100,1+100:end-100,1:end), [], 1));
    
    maxVs = 0;
    RotAngle_x = 0;
    dispZ = 0; % relative to the centerZ
    for rot = rot_x
        rdata = imrotate(data_mipyz, rot, 'bilinear', 'crop'); % loose
        rdata2 = fliplr(rdata);
        % low-pass filter
        rdata = imgaussfilt(rdata, 1);
        rdata2 = imgaussfilt(rdata2, 1);
        % optimize shift by xcorr
        [maxDisp, ~, maxV] = OptDispX(rdata, rdata2, 1, rangeZ);
        if maxV > maxVs
            RotAngle_x = rot;
            dispZ = maxDisp;
            maxVs = maxV;
        end
    end
    fprintf('estimated angle_x is %.2f, dispZ is %d\n', RotAngle_x, dispZ);
    if param.verbose
        figure(10), subplot(1,4,1), imshow(data_mipyz, []), title('MIP-YZ');
        subplot(1,4,2), imshow(imrotate(data_mipyz, RotAngle_x, 'bilinear', 'crop'), []), title('MIP-YZ');
        cutz = round((size(data_mipyz, 2) + dispZ) / 2);
        hold on, line([cutz, cutz],[1, size(data_mipyz,1)]), hold off;
    end

    % Step2. fine-tune rotation angle along y-axis
    data_rotxy = XxRotate3D(data, -RotAngle_y0, -RotAngle_x, dz/dxy);
    cutShape = round((size(data_rotxy) - [Nx, Ny, Nz])/2);
    data_rotxy = data_rotxy(cutShape(1)+ 1 : cutShape(1)+Nx,...
                         cutShape(2)+ 1 : cutShape(2)+Ny,...
                         cutShape(3)+ 1 : cutShape(3)+Nz);
    data_mipxz = squeeze(max(data_rotxy(1+100:end-100,1+100:end-100,1:end), [], 2));
    RotAngle_deltay = 0;
    maxVs2 = 0;
    for rot = rot_y
        rdata = imrotate(data_mipxz, rot, 'bilinear', 'crop'); % loose
        rdata2 = fliplr(rdata);
        [maxDisp, ~, maxV] = OptDispX(rdata, rdata2, 1, rangeZ);
        if maxV > maxVs2
            RotAngle_deltay = rot;
            dispZ = maxDisp;
            maxVs2 = maxV;  
        end
    end
    RotAngle_y = RotAngle_y0 + RotAngle_deltay;
    fprintf('estimated angle_y is %.2f, dispZ is %d\n', RotAngle_deltay, dispZ);
    if param.verbose
        figure(10), subplot(1,4,3), imshow(data_mipxz, []), title('MIP-XZ');
        subplot(1,4,4), imshow(imrotate(data_mipxz, RotAngle_deltay, 'bilinear', 'crop'), []), title('MIP-XZ');
        cutz = round((size(data_mipxz, 2) + dispZ) / 2);
        hold on, line([cutz, cutz],[1, size(data_mipxz,1)]), hold off;
    end
        
    % step3锛歳otate data in x & y axis
    data_rot = XxRotate3D(data, -RotAngle_y, -RotAngle_x, dz/dxy);
    cutShape = round((size(data_rot) - [Nx, Ny, Nz])/2);
    data_rot = data_rot(cutShape(1)+ 1 : cutShape(1)+Nx,...
                        cutShape(2)+ 1 : cutShape(2)+Ny,...
                        cutShape(3)+ 1 : cutShape(3)+Nz);

    %% save to file
    data_rot = uint16(XxNorm(data_rot) * 65535);
    OutSuffix = ['_rot-x' num2str(RotAngle_y) '-y' num2str(RotAngle_x)];
    
    % seperate into view A and view B and save to file
    viewA = data_rot(:,:,1:round((size(data_rot,3)+dispZ) / 2));
    data_rot2 = flip(data_rot, 3);
    viewB = data_rot2(:,:,1:round((size(data_rot,3)-dispZ) / 2));
    zslice = min(size(viewA,3), size(viewB, 3));
    zslice = min(cutZ+1, zslice);  % TODO: should prevent cutZ > zslice
    
    viewA = viewA(:,:,end-zslice+1:end-1);
    viewA = XxCrop3D(viewA, param.cut_x, param.cut_y, param.cut_z); 
    if param.rotR90
        viewA = rot90(viewA,3);
    end
    viewA = uint16(XxNorm(viewA) * 65535);
    XxWriteTiff(viewA, [filename(1:end-4) OutSuffix  '_A.tif']);
    
    viewB = viewB(:,:,end-zslice+1:end-1);
    viewB = XxCrop3D(viewB, param.cut_x, param.cut_y,  param.cut_z); 
    if param.rotR90
        viewB = rot90(viewB,3);
    end
    viewB = uint16(XxNorm(viewB) * 65535);
    XxWriteTiff(viewB, [filename(1:end-4) OutSuffix '_B.tif']);
    
    filename_rot = [filename(1:end-4) OutSuffix];
end

function [maxDispX, newMask, maxV] = OptDispX(mip, mask, deltaX, range)
    % crop canvas
    mip = mip(100+1:end-100,1:end);
    mask = mask(100+1:end-100,1:end);

    [Ny, Nx] = size(mip);
    maxDispX = 0;
    curVs = [];
    dispXs = [];
    maxRange = max(range(1), range(2));
    minRange = min(range(1), range(2));
    dispX = minRange;
    newMask = circshift(mask,[0, minRange]);
    while dispX < maxRange
        curMask = circshift(newMask,[0, deltaX]);
        curV = mean2(mip .* XxCrop(curMask,Ny,Nx));
        if dispX == minRange || curV > maxV
            maxV = curV;
            maxDispX = dispX;
        end
        newMask = curMask;
        dispX = dispX + deltaX;
        curVs = [curVs, curV];
        dispXs = [dispXs, dispX];
    end
%     fprintf('maxDisp=%d, xcorr=%.4f \n', maxDispX, maxV);
    figure(12), plot(dispXs, curVs);
end
