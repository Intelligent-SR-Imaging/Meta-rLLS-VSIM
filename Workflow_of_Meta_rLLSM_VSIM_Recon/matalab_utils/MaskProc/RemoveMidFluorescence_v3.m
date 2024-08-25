function [Mask, Mask_boost, accdisX, accAngle] = RemoveMidFluorescence_v3(data, pParams)
% v2: add a larger mask along y-axis to balance the intensity
% v3: calculate different mask along z-axis; enhance the intensity of virtual image

%% parameters
if nargin < 2
    pParams.gsWindow = 15; % pxls
    pParams.scaleFactor = 5; 
    pParams.sigama = 0.2; 
    pParams.deltaDispX = 1; % pxls
    pParams.deltaDispA = 0.1; % degrees
    pParams.modAmp = 0.3;
    pParams.nAverage = 5;
    pParams.maxIters = 5;
    pParams.showFlag = 1;
end
if nargin < 1
    DataDir = './DataForTest';
    mrcList = XxDir(DataDir, '*.mrc');
    [header, data] = XxReadMRC(mrcList{3});
    data = reshape(data, [header(1), header(2), header(3)]);
end

%% initialization
[Ny, Nx, Nz] = size(data);
MIP = mean(double(data), 3);
scale = 1.2;
initialMask = zeros(round(Ny*scale), round(Nx*scale));
Mask_boost = zeros(round(Ny*scale), round(Nx*scale));
Mask_boost(ceil(round(Ny*scale)/2):end,:) = 1;
initialMask(ceil(round(Ny*scale)/2),:) = 1;
filter_a = fspecial('gaussian',[pParams.gsWindow*10,1],pParams.gsWindow);
initialMask_a = XxNorm(imfilter(initialMask,filter_a));
filter_b = fspecial('gaussian',[pParams.gsWindow*pParams.scaleFactor*10,1],...
    pParams.gsWindow*pParams.scaleFactor);
initialMask_b = XxNorm(imfilter(initialMask,filter_b));

F_MIP = fft2(MIP);
F_Mask = fft2(XxCrop(initialMask_a,Ny,Nx));
crosscorr = abs(fftshift(ifft2(conj(F_MIP) .* F_Mask)));

cc = sum(crosscorr,2);
my = find(cc == max(cc));
center = round(Ny / 2);
offset = my - (center+1);
Mask_boost = circshift(Mask_boost,[-offset, 0]);
initialMask_a = circshift(initialMask_a,[-offset, 0]);
initialMask_b = circshift(initialMask_b,[-offset, 0]);

%% 
accdisX = -offset;
accAngle = 0;


%% initial optimization
Mask_a = initialMask_a;
Mask_b = initialMask_b;
if pParams.showFlag == 1
    figure(1); subplot(1,pParams.maxIters+1,1), imshow(Mask_a,[]);
    title('initial estimation');
end

for i = 1:pParams.maxIters
    % optimize dispX
    [curdx, Mask_a] = OptDispX(MIP, Mask_a, pParams.deltaDispX);
    Mask_b = circshift(Mask_b,[curdx, 0]);
    Mask_boost = circshift(Mask_boost,[curdx, 0]);
    % optimize dispAngle
    [curdAngle, Mask_a] = OptDispAngle(MIP, Mask_a, pParams.deltaDispA);
    Mask_b = imrotate(Mask_b,curdAngle,'bilinear','crop');
    Mask_boost = imrotate(Mask_boost,curdAngle,'bilinear','crop');
    if pParams.showFlag == 1
        subplot(1,pParams.maxIters+1,i+1), imshow(Mask_a,[]);
        title(['iter=' num2str(i) ...
            ', dx=' num2str(curdx) ', dA=' num2str(curdAngle)]);
    end
    if curdx == 0 && curdAngle == 0, break; end

    accdisX = accdisX + curdx;
    accAngle = accAngle + curdAngle;
end

%% linear regression
nAverage = pParams.nAverage;
nGroups = floor(Nz / nAverage);
data_ave = data(:,:,1:nGroups*nAverage);
data_ave = squeeze(mean(reshape(data_ave,[Ny,Nx,nAverage,nGroups]),3));
shift_y = zeros(nGroups,1);
F_Mask = XxCrop(Mask_a,Ny,Nx);
for i = 1:nGroups
    F_img = fft2(data_ave(:,:,i));
    crosscorr = abs(fftshift(ifft2(conj(F_img) .* F_Mask)));
    cc = sum(crosscorr,2);
%     cc(Ny/4:end) = 0;
    shift_y(i) = find(cc == max(cc))-1;
end
ind = shift_y>round(Ny/2);
shift_y(ind) = shift_y(ind)-Ny;
ind = (abs(shift_y) < 50);
xx = (nAverage+1)/2:nAverage:Nz;
p = polyfit(xx(ind)',shift_y(ind),1);
shift_y = round(polyval(p,1:Nz));
shift_y = shift_y - shift_y(round(Nz/2));

%% calculate final mask
Mask_boost = XxNorm(imfilter(Mask_boost,filter_a));
Mask = XxNorm(XxNorm(Mask_a)*pParams.sigama + XxNorm(Mask_b));
Mask = Mask * (1-2*pParams.modAmp);
Mask_boost = repmat(Mask_boost,[1,1,Nz]);
Mask = repmat(Mask,[1,1,Nz]);
for i = 1:Nz
    Mask_boost(:,:,i) = circshift(Mask_boost(:,:,i),[-shift_y(i), 0]);
    Mask(:,:,i) = circshift(Mask(:,:,i),[-shift_y(i), 0]);
end
Mask_boost = XxCrop(Mask_boost,Ny,Nx);
Mask = XxCrop(Mask,Ny,Nx);

%% display results
if pParams.showFlag == 1
    MIP_Masked = max(double(data).*(1-Mask),[],3);
    figure(2);
    subplot(2,2,1), imshow(MIP,[]); title('MIP');
    subplot(2,2,2), imshow(initialMask_a,[]); title('initial mask');
    subplot(2,2,3), imshow(MIP_Masked,[]); title('masked MIP');
    subplot(2,2,4), imshow(max(Mask,[],3),[]); title('optimized mask');
end
end

%% functions
function [dispX, newMask] = OptDispX(mip, mask, deltaX)
[Ny, Nx] = size(mip);
maxV = mean2(mip .* XxCrop(mask,Ny,Nx));
newMask = mask;
dispX = 0;
while 1
    curMask = circshift(newMask,[deltaX, 0]);
    curV = mean2(mip .* XxCrop(curMask,Ny,Nx));
    if curV < maxV
        break;
    else
        maxV = curV;
        newMask = curMask;
        dispX = dispX + deltaX;
    end
end
while 1
    curMask = circshift(newMask,[-deltaX, 0]);
    curV = mean2(mip .* XxCrop(curMask,Ny,Nx));
    if curV < maxV
        break;
    else
        maxV = curV;
        newMask = curMask;
        dispX = dispX - deltaX;
    end
end
end

function [dispA, newMask] = OptDispAngle(mip, mask, deltaA, maxA)
if nargin < 4, maxA = 0.1*10; end
[Ny, Nx] = size(mip);

% % blank the border
% mip(:,1:round(Nx/10))=0;
% mip(:,end-round(Nx/10):end)=0;
% mip(1:round(Ny/10),:)=0;
% mip(end-round(Ny/10):end,:)=0;

maxV = mean2(mip .* XxCrop(mask,Ny,Nx));
newMask = mask;
initMask = mask;
dispA = 0;
while 1
    curMask = imrotate(initMask,dispA+deltaA,'bilinear','crop');
    curV = mean2(mip .* XxCrop(curMask,Ny,Nx));
    if curV < maxV
        break;
    else
        maxV = curV;
        newMask = curMask;
        dispA = dispA + deltaA;
    end
    if abs(dispA) > maxA
        dispA = dispA - deltaA; 
        break;
    end
end
while 1
    curMask = imrotate(initMask,dispA-deltaA,'bilinear','crop');
    curV = mean2(mip .* XxCrop(curMask,Ny,Nx));
    if curV < maxV
        break;
    else
        maxV = curV;
        newMask = curMask;
        dispA = dispA - deltaA;
    end
    if abs(dispA) > maxA
        dispA = dispA + deltaA;
        break;
    end
end
end


