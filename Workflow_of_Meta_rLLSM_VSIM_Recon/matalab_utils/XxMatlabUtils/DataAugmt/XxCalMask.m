function mask = XxCalMask(img, ksize, thresh, mode)

if nargin < 4, mode = 1; end
kernel = fspecial('gaussian',[ksize,ksize],ksize);

if mode == 1
    mask = imfilter(img,kernel,'replicate');
    mask(mask >= thresh) = 1;
    mask(mask ~= 1) = 0;
    mask = logical(mask);
elseif mode == 2
    fd = imfilter(img,kernel,'replicate');
    kernel = fspecial('gaussian',[100,100],50);
    bg = imfilter(img,kernel,'replicate');
    mask = fd - bg;
    mask(mask >= thresh) = 1;
    mask(mask ~= 1) = 0;
    mask = logical(mask);
end

end