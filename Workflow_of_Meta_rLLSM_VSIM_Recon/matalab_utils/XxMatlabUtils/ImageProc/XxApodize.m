% average the value of edge to soften the edge of a singel xy section to
% reduce edge artifacts of fft

function apoimage = XxApodize(napodize, image)

[ny,nx,nz] = size(image);

if nz > 1
    apoimage = image;
    imageUp = image(1:napodize,:,:);
    imageDown = image(ny-napodize+1:ny,:,:);
    imageLeft = apoimage(:,1:napodize,:);
    imageRight = apoimage(:,nx-napodize+1:nx,:);
    for z0 = 1:nz
        diff = (flipud(imageDown(:,:,z0))-imageUp(:,:,z0))/2;
        l = (0:1:napodize-1)';
        fact = 1-sin((l+0.5)/napodize*pi/2);
        fact = repmat(fact, [1,nx]);
        factor = diff.*fact;
        apoimage(1:napodize,:,z0) = imageUp(:,:,z0)+factor;
        apoimage(ny-napodize+1:ny,:,z0) = imageDown(:,:,z0)-flipud(factor);
        
        diff = (fliplr(imageRight(:,:,z0))-imageLeft(:,:,z0))/2;
        l = (0:1:napodize-1);
        fact = 1-sin((l+0.5)/napodize*pi/2);
        fact = repmat(fact, [ny,1]);
        factor = diff.*fact;
        apoimage(:,1:napodize,z0) = imageLeft(:,:,z0)+factor;
        apoimage(:,nx-napodize+1:nx,z0) = imageRight(:,:,z0)-fliplr(factor);
    end
else
    apoimage = image;
    imageUp = image(1:napodize,:);
    imageDown = image(ny-napodize+1:ny,:);
    diff = (flipud(imageDown)-imageUp)/2;
    l = (0:1:napodize-1)';
    fact = 1-sin((l+0.5)/napodize*pi/2);
    fact = repmat(fact, [1,nx]);
    factor = diff.*fact;
    apoimage(1:napodize,:) = imageUp+factor;
    apoimage(ny-napodize+1:ny,:) = imageDown-flipud(factor);
    
    imageLeft = apoimage(:,1:napodize);
    imageRight = apoimage(:,nx-napodize+1:nx);
    diff = (fliplr(imageRight)-imageLeft)/2;
    l = (0:1:napodize-1);
    fact = 1-sin((l+0.5)/napodize*pi/2);
    fact = repmat(fact, [ny,1]);
    factor = diff.*fact;
    apoimage(:,1:napodize) = imageLeft+factor;
    apoimage(:,nx-napodize+1:nx) = imageRight-fliplr(factor);
end


