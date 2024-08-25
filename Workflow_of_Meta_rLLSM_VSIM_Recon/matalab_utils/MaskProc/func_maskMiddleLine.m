function data_masked = func_maskMiddleLine(header, data, pParams)
    background = pParams.background;
    angle = pParams.rotAngle;
    
    % remove mid fluorescence
    pParams.dxy = single(typecast(header(11),'single'));
    pParams.dz = single(typecast(header(13),'single')) * sin(angle*pi/180);
    [mask, mask_boost, accdisX, accAngle] = RemoveMidFluorescence_v3(data-background, pParams);
    fprintf('Mask dispX: %.1f, angle: %.1f\n', accdisX, accAngle);
    data_masked = double(data-background) .* (1-mask);
        
    % boost virtual image
    mask_bi_up = mask_boost;
    mask_bi_up(mask_bi_up>0.9) = 1;
    mask_bi_up = 1-mask_bi_up;
    mask_bi_down = mask_boost;
    mask_bi_down(mask_bi_down<0.1) = 0;
    data_up = data_masked.*mask_bi_up;
    data_down = data_masked.*mask_bi_down;
    data_up = data_up(:,:,round(end/4):round(end/4*3));
    data_down = data_down(:,:,round(end/4):round(end/4*3));
    ratio = sum(data_up(:))/sum(data_down(:));
    ratio = max(1, ratio);
    fprintf('Intensity ratio of real image and virtual image is : %f\n',ratio);
    
%     if pParams.showFlag == 1
%         % display masks
%         figure(5), subplot(2,2,1), imshow(mean(mask_bi_up,3), []);
%         subplot(2,2,2), imshow(mean(mask_bi_down,3),[]);
%         subplot(2,2,3), imshow(mean(data_up,3), []);
%         subplot(2,2,4), imshow(mean(data_down,3),[]);
%         clear('mask_bi_up','mask_bi_down','data_up','data_down');
%     end
    
    % mask data
    mask_boost = 1 + mask_boost * (ratio - 1);
    data_masked = data_masked .* mask_boost;
    data_masked = data_masked + background;
    
end