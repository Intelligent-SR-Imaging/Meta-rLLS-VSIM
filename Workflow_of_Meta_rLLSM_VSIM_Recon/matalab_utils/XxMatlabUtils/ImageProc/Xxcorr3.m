function [xshift, yshift, zshift] = Xxcorr3(img,temp)
% the shifts of img to temp
%  conv2(img1,img2(end:-1:1,end:-1:1) == normxcorr2(img2, img1)
   tc = convn(img, temp(end:-1:1,end:-1:1,end:-1:1));
   [yp, xp, zp] = find(tc==max(tc(:)));
   yshift = size(temp,1) - yp;
   xshift = size(temp,2) - xp;
   zshift = size(temp,3) - zp;
   % shift the img
%   img = imtranslate(img,[xshift yshift zshift],'FillValues',0);
end