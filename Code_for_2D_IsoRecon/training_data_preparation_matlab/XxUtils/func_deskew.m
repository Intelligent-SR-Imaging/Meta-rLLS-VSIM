function res2 = func_deskew(header, inArray2, angle, background)
inArray2=single(inArray2);
% inArray2=gpuArray(inArray2);
dy = single(typecast(header(11),'single'));  %  X-Y pixel size (in um) in real space
dz = single(typecast(header(13),'single'));%  Z pixel size (in um) in real space
dz = double(dz);
dy = double(dy);

sz = ceil(dz*size(inArray2,3)*cos(angle*pi/180)/dy + size(inArray2,1));
res2 = zeros([sz size(inArray2,2) size(inArray2,3)],'single')*background;
%%插值法平移
shift = ((1:size(inArray2,3))'-(size(inArray2,3)+1)/2)*dz/dy*cos(angle*pi/180)-(size(res2,1)-size(inArray2,1))/2;
[x1,y1]=meshgrid(1:single(size(inArray2,2)),1:single(size(inArray2,1)));
[x2,y2]=meshgrid(1:single(size(res2,2)),1:single(size(res2,1)));
for ii=1:size(inArray2,3)
    res2(:,:,ii)=interp2(x1,y1,inArray2(:,:,ii),x2,y2+shift(ii),'cubic',0);
end
end