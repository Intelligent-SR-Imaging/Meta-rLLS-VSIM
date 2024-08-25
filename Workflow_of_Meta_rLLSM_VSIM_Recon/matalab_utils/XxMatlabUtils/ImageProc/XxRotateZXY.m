function output = XxRotateZXY(inArray, angle_y, angle_x, angle_z)
% in matlab [y(h), x(w), z]
angleRad = angle_y*pi/180;
T4 = [1 0             0              0
      0 cos(angleRad) -sin(angleRad) 0
      0 sin(angleRad) cos(angleRad)  0
      0 0             0              1];


angleRad = angle_x*pi/180;
T6 = [cos(angleRad) 0 -sin(angleRad) 0
      0 1 0 0
      sin(angleRad) 0 cos(angleRad)   0
      0 0 0 1];
  
 
angleRad = angle_z*pi/180;
T8 = [cos(angleRad) -sin(angleRad) 0 0
      sin(angleRad) cos(angleRad) 0 0
      0 0 1 0
      0 0 0 1];
  
T = T8 * T6 * T4;

tform = affine3d(T);
% output = imwarp(inArray, tform, 'OutputView', imref3d(size(inArray)));
output = imwarp(inArray, tform);

end