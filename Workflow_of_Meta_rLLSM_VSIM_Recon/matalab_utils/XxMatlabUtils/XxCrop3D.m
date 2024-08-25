function data_crop = XxCrop3D(data, cut_x, cut_y, cut_z)
    [Nx, Ny, Nz] = size(data);
    
    % crop data, defined by cut_xyz
    x = max(round(cut_x * double(Nx)), 1);                                                                                            
    y = max(round(cut_y * double(Ny)), 1);
    z = max(round(cut_z * double(Nz)), 1);
    data_crop = data(x(1):x(2), y(1):y(2), z(1):z(2));
end
