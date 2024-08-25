function data_deskew = PreProcess(data, header, params)
% remove middle flur
data_masked = func_maskMiddleLine(header,data,params);

% data read from mrc files, has shape (nx, ny, nz)
data_deskew = func_deskew(header,data_masked, params.rotAngle, 0);
data_deskew = data_deskew - params.background;
data_deskew(data_deskew<0) = 0;
data_deskew = XxNorm(data_deskew);
data_deskew = uint16(data_deskew*65535);
end