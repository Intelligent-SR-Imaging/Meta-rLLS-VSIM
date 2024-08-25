%Step3: Split into dual views by rotation and registration
%Created on Tue Sep 30 19:31:36 2023
% Last Updated by Lin Yuhuan
addpath(genpath('./matalab_utils'));

Current_path = pwd;
% The path to data
Data_path ="/Demo_Data_for_3D_IsoRecon/step2/recon/0001/F-actin.tif";
Data_path = strcat(Current_path,Data_path);
%The path to save
Save_path = "/Demo_Data_for_3D_IsoRecon/step3";
Save_path = strcat(Current_path,Save_path);
%the design angle between the detection objective and coverslip
RotAngle_Y = 31.5;
%rotation range of the image around the X axis and Y axis during registration
Range_x_rot = -0.9:0.1:0.9;
Range_y_rot =  0.0:0.1:0.0;
%the split position range
range_z_split = [-30, 30];

% rotation parameters
param_rot = [];
param_rot.dxy = 0.0926 * 2 / 3;
param_rot.dz = 0.0926;
param_rot.RotAngle_y0 = RotAngle_Y;
param_rot.scale_x = 1.0;  % expand canvas for lossless rotation
param_rot.scale_y = 1.0;
param_rot.scale_z = 1.0;
param_rot.rot_x =Range_x_rot;
param_rot.rot_y = Range_y_rot;
param_rot.rangeZ = range_z_split;
param_rot.cutZ = 101;
param_rot.cut_x = [0.0 1.0]; % crop canvas to reduce storage
param_rot.cut_y = [0.0 1.0];
param_rot.cut_z = [0.0 1.0];
param_rot.rotR90 = 3;
param_rot.verbose = 1;
%read data
data_isoxy = XxReadTiff(Data_path);

if ~exist(Save_path,'dir'), mkdir(Save_path); end
[~,dataFile,ext] = fileparts(Data_path);
saveFile = fullfile(Save_path, [dataFile,ext]);
saveFile = saveFile{1};


% rotation and split views
[viewA,viewB,file_rot] = isolattice_rotate(data_isoxy, param_rot, saveFile, 0);
