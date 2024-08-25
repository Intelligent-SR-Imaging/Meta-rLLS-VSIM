%Step1: Remove Background Fluorescence and DeSkew
%Created on Tue Sep 30 19:31:36 2023
% Last Updated by Lin Yuhuan
clc, clear;
addpath(genpath('./matalab_utils'));

Current_path = pwd
%which channel your data use
channel = '488';
%what kind your data is
celllabel = 'F-actin/';
% The path to data
Data_Path = '/Demo_Data_for_3D_IsoRecon/data/';
DataName = 'rLLSM_488_Cyc1_Ch1_St5.mrc';
Filepath = [pwd,Data_Path,celllabel,DataName];
%The path to save
Save_path='/Demo_Data_for_3D_IsoRecon/step1/';
Save_path = [Current_path  Save_path];
SaveDir = [Save_path celllabel '/' channel];
if ~exist(SaveDir,'dir'), mkdir(SaveDir); end

pParams = Parameters_Lifeact;

% remove background fluorescence and deSkew
for i = 1:1:1
    fprintf('Processing %s files ...\n', Filepath);
    
    % deskew and remove background
    [header, data] = XxReadMRC(Filepath);
    data = reshape(data, header(1), header(2), header(3));
    data_deskew = PreProcess(data, header, pParams);
    data_deskew = data_deskew(round((size(data_deskew,1)-size(data_deskew,2))/2):round((size(data_deskew,1)+size(data_deskew,2))/2)-1,:,:);
    
    mipxy(:,:,i)=max(data_deskew,[],3);
    mipxz(:,:,i)=squeeze(max(data_deskew,[],1));
    mipyz(:,:,i)=squeeze(max(data_deskew,[],2));
    
    % save preprocessed data
    outFile = [SaveDir '/' num2str(i,'%.3d') '.tif'];
    XxWriteTiff(uint16(data_deskew), outFile); 
end