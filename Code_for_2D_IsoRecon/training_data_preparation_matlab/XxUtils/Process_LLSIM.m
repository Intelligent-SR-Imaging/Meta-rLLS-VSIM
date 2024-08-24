function process_LLSIM(DataPath, SavePath)
StruPath = dir(DataPath);
% parpool('local',4);
for i = 3:length(StruPath)
    fprintf('Processing structure %s\n', StruPath(i).name);
    stru = [StruPath(i).folder '/' StruPath(i).name '/'];
    CellPath = dir(stru);
%     parfor j = 3:length(CellPath)
     for j = 3:length(CellPath)
        fprintf('Processing cell %s\n', CellPath(j).name);
        cellpath = [CellPath(j).folder '/' CellPath(j).name '/'];

        mrcList = XxDir(cellpath,'Illum*_Cyc1_Ch*_St5-wiener0.02-fixk0-Fd.mrc');
        if isempty(mrcList{1})
            mrcList = XxDir(cellpath,'Illum*_Cyc1_Ch*_St5-wiener0.03-fixk0-Fd.mrc');
        end
        
        if isempty(mrcList{1})
            mrcList = XxDir(cellpath,'Illum*_Cyc1_Ch*_St5-wiener0.05-fixk0-Fd.mrc');
        end
        
        if isempty(mrcList{1})
            mrcList = XxDir(cellpath,'Illum*_Cyc1_Ch*_St5-wiener0.02-fixk0.mrc');
        end
        
        if isempty(mrcList{1})
            mrcList = XxDir(cellpath,'Illum*_Cyc1_Ch*_St5-wiener0.03-fixk0.mrc');
        end
        if isempty(mrcList{1})
            continue;
        end
        
        save_deskew_path = [SavePath '/' StruPath(i).name '/' CellPath(j).name '/'];
        if ~exist(save_deskew_path, 'dir'), mkdir(save_deskew_path); end
        
        % data processing
        for t = 1:length(mrcList)
            fprintf('Processing LLSIM %s\n', mrcList{t});
            % read the LLSIM data
            [header, data] = XxReadMRC(mrcList{t});
            data = reshape(data, [header(1), header(2), header(3)]);
            % deskew LLSIM data
            data_deskew = func_deskew(header, data, 30.8, 0 );
            % ignore the negative values of LLSIM
            data_deskew(data_deskew<0) = 0;
            % normalize and save the LLSIM
            data_deskew = XxNorm(data_deskew);
            data_deskew = uint16(data_deskew*65535);
            XxWriteTiff(data_deskew, [save_deskew_path 'gt.tif']);

        end
    end
end
end