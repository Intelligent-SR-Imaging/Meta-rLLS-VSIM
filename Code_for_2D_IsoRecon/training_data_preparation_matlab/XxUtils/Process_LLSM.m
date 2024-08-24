function Process_LLSM(DataPath, SavePath)
% addpath(genpath('./XxUtils'));

StruPath = dir(DataPath);
for i = 3:length(StruPath)
    fprintf('Processing structure %s\n', StruPath(i).name);
    stru = [StruPath(i).folder '/' StruPath(i).name '/'];

    Cell = dir(stru);
    % parfor j = 3:length(Cell)
    for j = 3:length(Cell)
        fprintf('Processing cell %s\n', Cell(j).name);
%         mrcList= cell(1,5);
        CellPath = [Cell(j).folder '/' Cell(j).name '/'];
        % here each cell has 5 different SNR images
%         mrcList{1} = char(XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St1.mrc')));
%         mrcList{2} = char(XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St2.mrc')));
%         mrcList{3} = char(XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St3.mrc')));
%         mrcList{4} = char(XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St4.mrc')));
%         mrcList{5} = char(XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St5.mrc')));
        
        mrcList = XxSort(XxDir(CellPath,'Illum*_Cyc1_Ch*_St*.mrc'));
        
        savepath = [SavePath '/' StruPath(i).name '/' Cell(j).name];
        if ~exist(savepath, 'dir')
            mkdir(savepath)
        end
        
        % data processing
        for k=1:1:(length(mrcList)-1)
            fprintf('Processing SNR %s\n', mrcList{k});
            % read raw-LLSM data
            [header, data] = XxReadMRC(mrcList{k});
            data = reshape(data, [header(1), header(2), header(3)]);
            data = single(data);
            data1 = zeros(size(data,1),size(data,2),size(data,3)/3);
            % generate LLSM-WF images from raw-LLSM images
            for l=1:size(data, 3)/3
                data1(:,:,l) = (data(:,:,(l-1)*3+1)+data(:,:,(l-1)*3+2)+data(:,:,(l-1)*3+3))/3;
            end
            % deskew LLSM-WF
            data_deskew = func_deskew(header, data1, 30.8, 0 );
            % subtract background noise
            data_deskew = data_deskew-100;
            data_deskew(data_deskew<0) = 0;
            % normalize and save the LLSM-WF images
            data_deskew = XxNorm(data_deskew);
            data_deskew = uint16(data_deskew*65535);
            XxWriteTiff(data_deskew, [savepath filesep num2str(k) '.tif']);
            
        end
    end
end
end