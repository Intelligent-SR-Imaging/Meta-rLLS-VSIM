function make_finetune_dataset(LR_path, SR_path, save_path, stru_selected, snr_selected, test_cell_selected, SegX, SegY, SegZ, num_seg)
% select a specific task (i.e. a specific structure with a specific SNR for
% finetuning)
if nargin < 4, stru_selected = 1; end
if nargin < 5, snr_selected = 1; end
% select a test cell from 3 cells (1 or 2 or 3), and the other two cells are used for
% finetuning
if nargin < 6, test_cell_selected = 3; end
% segmentation size and number for finetuning
if nargin < 7, SegX = 64; end
if nargin < 8, SegY = 64; end
if nargin < 9, SegZ = 8; end
if nargin < 10, num_seg = 500; end

Save_path = [save_path '/train'];
save_test_path_input = [save_path '/test/input'];
save_test_path_gt = [save_path '/test/gt'];
save_path_input = [Save_path '/input'];
if ~exist(save_path_input,'dir')
    mkdir(save_path_input)
end
save_path_gt = [Save_path '/gt'];
if ~exist(save_path_gt,'dir')
    mkdir(save_path_gt)
end
if ~exist(save_test_path_gt,'dir')
    mkdir(save_test_path_gt)
end
if ~exist(save_test_path_input,'dir')
    mkdir(save_test_path_input)
end

stru_gt = dir(SR_path);
stru_lr = dir(LR_path);


% segmentation size and number for testing
test_whole_cell = true;
upscale = 1.5;
SegX_test = 384;
SegY_test = SegX_test;
SegZ_test = 30;
num_seg_test = 1;

n_total_test = 0;
n_total = 0;
for i=3:length(stru_gt)
    if i~=(stru_selected+2)
        continue
    end
    fprintf('Processing structures %s\n', stru_gt(i).name);
    stru_gt_path = [stru_gt(i).folder '/' stru_gt(i).name '/'];
    cell_gt = dir(stru_gt_path);
    stru_lr_path = [stru_lr(i).folder '/' stru_gt(i).name '/'];
    cell_lr = dir(stru_lr_path);
    cell_lr_path = [cell_lr(3).folder '/' cell_gt(3).name '/'];
    tifList_lr = XxSort(XxDir(cell_lr_path,'*.tif'));
    for t = 1:length(tifList_lr)
        if t~=snr_selected
            continue;
        end
        for j=3:length(cell_gt)
            fprintf('Processing cells %s\n', cell_gt(j).name);
            cell_gt_path = [cell_gt(j).folder '/' cell_gt(j).name '/'];
            cell_lr_path = [cell_lr(j).folder '/' cell_gt(j).name '/'];
            tifList_gt = XxSort(XxDir(cell_gt_path,'gt.tif'));
            tifList_lr = XxSort(XxDir(cell_lr_path,'*.tif'));
            data_gt = XxReadTiff(tifList_gt{1});
            data_gt = single(data_gt);
            data_gt = data_gt / 65535;
            
            data_lr = XxReadTiff(tifList_lr{t});
            data_lr = single(data_lr);
            data_lr = data_lr / 65535;
            
            
            
            % save patches
            if  ismember(j, test_cell_selected+2)   % test set               
                if test_whole_cell
                    if mod(size(data_lr,2)-4,2)==0
                        Nx = size(data_lr,2)-4;
                    else
                        Nx = size(data_lr,2)-4-1;
                    end
                    if mod(size(data_lr,1)-4,2)==0
                        Ny = size(data_lr,1)-4;
                    else
                        Ny = size(data_lr,1)-4-1;
                    end
                    halfx = round(Nx/2);
                    halfy = round(Ny/2);
                    halfx_h = round(Nx*upscale/2);
                    halfy_h = round(Ny*upscale/2);
                    y1 = round(round(size(data_gt,1)/2)/upscale) - halfy + 1;
                    y2 = round(round(size(data_gt,1)/2)/upscale) + halfy;
                    x1 = round(round(size(data_gt,2)/2)/upscale) - halfx + 1;
                    x2 = round(round(size(data_gt,2)/2)/upscale) + halfx;
                    y1_h = round(size(data_gt,1)/2) - halfy_h + 1;
                    y2_h = round(size(data_gt,1)/2) + halfy_h;
                    x1_h = round(size(data_gt,2)/2) - halfx_h + 1;
                    x2_h = round(size(data_gt,2)/2) + halfx_h;
                    
                    cur_data = uint16(65535*data_lr(y1:y2,x1:x2,:));
                    cur_gt = uint16(65535*data_gt(y1_h:y2_h,x1_h:x2_h,:));
                    XxWriteTiff(cur_data, [save_test_path_input filesep num2str(n_total_test, '%.8d') '.tif']);
                    XxWriteTiff(cur_gt, [save_test_path_gt filesep num2str(n_total_test, '%.8d')   '.tif']);               
                    
                else
                    [data_seg_test,gt_seg_test] = XxDataSeg_ForTrain_Lattice_3D(data_lr, data_gt, num_seg_test, SegX_test, SegY_test, SegZ_test);
                    cur_data = uint16(65535 * data_seg_test(:,:,:,1));
                    n_total_test = n_total_test + 1;
                    XxWriteTiff(cur_data, [save_test_path_input filesep num2str(n_total_test, '%.8d') '.tif']);
                    cur_gt = uint16(65535 * gt_seg_test(:,:,:,1));
                    XxWriteTiff(cur_gt, [save_test_path_gt filesep num2str(n_total_test, '%.8d')   '.tif']);
                end
                
            else   % finetuning set
                
                % segmentation
                [data_seg,gt_seg] = XxDataSeg_ForTrain_Lattice_3D(data_lr, data_gt, num_seg, SegX, SegY, SegZ);
                for l=1:size(data_seg,4)
                    % size of LLSM (7,h,w)
                    cur_data = uint16(65535 * data_seg(:,:,1:7,l));
                    n_total = n_total + 1;
                    XxWriteTiff(cur_data, [save_path_input filesep num2str(n_total, '%.8d') '.tif']);
                    % size of LLSIM (3,h,w)
                    cur_gt = uint16(65535 * gt_seg(:,:,3:5,l));
                    XxWriteTiff(cur_gt, [save_path_gt filesep num2str(n_total, '%.8d')   '.tif']);
                end
            end
            
            
            
        end
    end
end
end