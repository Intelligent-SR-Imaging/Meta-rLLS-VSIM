function make_meta_task(LR_path, SR_path, save_path, SegX, SegY, SegZ, num_seg)
% segmentation size and number for training
if nargin < 4, SegX = 64; end
if nargin < 5, SegY = 64; end
if nargin < 6, SegZ = 8; end
if nargin < 7, num_seg = 50; end
% segmentation size and number for testing
SegX_test = 256;
SegY_test = SegX_test;
SegZ_test = 8;
num_seg_test = 1;

Save_path = [save_path '/meta_train'];
save_val_path_input = [save_path '/meta_val/task1/input'];
save_val_path_gt = [save_path '/meta_val/task1/gt'];
save_test_path_input = [save_path '/meta_test/input'];
save_test_path_gt = [save_path '/meta_test/gt'];
if ~exist(save_test_path_gt,'dir')
    mkdir(save_test_path_gt)
end
if ~exist(save_test_path_input,'dir')
    mkdir(save_test_path_input)
end
if ~exist(save_val_path_gt,'dir')
    mkdir(save_val_path_gt)
end
if ~exist(save_val_path_input,'dir')
    mkdir(save_val_path_input)
end

stru_gt = dir(SR_path);
stru_lr = dir(LR_path);

task_num = 0;
n_total_test = 0;
% consider a structure as a meta task
for i=3:length(stru_gt)
    fprintf('Processing structures %s\n', stru_gt(i).name);
    stru_gt_path = [stru_gt(i).folder '/' stru_gt(i).name '/'];
    cell_gt = dir(stru_gt_path);
    stru_lr_path = [stru_lr(i).folder '/' stru_gt(i).name '/'];
    cell_lr = dir(stru_lr_path);
    cell_lr_path = [cell_lr(3).folder '/' cell_gt(3).name '/'];
    tifList_lr = XxSort(XxDir(cell_lr_path,'*.tif'));
    % here only consider SNR-2 and SNR-5 as a meta task
    for t = 1:1:length(tifList_lr)
        if i==4 && t==1
            task_num = task_num;
        else
            task_num = task_num+1;
        end
        save_path_input = [Save_path '/task' num2str(task_num, '%.3d') '/input'];
        if ~exist(save_path_input,'dir')
            mkdir(save_path_input)
        end
        save_path_gt = [Save_path '/task' num2str(task_num, '%.3d') '/gt'];
        if ~exist(save_path_gt,'dir')
            mkdir(save_path_gt)
        end
        n_total = 0;
        n_total_val = 0;
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
            
            % segmentation
            [data_seg,gt_seg] = XxDataSeg_ForTrain_Lattice_3D(data_lr, data_gt, num_seg, SegX, SegY, SegZ);
            
            % save patches
            if i==4 && t==1  % validation set
                if n_total_val<100
                    for l=1:size(data_seg,4)
                        % size of LLSM (7,h,w)
                        cur_data = uint16(65535 * data_seg(:,:,1:7,l));
                        n_total_val = n_total_val + 1;
                        XxWriteTiff(cur_data, [save_val_path_input filesep num2str(n_total_val, '%.8d') '.tif']);
                        % size of LLSIM (3,h,w)
                        cur_gt = uint16(65535 * gt_seg(:,:,3:5,l));
                        XxWriteTiff(cur_gt, [save_val_path_gt filesep num2str(n_total_val, '%.8d')   '.tif']);
                    end
                end
                
            elseif ismember(i,[3,6,5]) && t==1 && j==3   % test set
                
                [data_seg_test,gt_seg_test] = XxDataSeg_ForTrain_Lattice_3D(data_lr, data_gt, num_seg_test, SegX_test, SegY_test, SegZ_test);
                cur_data = uint16(65535 * data_seg_test(:,:,1:7,1));
                n_total_test = n_total_test + 1;
                XxWriteTiff(cur_data, [save_test_path_input filesep num2str(n_total_test, '%.8d') '.tif']);
                cur_gt = uint16(65535 * gt_seg_test(:,:,3:5,1));
                XxWriteTiff(cur_gt, [save_test_path_gt filesep num2str(n_total_test, '%.8d')   '.tif']);
                
            else   % training set
                
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
