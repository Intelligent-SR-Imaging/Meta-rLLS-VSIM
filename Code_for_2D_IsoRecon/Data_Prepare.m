clear;

addpath(genpath('./XxUtils'));
% flag of making training dataset
flag_make_training_data = true;
% flag of making finetuning dataset
flag_make_finetuning_data = false;
Current_path = pwd;

% make training dataset
if flag_make_training_data
    Save_path = '/Code_for_2D_IsoRecon/data/train';
    Save_path = [Current_path Save_path];
    
    % raw data path
    data_path = '/Code_for_2D_IsoRecon/BioSR_for_LLS-SIM';
    data_path = [Current_path data_path];
    % save path for processed LLSM
    LLSM_name = 'Raw_LLS_SIM_SNR*.mrc';
    save_path1 = [Save_path '/LLSM-LR'];
    % save path for processed LLS-SIM
    LLS_SIM_name = 'GT_LLS_SIM.mrc';
    save_path2 = [Save_path  '/LLSIM-HR'];
    % save path for meta training datset
    save_path3 =  [Save_path '/train'];
    % preprocessing LLSM data
    Process_LLSM(data_path, save_path1, LLSM_name);
    % preprocessing LLS-SIM data
    Process_LLSIM(data_path, save_path2, LLS_SIM_name);
    make_meta_task(save_path1, save_path2, save_path3);
end

% make finetuning dataset
if flag_make_finetuning_data
    % raw data path
    Save_path = '/Code_for_2D_IsoRecon/data';
    Save_path = [Current_path Save_path];
    
    data_path = '/Code_for_2D_IsoRecon/Demo_Data_for_VSI_SR/Lattice-SIM';
    data_path = [Current_path data_path];
    % save path for processed LLSM
    LLSM_name1 = 'Illum488_Cyc1_Ch1_St4.mrc';   % for meta finetuning
    save_path1 = [Save_path '/LLSM-LR'];
    % save path for processed LLS-SIM
    LLS_SIM_name1 = 'Illum488_Cyc1_Ch1_St5-wiener0.02-fixk0-Fd.mrc';  % for meta finetuning
    save_path2 = [Save_path  '/LLSIM-HR'];
    % save path for finetuning dataset
    save_path4 =  [Save_path '/finetune'];
    
    % preprocessing LLSM data
    Process_LLSM(data_path, save_path1, LLSM_name1);
    % preprocessing LLS-SIM data
    Process_LLSIM(data_path, save_path2, LLS_SIM_name1);
    make_finetune_dataset(save_path1, save_path2, save_path4);
end
