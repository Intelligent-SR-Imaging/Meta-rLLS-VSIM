clear;
addpath(genpath('./XxUtils'));

Current_path = pwd;
% flag of making training dataset
flag_make_training_data = false;
% flag of making finetuning dataset
flag_make_finetuning_data = true;
% raw data path
Data_path = '/Code_for_2D_IsoRecon/DemoData_for_VSI_SR_Finetune/Lattice-SIM';
Data_path = [Current_path Data_path];
% save path for processed LLSM

Save_path = '/Code_for_2D_IsoRecon/data';
Save_path = [Current_path Save_path];
save_path1 = [Save_path '/LLSM-LR'];
% save path for processed LLS-SIM
save_path2 = [Save_path  '/LLSIM-HR'];
% save path for meta training datset
save_path3 =  [Save_path '/train'];
% save path for finetuning dataset
save_path4 = [Save_path  '/finetune'];

% preprocessing LLSM data
Process_LLSM(data_path, save_path1);
% preprocessing LLS-SIM data
Process_LLSIM(data_path, save_path2);
% make training dataset
if flag_make_training_data
    make_meta_task(save_path1, save_path2, save_path3);
end
% make finetuning dataset
if flag_make_finetuning_data
    make_finetune_dataset(save_path1, save_path2, save_path4);
end
