from scipy.ndimage import gaussian_filter
import numpy as np
import tifffile as tiff
import os


def make_data_tensor_sr_2d(tasks_path, meta_batch, task_batch, spt_num, flag_filter=True, sigma=0.6):
    input_meta = []
    label_meta = []
    tasks_dir = os.listdir(tasks_path)
    print(tasks_dir,tasks_path)
    sampled_tasks = np.random.choice(tasks_dir, size=meta_batch, replace=False)

    for t in range(meta_batch):
        input_task = []
        label_task = []
        fn = os.listdir(tasks_path + '/' + sampled_tasks[t] + '/input')
        selected_fn = np.random.choice(fn, size=task_batch, replace=False)

        for i in range(task_batch):
            image = tiff.imread(tasks_path + '/' + sampled_tasks[t] + '/input/' + selected_fn[i])
            # print('lr path:', tasks_path + '/' + sampled_tasks[t] + '/input/' + selected_fn[i])
            image = np.array(image) / 65535
            if len(image.shape) < 3:
                image = image[np.newaxis, ...]
            input_task.append(image)
            image_sr = tiff.imread(tasks_path + '/' + sampled_tasks[t] + '/gt/' + selected_fn[i])
            # print('sr path:', tasks_path + '/' + sampled_tasks[t] + '/gt/' + selected_fn[i])
            image_sr = np.array(image_sr) / 65535
            if len(image_sr.shape) < 3:
                image_sr = image_sr[np.newaxis, ...]
            label_task.append(image_sr)
        input_meta.append(np.asarray(input_task))
        label_meta.append(np.asarray(label_task))

    input_meta = np.asarray(input_meta)
    label_meta = np.asarray(label_meta)
    inputa = input_meta[:, :spt_num, ...]
    labela = label_meta[:, :spt_num, ...]
    inputb = input_meta[:, spt_num:, ...]
    labelb = label_meta[:, spt_num:, ...]
    # meta_bs, task_bs, ch, h, w
    if flag_filter:
        for i in range(labela.shape[0]):
            for j in range(labela.shape[1]):
                for k in range(labela.shape[2]):
                    labela[i, j, k, ...] = gaussian_filter(labela[i, j, k, ...], sigma=sigma)

    if flag_filter:
        for i in range(labelb.shape[0]):
            for j in range(labelb.shape[1]):
                for k in range(labelb.shape[2]):
                    labelb[i, j, k, ...] = gaussian_filter(labelb[i, j, k, ...], sigma=sigma)

    return inputa, labela, inputb, labelb


def make_batch_finetune_data(image_path, image_dir, gt_path, batch, flag_filter=1, sigma=0.6):
    inputs = []
    labels = []

    fs = np.random.choice(image_dir, batch, replace=False)
    for i in range(batch):
        lr_image_path = image_path + '/' + fs[i]
        # print(lr_image_path)
        lr_image = tiff.imread(lr_image_path)
        lr_image = np.array(lr_image) / 65535
        sr_image_path = gt_path + '/' + fs[i]
        # print(sr_image_path)
        sr_image = tiff.imread(sr_image_path)
        sr_image = np.array(sr_image) / 65535
        if len(sr_image.shape) < 3:
            sr_image = sr_image[np.newaxis, ...]
        if len(lr_image.shape) < 3:
            lr_image = lr_image[np.newaxis, ...]
        if flag_filter:
            for j in range(sr_image.shape[0]):
                sr_image[j, ...] = gaussian_filter(sr_image[j, ...], sigma)

        inputs.append(lr_image)
        labels.append(sr_image)
    # bs,ch,h,w
    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels



# if __name__ == '__main__':

