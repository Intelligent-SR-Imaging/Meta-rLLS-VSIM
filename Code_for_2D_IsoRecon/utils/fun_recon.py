from scipy.ndimage import rotate
import numpy as np
import tifffile as tiff
import os
from .utils import XxPrctileNorm, read_mrc
import torch
import cv2
import copy
import multiprocessing


def rotation(Prediction_File, model, device, save_path1, save_path2, tp):

    # loading model
    model = model.to(device)

    # loading data
    if '.tif' in Prediction_File:
        input_data = tiff.imread(Prediction_File)
    else:
        header, input_data = read_mrc(Prediction_File)
        input_data = input_data.swapaxes(0, 2)
    input_data = np.array(input_data).astype('float32')

    # input_data = input_data / 65535
    input_data = XxPrctileNorm(input_data)
    (nz, ny, nx) = input_data.shape
    end_slice = nz-3

    angle = np.linspace(0, 120, 3)
    outs = []
    out1 = []

    # rotation and prediction
    # 0 degree
    ci = 0
    for i in range(3, end_slice):
        print('Processing: %d slice' % (i-2))
        savepath2 = save_path2 + '/' + '%.7d' % (i-3+1 + (tp - 1)*(end_slice-3))
        if not os.path.exists(savepath2):
            os.makedirs(savepath2)

        input_data1 = input_data[i-3:i+4]
        origin = np.array(input_data1[3, ...]*65535).astype('uint16')
        # save original LLSM data
        tiff.imwrite(savepath2 + '/0_rot0.tif', origin)
        ci += 1

        for k in range(len(angle)):
            # rotation
            if angle[k] == 0:
                tile_data = input_data1[np.newaxis, ...]
                tile_data = torch.as_tensor(tile_data).type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    tile_results = model(tile_data).cpu().numpy()
                tile_results = np.squeeze(tile_results)
                # bs, 3, h, w
                tile_results = XxPrctileNorm(tile_results)
                outs.append(tile_results)

        # average repetitive frames to generate final outputs
        if ci==3:
            if i == 5:
                out1.append((outs[0][1]+outs[1][0])/2)
                out1.append((outs[0][2]+outs[1][1]+outs[2][0])/3)
            else:
                out1.append((outs[0][2]+outs[1][1]+outs[2][0])/3)
            outs.pop(0)
            ci -= 1

    out1.append((outs[1][1] + outs[0][2]) / 2)
    out1 = np.array(out1)
    # save single direction SR data
    for i in range(0, end_slice - 3):
        savepath1 = save_path1 + '/' + '%.7d' % (i + 1 + (tp - 1)*(end_slice-3))
        if not os.path.exists(savepath1):
            os.makedirs(savepath1)
        for k in range(len(angle)):
            if angle[k] == 0:
                tile_results = np.array(out1[i] * 65535).astype('uint16')
                tiff.imwrite(savepath1 + '/0_rot0.tif', tile_results)

    # 60 degree
    del outs, out1
    outs = []
    out1 = []
    ci = 0
    for i in range(3, end_slice):
        print('Processing: %d slice' % (i - 2))
        input_data1 = input_data[i - 3:i + 4]
        savepath2 = save_path2 + '/' + '%.7d' % (i - 3 + 1 + (tp - 1)*(end_slice-3))
        ci += 1
        for k in range(len(angle)):
            if angle[k] == 60:
                output_data = rotate(input_data1, angle[k], axes=(1, 2), reshape=True)
                # output_data1 = XxPrctileNorm(output_data[3, ...])*65535
                output_data1 = output_data[3, ...] * 65535
                output_data1[output_data1 > 65535] = 65535
                output_data1[output_data1 < 0] = 0
                output_data1 = np.array(output_data1).astype('uint16')
                # tiff.imwrite(savepath2 + '/' + str(k) + '_rot' + str(int(angle[k])) + '.tif', output_data1)
                output_data = output_data[np.newaxis, ...]
                output_data = torch.as_tensor(output_data).type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    output_data = model(output_data).cpu().numpy()
                output_data = np.squeeze(output_data)
                tile_results1 = output_data
                tile_results1 = XxPrctileNorm(tile_results1)
                outs.append(tile_results1)
        if ci==3:
            if i == 5:
                out1.append((outs[0][1]+outs[1][0])/2)
                out1.append((outs[0][2]+outs[1][1]+outs[2][0])/3)
            else:
                out1.append((outs[0][2]+outs[1][1]+outs[2][0])/3)
            outs.pop(0)
            ci -= 1

    out1.append((outs[1][1] + outs[0][2]) / 2)
    out1 = np.array(out1)
    # output images
    for i in range(0, end_slice - 3):
        savepath1 = save_path1 + '/' + '%.7d' % (i + 1 + (tp - 1)*(end_slice-3))
        if not os.path.exists(savepath1):
            os.makedirs(savepath1)
        for k in range(len(angle)):
            if angle[k] == 60:
                tile_results = np.array(out1[i] * 65535).astype('uint16')
                tiff.imwrite(savepath1 + '/' + str(k) + '_rot' + str(int(angle[k])) + '.tif', tile_results)

    # 120 degree
    del outs, out1
    outs = []
    out1 = []
    ci = 0
    for i in range(3, end_slice):
        print('Processing: %d slice' % (i - 2))
        input_data1 = input_data[i - 3:i + 4]
        savepath2 = save_path2 + '/' + '%.7d' % (i - 3 + 1 + (tp - 1)*(end_slice-3))
        ci += 1
        for k in range(len(angle)):
            if angle[k] == 120:
                output_data = rotate(input_data1, angle[k], axes=(1, 2), reshape=True)
                # output_data1 = XxPrctileNorm(output_data[3, ...])*65535
                output_data1 = output_data[3, ...] * 65535
                output_data1[output_data1 > 65535] = 65535
                output_data1[output_data1 < 0] = 0
                output_data1 = np.array(output_data1).astype('uint16')
                # tiff.imwrite(savepath2 + '/' + str(k) + '_rot' + str(int(angle[k])) + '.tif', output_data1)
                output_data = output_data[np.newaxis, ...]
                output_data = torch.as_tensor(output_data).type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    output_data = model(output_data).cpu().numpy()
                output_data = np.squeeze(output_data)
                tile_results1 = output_data
                tile_results1 = XxPrctileNorm(tile_results1)
                outs.append(tile_results1)
        if ci == 3:
            if i == 5:
                out1.append((outs[0][1] + outs[1][0]) / 2)
                out1.append((outs[0][2] + outs[1][1] + outs[2][0]) / 3)
            else:
                out1.append((outs[0][2] + outs[1][1] + outs[2][0]) / 3)
            outs.pop(0)
            ci -= 1

    out1.append((outs[1][1] + outs[0][2]) / 2)
    out1 = np.array(out1)
    # output images
    for i in range(0, end_slice - 3):
        savepath1 = save_path1 + '/' + '%.7d' % (i + 1 + (tp - 1)*(end_slice-3))
        if not os.path.exists(savepath1):
            os.makedirs(savepath1)
        for k in range(len(angle)):
            if angle[k] == 120:
                tile_results = np.array(out1[i] * 65535).astype('uint16')
                tiff.imwrite(savepath1 + '/' + str(k) + '_rot' + str(int(angle[k])) + '.tif', tile_results)

    del outs, out1
    return nz


def cal_avg(outs):
    outs1 = []
    for j in range(outs.shape[0]):
        if j == 0:
            outs1.append((outs[0][1] + outs[1][0]) / 2)
        elif j == outs.shape[0] - 1:
            outs1.append((outs[j][1] + outs[j - 1][2]) / 2)
        else:
            outs1.append((outs[j - 1][2] + outs[j][1] + outs[j + 1][0]) / 3)
    return outs1


def isotropic_recon(otfPath, save_path, param):
    if param['lambda'] == 560e-3:
        otfPath = os.path.join(otfPath, '560/')
    elif param['lambda'] == 642e-3:
        otfPath = os.path.join(otfPath, '642/')
    else:
        otfPath = os.path.join(otfPath, '488/')

    param['otf0'] = os.path.join(otfPath, 'OTF0.tif')
    param['otf_a0'] = os.path.join(otfPath, 'OTF1.tif')
    param['apo'] = os.path.join(otfPath, 'apo_0.9_1.5.tif')

    ImgList = np.sort(os.listdir(param['image_folder']))
    # ImgNum = len(ImgList)
    ImgNum = param['Nz']
    # TimePoints = ImgNum // param['Nz']
    band0 = tiff.imread(param['image_folder']+'/'+ImgList[0]+'/0_rot0.tif')
    [Ny, Nx] = band0.shape
    param['crop_y'] = Ny
    param['crop_x'] = Nx
    img_to_save1 = np.zeros((param['tp'], Ny, Nx))
    curParam = copy.deepcopy(param)

    p = multiprocessing.Pool(param['core_num'])
    res = [p.apply_async(recombined, (nn, param, curParam, ImgList, save_path)) for nn in range(param['tp'])]
    aug_results = [r.get()[1] for r in res]
    aug_num = [r.get()[0] for r in res]
    p.close()
    p.join()

    # img_to_save1 = np.array(aug_results)
    aug_num = np.array(aug_num)
    aug_results = np.array(aug_results)
    for nn in range(param['tp']):
        img_to_save1[aug_num[nn], ...] = aug_results[nn]

    img_to_save = np.uint16(65535 * XxPrctileNorm(img_to_save1))
    # img_to_save = img_to_save.swapaxes(0, 2)

    full_name = os.path.join(save_path, 'MIP-Recon_SR-f' + str(param['apo_factor']) + '-w' + str(param['otf_weight']))
    if param['use_origin_sr0']:
        full_name = full_name + '-useorisr'
    if param['adjust_intens']:
        full_name = full_name + '-adjustintens'
    tiff.imwrite(full_name+'.tif', img_to_save)


def recombined(nn, param, curParam, ImgList, save_path):
    print('Reconstruct timepoint %d/%d' % (nn + 1, param['tp']))
    # calculate z - axis intensity profile
    if param['adjust_intens'] == 1:
        Intens = np.zeros((1, param['Nz']))
        for t in range(nn, nn + 1):
            for z in range(param['Nz']):
                i = t * param['Nz'] + z
                curParam['wf'] = param['wf'] + '/' + ImgList[i] + '/0_rot0.tif'
                band0 = tiff.imread(curParam['wf'])
                thresh = np.percentile(band0, param['adjust_thresh'])
                Intens[0, z] = band0[band0 > thresh].mean()
            # Intens[0, :] = np.convolve(Intens[0, :], np.ones((10,)) / 10, 'same')
            Intens[0, :] = smooth(Intens[0, :], 11)
            Intens[0, :] = Intens[0, :] / Intens[0, :].max()

    # SIM assembly
    img_to_save11 = np.zeros((param['crop_y'], param['crop_x'], param['Nz']))
    for t in range(nn, nn + 1):
        for z in range(param['Nz']):
            i = t * param['Nz'] + z
            print('Reconstruct slice ' + str(i + 1) + '/' + str(int(param['Nz']*param['tp'])) + '...')
            curParam = copy.deepcopy(param)
            curParam['image_folder'] = os.path.join(param['image_folder'], ImgList[i])
            curParam['wf'] = os.path.join(param['wf'], ImgList[i]) + '/0_rot0.tif'
            curParam['origin_sr0'] = param['origin_sr0'] + '/00' + '%.3d' % i + '.tif'
            sr_img = Xx_recombine_singleslice_v2(curParam)
            if param['adjust_intens'] == 1:
                thresh = np.percentile(sr_img, param['adjust_thresh'])
                mean_value = sr_img[sr_img > thresh].mean()
                sr_img = Intens[0, z] * sr_img / mean_value
            img_to_save11[:, :, z] = sr_img

    img_to_save1 = img_to_save11.max(axis=2)
    img_to_save11 = np.uint16(65535 * XxPrctileNorm(img_to_save11))
    img_to_save11 = img_to_save11.swapaxes(0, 2)
    img_to_save11 = img_to_save11.swapaxes(1, 2)
    tiff.imwrite(os.path.join(save_path, param['save_name'] + '.tif'), img_to_save11)

    return nn, img_to_save1


def Xx_recombine_singleslice_v2(param):
    # v2 for random size assembly
    # combine seperated bands with Wiener filter

    # load setting parameters
    crop_x = param['crop_x']
    crop_y = param['crop_y']
    wf_sigma = param['wf_sigma']
    n_ang = param['n_ang']
    offset_ang = param['offset_ang']
    int_ang = np.uint8(180 / n_ang)

    isApoBand0 = param['isApoBand0']
    apo = tiff.imread(param['apo'])
    apo = apo**(param['apo_factor'])
    if not isApoBand0:
        apo = np.ones(apo.shape)
    apo = cv2.resize(apo, dsize=(crop_x, crop_y), interpolation=cv2.INTER_CUBIC)

    otf0 = np.double(tiff.imread(param['otf0'])) / 65535 * 2
    otf0 = cv2.resize(otf0, dsize=(crop_x, crop_y), interpolation=cv2.INTER_CUBIC)

    otf_a0 = np.double(tiff.imread(param['otf_a0'])) / 65535
    otf_a0 = rotate(otf_a0, 90)
    otf_weight = param['otf_weight']

    image_folder = param['image_folder']
    use_origin_sr0 = param['use_origin_sr0']

    # calculate band0
    # wf = np.double(readStack(param.wf, true))
    wf = np.double(tiff.imread(param['wf']))
    wf = cv2.resize(wf, dsize=(crop_x, crop_y), interpolation=cv2.INTER_CUBIC)
    sr_ft0 = np.fft.fftshift(np.fft.fft2(wf)) * otf0 / (otf0 ** 2 + wf_sigma ** 2)
    sr_ft0 = sr_ft0 * apo
    band0 = abs(np.fft.ifft2(np.fft.ifftshift(sr_ft0)))
    sr_ft0 = np.fft.fftshift(np.fft.fft2(wf)) * apo
    [x, y] = wf.shape

    # load network restored sr images
    sr_fts = np.zeros((n_ang, x, y), dtype=complex)
    imgfiles = np.sort(os.listdir(image_folder))
    for i in range(n_ang):
        if i == 0 and use_origin_sr0:
            img = np.double(tiff.imread(param['origin_sr0']))
        else:
            img = np.double(tiff.imread(os.path.join(image_folder,imgfiles[i])))
            img = rotate(img, -i * 60)
        if i == 0:
            im_mean = img.mean()
        img = XxCrop(img, crop_y, crop_x)
        img = 65535 * XxPrctileNorm(img)
        sr_fts[i,:,:] = np.fft.fftshift(np.fft.fft2(img))


    # generate wiener filter denominators
    wfd_full = otf0
    otfs = np.zeros((n_ang, x, y))
    wfds = np.zeros((n_ang, x, y))
    for i in range(n_ang):
        otf = rotate(otf_a0, -np.double(i * int_ang + offset_ang), reshape=False)
        otf = cv2.resize(otf, dsize=(crop_x, crop_y), interpolation=cv2.INTER_CUBIC)
        otfs[i,:,:] = otf
        wfds[i,:,:] = otf0 + otf_weight * otf
        wfd_full = wfd_full + otf
    wfd_full = wfd_full + wf_sigma ** 2

    # recombine bands
    sr_ft = np.zeros((x, y))
    for i in range(n_ang):
        sr_ft = sr_ft + np.squeeze(wfds[i,:,:] * sr_fts[i,:,:])
    sr_ft = sr_ft - (n_ang - 1) * otf0 * sr_ft0
    sr_ft = sr_ft / wfd_full
    sr_img = np.real(np.fft.ifft2(np.fft.ifftshift(sr_ft)))

    # clip to zero
    sr_img[sr_img < 0] = 0
    sr_img = XxPrctileNorm(sr_img)
    sr_img = im_mean * sr_img / sr_img.mean()
    return sr_img


def XxCrop(img, crop_y, crop_x):
    Ny, Nx = img.shape
    if Ny == crop_y and Nx == crop_x:
        return img
    midy = round(Ny / 2)-1
    midx = round(Nx / 2)-1

    rx = np.floor(crop_x / 2)
    ry = np.floor(crop_y / 2)

    if crop_y % 2 == 0:
        y1 = int(midy-ry+1)
        y2 = int(midy+ry+1)
    else:
        y1 = int(midy-ry)
        y2 = int(midy+ry+1)

    if crop_x % 2 == 0:
        x1 = int(midx-rx+1)
        x2 = int(midx+rx+1)
    else:
        x1 = int(midx-rx)
        x2 = int(midx+rx+1)
    img_crop = img[y1:y2, x1:x2]
    return img_crop


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


if __name__ == '__main__':
    a=1


