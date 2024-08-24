import math
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import tifffile as tiff
import cv2
from typing import Iterable, Optional
import numpy as np
import torch
from torch import nn
import torchvision as tv
from scipy.interpolate import interp2d
import os


rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i4"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]



# img torch tensor [bs, ch, y, x]
def ssim(image1, image2, cuda, K=(0.01, 0.03), window_size=11, L=1):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = window.to(cuda)

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2
    C2 = K[1] ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = np.max(img1)
    # PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_mask(img1, img2, thresh=0.01):
    mask = XxCalMask(XxPrctileNorm(img2), thresh=thresh)
    img1 = img1*mask
    img2 = img2*mask
    # mse = np.mean((img1 - img2) ** 2)
    mse = np.sum((img1 - img2) ** 2) / np.sum(mask)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = np.max(img1)
    # PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def linear_tf(img, gt):
    x = np.reshape(img.transpose(), (img.size, 1))
    img_s = img.transpose().shape
    y = np.reshape(gt.transpose(), (gt.size, 1))
    X = np.concatenate([x, np.ones((img.size, 1))], axis=1)
    c = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    img_n = np.reshape(c[0] * x + c[1], img_s).transpose()
    return img_n


def fft2(inputs, device):
    inputs = apodize2d(inputs, device, napodize=10)
    fft = torch.fft.fft2(torch.complex(inputs, torch.zeros_like(inputs)))
    absfft = torch.pow(torch.abs(fft) + 1e-8, 0.1)
    output = torch.fft.fftshift(absfft)
    return output


def fft3(inputs, device):
    # inputs = apodize2d(inputs, device, napodize=10)
    fft = torch.fft.fftn(torch.complex(inputs, torch.zeros_like(inputs)))
    absfft = torch.pow(torch.abs(fft) + 1e-8, 0.1)
    output = torch.fft.fftshift(absfft)
    return output


def apodize2d(img, device, napodize=10):
    bs, ch, ny, nx = img.shape
    img_apo = img[:, :, napodize:ny-napodize, :]

    imageUp = img[:, :, 0:napodize, :]
    imageDown = img[:, :, ny-napodize:, :]
    diff = (torch.flip(imageDown, dims=[2]) - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis]
    fact = torch.as_tensor(fact, dtype=torch.float32)
    fact = torch.tile(fact, [bs, ch, 1, nx])
    fact = fact.to(device)
    factor = diff * fact
    imageUp = torch.add(imageUp, factor)
    imageDown = torch.subtract(imageDown, factor.flip(dims=[2]))
    img_apo = torch.concat([imageUp, img_apo, imageDown], dim=2)

    imageLeft = img_apo[:, :, :, 0:napodize]
    imageRight = img_apo[:, :, :, nx-napodize:]
    img_apo = img_apo[:, :, :, napodize:nx-napodize]
    diff = (imageRight.flip(dims=[3]) - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, np.newaxis, :]
    fact = torch.as_tensor(fact, dtype=torch.float32)
    fact = torch.tile(fact, [bs, ch, ny, 1])
    fact = fact.to(device)
    factor = diff * fact
    imageLeft = torch.add(imageLeft, factor)
    imageRight = torch.subtract(imageRight, factor.flip(dims=[3]))
    img_apo = torch.concat([imageLeft, img_apo, imageRight], dim=3)

    return img_apo


def gram_matrix(x: torch.tensor) -> torch.tensor:
    # from "Neural Style Transfer Using PyTorch" tutorial
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def XxPrctileNorm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y < 0] = 0.
    y[y > 1] = 1.
    return y


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def XxCalMask(img, ksize=11, thresh=5e-2):
    # img       -- 2D img to calculate foreground
    # ksize     -- the size of first gaussian kernel, typically set as 5~10
    # thresh    -- lower thresh leads to larger mask, typically set as
    #              [1e-3, 5e-2]
    fd = cv2.GaussianBlur(img, (ksize, ksize), ksize, sigmaY=ksize)
    bg = cv2.GaussianBlur(img, (101, 101), 50, sigmaY=50)
    mask = fd - bg
    mask[mask >= thresh] = 1
    mask[mask != 1] = 0
    mask = np.array(mask).astype('bool')
    return mask


def read_mrc(filename, filetype='image'):
    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        data = np.ndarray(shape=(nx, ny, nz), dtype=data_type)
        for iz in range(nz):
            data_2d = imgrawdata[nx * ny * iz:nx * ny * (iz + 1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return header, data


def write_mrc(filename, img_data, header):
    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6
    # header[0][3] = 6

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    img_data_n = img_data.reshape(nx * ny * nz, order='F')
    img_data_n.tofile(fd)
    fd.close()
    return


def preprocess_LLSM(data_path, save_path, angle=30.8, bg=100):
    files = np.sort(os.listdir(data_path))
    for i in range(len(files)):
        print('Preprocessing file:', files[i])
        header, img1 = read_mrc(data_path + '/' + files[i])
        img1 = np.float32(img1)
        img = np.zeros((img1.shape[0], img1.shape[1], img1.shape[2]//3))
        for j in range(img1.shape[2]//3):
            img[..., j] = np.mean(img1[..., j*3:(j+1)*3], axis=2)
        img = func_deskew(header, img, angle, 0)
        img = img - bg
        img[img < 0] = 0
        img = XxPrctileNorm(img)
        img = np.swapaxes(img, 0, -1)
        img = np.swapaxes(img, 1, 2)
        tiff.imwrite(save_path + '/' + '%.3d'%i + '.tif', np.uint16(65535*img))


def func_deskew(header, img, angle, bg):
    img = np.float32(img)
    dy = header[0][10]
    dz = header[0][12]
    sz = np.ceil(dz*img.shape[2]*np.cos(angle*np.pi/180)/dy+img.shape[0])
    res2 = np.zeros(shape=(int(sz), img.shape[1], img.shape[2]), dtype=np.float32)*bg
    shift = (np.arange(1, img.shape[2]+1).transpose()-(img.shape[2]+1)/2)*dz/dy*np.cos(angle*np.pi/180)-(res2.shape[0]-img.shape[0])/2
    x1, y1 = np.arange(1, img.shape[1]+1), np.arange(1, img.shape[0]+1)
    x2, y2 = np.arange(1, res2.shape[1]+1), np.arange(1, res2.shape[0]+1)
    for i in range(img.shape[2]):
        f = interp2d(x1, y1, img[..., i], 'cubic', fill_value=0)
        res2[..., i] = f(x2, y2+shift[i])
    return res2


class loss_mse_ssim_fft(nn.Module):
    def __init__(self, device, ssim_param=1e-1, fft_param=1):
        super(loss_mse_ssim_fft, self).__init__()
        self.device = device
        self.fft_param = fft_param
        self.ssim_param = ssim_param
        self.mse_param = 1

    def forward(self, y_true, y_pred):
        loss = 0
        for i in range(y_pred.shape[1]):
            x = y_true[:, i, :, :]
            y = y_pred[:, i, :, :]
            x = torch.unsqueeze(x, 1)
            y = torch.unsqueeze(y, 1)

            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
            y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

            if self.fft_param > 0:
                x_fft = fft2(x, self.device)
                y_fft = fft2(y, self.device)
                fft_loss = self.fft_param * torch.mean(torch.square((y_fft - x_fft)))
            else:
                fft_loss = 0

            if self.ssim_param > 0:
                ssim_loss = self.ssim_param * (1 - ssim(x, y, self.device))
            else:
                ssim_loss = 0

            mse_loss = self.mse_param * torch.mean(torch.square((y-x)))
            loss += (mse_loss + ssim_loss + fft_loss)
        # print('mse:', mse_loss.item(), 'ssim:', ssim_loss.item(), 'fft:', fft_loss.item())
        return loss/y_pred.shape[1]


def psnr_mask_v2(img1, img2, mask):
    img1 = img1*mask
    img2 = img2*mask
    # mse = np.mean((img1 - img2) ** 2)
    mse = np.sum((img1 - img2) ** 2) / np.sum(mask)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = np.max(img1)
    # PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def print_time():
    print('Time: ', strftime('%b-%d %H:%M:%S', localtime()))


def save_model(model, model_path, step, flag=0):
    if flag == 0:
        model_name = 'gmodel_{}.pth'.format(str(step))
    else:
        model_name = 'dmodel_{}.pth'.format(str(step))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))


def save_checkpoint(model, optim, checkpoint_path, step, flag=0):
    if flag == 0:
        checkpoint_name = 'g_latest.pth'
    else:
        checkpoint_name = 'd_latest.pth'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = {'model': model.state_dict(),
                  'optim': optim.state_dict(),
                  'step': step}
    torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_name))
