"""
utils for training
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import numpy as np


def Align(img, x1, y1, z1, padValue=0):
    """This function is used to a lign a 3D image to a specific size
           Args:
                img (numpy.array): a 3D image stack
                x1, y1, z1, (int): Size to align
                padValue(value) : the value for padding
    """
    x2, y2, z2 = img.shape

    x_m = max(x1, x2)
    y_m = max(y1, y2)
    z_m = max(z1, z2)

    img_tmp = np.ones((x_m, y_m, z_m)) * padValue

    x_o = round((x_m - x2) / 2)
    y_o = round((y_m - y2) / 2)
    z_o = round((z_m - z2) / 2)

    img_tmp[x_o:x2 + x_o, y_o:y2 + y_o, z_o:z2 + z_o] = img

    x_o = round((x_m - x1) / 2)
    y_o = round((y_m - y1) / 2)
    z_o = round((z_m - z1) / 2)

    img = img_tmp[x_o:x1 + x_o, y_o:y1 + y_o, z_o:z1 + z_o]
    return img


def padding(view):
    """This function is used to pad a whole cell image when reconstruction crop by crop
               Args:
                    view (numpy.array): a 3D image stack
    """
    H = view.shape[0]
    W = view.shape[1]
    Z = view.shape[2]
    pad_1 = np.zeros((H, 8, Z))
    pad_2 = np.zeros((H, 8, Z))
    pad_3 = np.zeros((8, W + 16, Z))
    pad_4 = np.zeros((8, W + 16, Z))
    view = np.concatenate((pad_1, view), axis=1)
    view = np.concatenate((view, pad_2), axis=1)
    view = np.concatenate((pad_3, view), axis=0)
    view = np.concatenate((view, pad_4), axis=0)
    return view
