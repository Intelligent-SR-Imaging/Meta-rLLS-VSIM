"""
Functions in this file are used to get sections in volume quickly
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""

import random
import numpy as np
from math import cos,sin,pi,floor,tan


def get_batch_cor(d_cor_ori,off,direction):
    """get the plane's x,y,z cor in volume"""
    batch_d_cor = []
    for offset in range(off):
        if direction == "X":
            d_cor =d_cor_ori+ np.array([offset, 0, 0]).reshape(1,1, 1, 1, 1, 3).astype(int)
        elif direction == "Y":
            d_cor =d_cor_ori+ np.array([0, offset, 0]).reshape(1,1, 1, 1, 1, 3).astype(int)
        elif direction == "Z":
            d_cor =d_cor_ori+ np.array([0, 0, offset]).reshape(1,1, 1, 1, 1, 3).astype(int)

        batch_d_cor.append(d_cor)
    batch_d_cor = np.concatenate(batch_d_cor,axis=0)
    return batch_d_cor

def section_position(H,W,theta,axis):
    """get the section's x,y,z cor in volume, which rotated theta around axis based on plane"""
    x = []
    y = []
    z = []
    for i in range(H):
        y+=[j for j in range(W)]

    for i in range(H):
        x+=[i for _ in range(W)]

    for i in range(H):
        z+=[0 for _ in range(W)]

    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    z = np.array(z).reshape(-1,1)

    cor = np.concatenate((x,y,z),axis = 1).transpose(1,0)

    if axis=="X":
        map = \
            [[1,0,0],
             [0,cos(theta),-sin(theta)],
             [0,sin(theta),cos(theta)]
        ]
        map = np.array(map)

        rot_cor = np.matmul(map, cor)

        if theta > pi / 2:
            max = -min(rot_cor[1, :])
            off = np.array([0, max, 0]).reshape(-1, 1)
            rot_cor += off

    elif axis == "Y":
        map = \
            [
             [cos(theta), 0, -sin(theta)],
             [0, 1, 0],
             [sin(theta), 0, cos(theta)]
             ]
        map = np.array(map)

        rot_cor = np.matmul(map, cor)
        if theta > pi / 2:
            max = -min(rot_cor[0, :])
            off = np.array([max, 0, 0]).reshape(-1, 1)
            rot_cor += off

    else:
        raise "Axis Not Found!"

    rot_cor=  rot_cor.transpose(1,0) + np.array([1, 1, 1]).reshape(1, -1)

    return rot_cor

def discret_cor(rot_cor):
    """get the section's neighbors x,y,z cor in volume"""
    rot_cor = rot_cor.reshape(-1,1,3)
    rot_cor_d = np.floor(rot_cor)
    d_cor = []
    for x_off in range(-1,3):
        for y_off in range(-1, 3):
            for z_off in range(-1, 3):
                off = np.array([x_off,y_off,z_off]).reshape(-1,1,3)
                d_cor.append(rot_cor_d + off)

    discret_cor = np.concatenate(d_cor,axis=1)
    discret_cor = discret_cor.reshape(1,-1,4,4,4,3)
    return discret_cor

def Bicubic_function(cor,method = "conv3"):
    """calculate the neighbors weight based on diffearent interpolation"""
    if method == "conv3":
        cor_floor = np.floor(cor)
        x_off = cor[:, 0] - cor_floor[:, 0]
        x_a = x_off + 1
        x_b = x_off
        x_c = x_off - 1
        x_d = x_off + 2

        y_off = cor[:, 1] - cor_floor[:, 1]
        y_a = y_off + 1
        y_b = y_off
        y_c = y_off - 1
        y_d = y_off + 2

        z_off = cor[:, 2] - cor_floor[:, 2]
        z_a = z_off + 1
        z_b = z_off
        z_c = z_off - 1
        z_d = z_off + 2

        x_a_weight = (-(x_a ** 3) + 2 * (x_a ** 2) - x_a).reshape(-1, 1)
        x_b_weight = (3 * (x_b ** 3) - 5 * (x_b ** 2) + 2).reshape(-1, 1)
        x_c_weight = (-3 * (x_c ** 3) + 4 * (x_c ** 2) + x_c).reshape(-1, 1)
        x_d_weight = (x_d ** 3 - x_d).reshape(-1, 1)

        x_weight = np.concatenate((x_a_weight, x_b_weight, x_c_weight, x_d_weight), axis=1)
        x_weight = x_weight.reshape(-1, 4, 1, 1).repeat(4, axis=2).repeat(4, axis=3)

        y_a_weight = (-(y_a ** 3) + 2 * (y_a ** 2) - y_a).reshape(-1, 1)
        y_b_weight = (3 * (y_b ** 3) - 5 * (y_b ** 2) + 2).reshape(-1, 1)
        y_c_weight = (-3 * (y_c ** 3) + 4 * (y_c ** 2) + y_c).reshape(-1, 1)
        y_d_weight = (y_d ** 3 - y_d).reshape(-1, 1)

        y_weight = np.concatenate((y_a_weight, y_b_weight, y_c_weight, y_d_weight), axis=1)
        y_weight = y_weight.reshape(-1, 1, 4, 1).repeat(4, axis=1).repeat(4, axis=3)

        z_a_weight = (-(z_a ** 3) + 2 * (z_a ** 2) - z_a).reshape(-1, 1)
        z_b_weight = (3 * (z_b ** 3) - 5 * (z_b ** 2) + 2).reshape(-1, 1)
        z_c_weight = (-3 * (z_c ** 3) + 4 * (z_c ** 2) + z_c).reshape(-1, 1)
        z_d_weight = (z_d ** 3 - z_d).reshape(-1, 1)

        z_weight = np.concatenate((z_a_weight, z_b_weight, z_c_weight, z_d_weight), axis=1)
        z_weight = z_weight.reshape(-1, 1, 1, 4).repeat(4, axis=1).repeat(4, axis=2)

        bubic_weight = x_weight * y_weight * z_weight
        bubic_weight = bubic_weight.reshape(1,-1,4,4,4)
        return bubic_weight
    elif method == "linear":
        cor_floor = np.floor(cor)
        x_off = cor[:, 0] - cor_floor[:, 0]
        x_a = np.zeros_like(x_off)
        x_b = 1-x_off
        x_c = x_off
        x_d = np.zeros_like(x_off)

        y_off = cor[:, 1] - cor_floor[:, 1]
        y_a = np.zeros_like(x_off)
        y_b = 1 - y_off
        y_c = y_off
        y_d = np.zeros_like(x_off)

        z_off = cor[:, 2] - cor_floor[:, 2]
        z_a = np.zeros_like(x_off)
        z_b = 1 - z_off
        z_c = z_off
        z_d = np.zeros_like(x_off)

        x_a_weight = (x_a).reshape(-1, 1)
        x_b_weight = (x_b).reshape(-1, 1)
        x_c_weight = (x_c).reshape(-1, 1)
        x_d_weight = (x_d).reshape(-1, 1)

        x_weight = np.concatenate((x_a_weight, x_b_weight, x_c_weight, x_d_weight), axis=1)
        x_weight = x_weight.reshape(-1, 4, 1, 1).repeat(4, axis=2).repeat(4, axis=3)

        y_a_weight = (y_a).reshape(-1, 1)
        y_b_weight = (y_b).reshape(-1, 1)
        y_c_weight = (y_c).reshape(-1, 1)
        y_d_weight = (y_d).reshape(-1, 1)

        y_weight = np.concatenate((y_a_weight, y_b_weight, y_c_weight, y_d_weight), axis=1)
        y_weight = y_weight.reshape(-1, 1, 4, 1).repeat(4, axis=1).repeat(4, axis=3)

        z_a_weight = (z_a).reshape(-1, 1)
        z_b_weight = (z_b).reshape(-1, 1)
        z_c_weight = (z_c).reshape(-1, 1)
        z_d_weight = (z_d).reshape(-1, 1)

        z_weight = np.concatenate((z_a_weight, z_b_weight, z_c_weight, z_d_weight), axis=1)
        z_weight = z_weight.reshape(-1, 1, 1, 4).repeat(4, axis=1).repeat(4, axis=2)

        bubic_weight = x_weight * y_weight * z_weight
        return bubic_weight


def get_adptive_size(X_dim,Y_dim,Z_dim,theta,axis):
    if theta > pi/2:
        theta = pi - theta
    if axis == "X":

        H = X_dim-4
        WY_max = floor((Y_dim-4)/cos(theta))
        WZ_max = floor((Z_dim-4)/sin(theta))
        W = min(WY_max,WZ_max)-4
        if WY_max>WZ_max:
            direction = "Y"
            off = floor(Y_dim-Z_dim/tan(theta))-2
        else:
            direction = "Z"
            off = floor(Z_dim - tan(theta) * Y_dim)-2

    elif axis == "Y":
        W = Y_dim-4
        HX_max = floor((X_dim-4)/cos(theta))
        HZ_max = floor((Z_dim-4)/sin(theta))

        H = min(HX_max,HZ_max)
        if HX_max>HZ_max:
            direction = "X"
            off = floor(X_dim - Z_dim/tan(theta))-2
        else:
            direction = "Z"
            off = floor(Z_dim - tan(theta) * X_dim)-2

    return H,W,direction,off

def get_section(vol,d_cor,H,W,bubic_weight,batch):
    """Get section in a volume"""
    section_near = vol[d_cor[:,:,:,:,:,0],d_cor[:,:,:,:,:,1],d_cor[:,:,:,:,:,2]]
    w_section_near = section_near*bubic_weight
    w_section_near = w_section_near.reshape(batch,-1,4*4*4)
    w_section_near = w_section_near.sum(axis = 2)
    section = w_section_near.reshape(batch,H,W)

    return section

def move_section(d_cor,direction,off):
    """quick move the position to get section"""
    offset = random.randint((0,off))
    if direction == "X":
        d_cor += np.array([offset,0,0]).reshape(1,1,1,1,3).astype(int)
    elif direction == "Y":
        d_cor += np.array([0,offset,0]).reshape(1,1,1,1,3).astype(int)
    elif direction == "Z":
        d_cor += np.array([0,0,offset]).reshape(1,1,1,1,3).astype(int)
    return d_cor


