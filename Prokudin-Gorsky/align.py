import numpy as np
from scipy.fft import fft2, ifft2

def unstack(img, height):
    return img[:height, :], img[height:2*height, :], img[2*height:3*height, :]

def crop(frame, v_crop, h_crop):
    return frame[v_crop:frame.shape[0] - v_crop, h_crop:frame.shape[1] - h_crop]

def find_shift(A, B):
    corr = ifft2(fft2(A)*np.conj(fft2(B)))
    return np.unravel_index(np.argmax(corr), corr.shape)

def raw_to_rel(raw_shift, coord, shape):
    return -((coord + raw_shift) % shape - coord)

def roll2d(arr, u, v):
    arr = np.copy(arr)
    arr = np.roll(arr, u, axis = 0)
    arr = np.roll(arr, v, axis = 1)
    return arr

def align(img, g_abs):
    img = (img*255).astype(np.uint8)
    
    frame_height = img.shape[0] // 3
    B, G, R = unstack(img, frame_height)
    
    v_crop, h_crop = (np.array(B.shape) * 0.06).astype(np.int32)
    inner_shape = np.array([B.shape[0] - 2*v_crop, B.shape[1] - 2*h_crop])
    B, G, R = map(lambda x: crop(x, v_crop, h_crop), (B, G, R))
        
    G_B_raw, G_R_raw = find_shift(G, B), find_shift(G, R)
    g_rel = np.array([g_abs[0] - frame_height - v_crop, g_abs[1] - h_crop])
    G_B, G_R = map(lambda x: raw_to_rel(x, g_rel, inner_shape), (G_B_raw, G_R_raw))
    
    b_abs = g_abs[0] - frame_height + G_B[0], g_abs[1] + G_B[1]
    r_abs = g_abs[0] + frame_height + G_R[0], g_abs[1] + G_R[1]

    B_new = roll2d(B, *-G_B)
    R_new = roll2d(R, *-G_R)
    
    return np.dstack((R_new, G, B_new)), b_abs, r_abs
