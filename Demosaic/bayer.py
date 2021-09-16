import numpy as np
from scipy.ndimage import convolve

def get_pattern_tile(pattern, shape):
    tile = np.tile(pattern, np.array(shape) // 2 + np.array(shape) % 2)
    if shape[0] % 2:
        tile = tile[:-1, :]
    if shape[1] % 2:
        tile = tile[:, :-1]
    return tile.astype(bool)

def get_bayer_masks(n_rows, n_cols):
    red_tile = np.array([[0, 1], [0, 0]], dtype=bool)
    green_tile = np.eye(2, dtype=bool)
    blue_tile = np.array([[0, 0], [1, 0]], dtype=bool)
    
    shape = (n_rows, n_cols)
    red = get_pattern_tile(red_tile, shape)
    green = get_pattern_tile(green_tile, shape)
    blue = get_pattern_tile(blue_tile, shape)
    
    mask = np.dstack((red, green, blue))
    return mask

def get_colored_img(raw_img):
    bayer_masks = get_bayer_masks(*raw_img.shape)
    ext_img = np.repeat(raw_img[:,:,None], 3, axis = 2)
    return ext_img * bayer_masks

# 7 minutes solution :(
# def bilinear_interpolation(img):
#     masks = get_bayer_masks(*img.shape[:-1]) 
#     for i in range(1, img.shape[0] - 1):
#         for j in range(1, img.shape[1] - 1):
#             for channel in range(3):
#                 if masks[i, j, channel]:
#                     continue
#                 area = np.nonzero(masks[i-1:i+2, j-1:j+2, channel])
#                 img[i, j, channel] = np.mean(img[i - 1 + area[0], j - 1 + area[1], channel])    
#     return img

# 5 seconds solution :)
def bilinear_interpolation(img):
    blue_2 = get_pattern_tile([[1,0],[0,1]], img.shape[:-1])
    blue_4 = get_pattern_tile([[0,0],[1,0]], img.shape[:-1])
    green = get_pattern_tile([[0,1],[1,0]], img.shape[:-1])
    red_2 = blue_2
    red_4 = get_pattern_tile([[0,1],[0,0]], img.shape[:-1])

    filter_2 = np.ones((3,3)) / 2
    filter_4 = np.ones((3,3)) / 4
    convolved_2_blue = convolve(img[:, :, 0], filter_2)
    convolved_2_red = convolve(img[:, :, 2], filter_2)
    convolved_4_blue = convolve(img[:, :, 0], filter_4)
    convolved_4_red = convolve(img[:, :, 2], filter_4)
    convolved_green = convolve(img[:, :, 1], filter_4)

    convolved_2_blue[~blue_2] = 0
    convolved_2_red[~red_2] = 0
    convolved_4_blue[~blue_4] = 0
    convolved_4_red[~red_4] = 0
    convolved_green[~green] = 0

    img[:, :, 0] += convolved_2_blue + convolved_4_blue
    img[:, :, 1] += convolved_green
    img[:, :, 2] += convolved_2_red + convolved_4_red
    
    return img

def improved_interpolation(raw_img):
    G_R = np.array([
        [0,0,-1,0,0],
        [0,0,2,0,0],
        [-1,2,4,2,-1],
        [0,0,2,0,0],
        [0,0,-1,0,0]
    ]) / 8
    G_B = G_R
    
    R_G_1 = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]
    ]) / 8
    R_G_2 = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0.5, 0, 5, 0, 0.5],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0]
    ]) / 8
    R_B = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]
    ]) / 8
    
    B_G_1 = R_G_1
    B_G_2 = R_G_2
    B_R = R_B
    
    G_R_mask = get_pattern_tile([[0, 1], [0, 0]], raw_img.shape[:2])
    G_B_mask = get_pattern_tile([[0, 0], [1, 0]], raw_img.shape[:2])
    G_mask = get_pattern_tile([[1, 0], [0, 1]], raw_img.shape[:2])
    R_G_1_mask = get_pattern_tile([[1,0],[0,0]], raw_img.shape[:2])
    R_G_2_mask = get_pattern_tile([[0,0],[0,1]], raw_img.shape[:2])
    R_B_mask = G_B_mask
    B_G_1_mask = R_G_2_mask
    B_G_2_mask = R_G_1_mask
    B_R_mask = G_R_mask
    
    raw_img = raw_img.astype(np.int32)
    G = convolve(raw_img, G_R)*G_R_mask + convolve(raw_img, G_B)*G_B_mask + raw_img*G_mask
    R = convolve(raw_img, R_G_1)*R_G_1_mask + convolve(raw_img, R_G_2)*R_G_2_mask + convolve(raw_img, R_B)*R_B_mask + raw_img*G_R_mask
    B = convolve(raw_img, B_G_1)*B_G_1_mask + convolve(raw_img, B_G_2)*B_G_2_mask + convolve(raw_img, B_R)*B_R_mask + raw_img*G_B_mask
    
    return np.dstack((R, G, B)).clip(0, 255).astype(np.uint8)
    
def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    
    C = 3
    H, W = img_pred.shape[:2]
    mse = np.sum(((img_pred-img_gt)**2)) / (C * H * W)
    if mse == 0:
        raise(ValueError)
    psnr = 10 * np.log10(np.max(img_gt)**2/mse)
    
    return psnr