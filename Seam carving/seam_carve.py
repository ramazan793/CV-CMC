import numpy as np
from scipy.signal import correlate2d

def get_brightness(img):
    Y = np.average(img, weights = [0.299, 0.587, 0.114], axis = 2)
    return Y

def get_grad(img):
    Y = get_brightness(img)
    gradX_filter = np.array([
        [0,-1,0],
        [0,0,0],
        [0,1,0]
    ])
    gradY_filter = np.array([
        [0,0,0],
        [-1,0,1],
        [0,0,0]
    ])
    gradX = correlate2d(Y, gradX_filter, mode='same', boundary='symm')
    gradY = correlate2d(Y, gradY_filter, mode='same', boundary='symm')
    grad = np.linalg.norm((gradX, gradY), ord=2, axis=0)
    return grad

def get_vertical_energy(grad_img):
    res = np.copy(grad_img)
    for i in range(1, grad_img.shape[0]):
        upper_row = res[i - 1, :]
        for j in range(grad_img.shape[1]):
            adj = upper_row[max(0, j - 1):min(j + 1, grad_img.shape[1])+1]
            res[i,j] += np.min(adj)
    return res

def get_horizontal_energy(grad_img):
    return get_vertical_energy(grad_img.T).T


def get_vertical_seam(img_energy):
    j = np.argmin(img_energy[img_energy.shape[0]-1, :])
    seam = np.zeros_like(img_energy, dtype=np.float64)
    seam[img_energy.shape[0]-1, j] = 1
    for i in reversed(range(0, img_energy.shape[0]-1)):
        adj = img_energy[i, max(0, j - 1):min(j + 1, img_energy.shape[1])+1]
        if j == 0:
            j += np.argmin(adj)
        else:
            j += np.argmin(adj) - 1
        seam[i,j] = 1
    return seam.astype(bool)

def get_horizontal_seam(img_energy):
    return get_vertical_seam(img_energy.T).T

def crop_horizontal(img, seam, colored = True):
    if colored:
        return img[~seam, :].reshape(img.shape[0], -1, 3)
    return img[~seam].reshape(img.shape[0], -1)

def crop_vertical(img, seam, colored = True):
    if colored:
        return np.transpose(crop_horizontal(np.transpose(img, axes=(1,0,2)), seam.T), axes=(1,0,2))
    return crop_horizontal(img.T, seam.T, colored).T

def expand_horizontal(img, seam, colored = True):
    if colored:
        coords = np.where(seam)
        coords2 = (coords[0], np.clip(coords[1] + 1, 0, img.shape[1]-1))

        mean = np.mean((img[coords[0], coords[1], :], img[coords2[0], coords2[1], :]), dtype=np.uint32, axis = 0).astype(np.uint8)

        t = np.zeros_like(img)
        t[coords2[0], coords2[1],:] = [1,1,1]
        index = np.argwhere(t)

        ravel_ind = 3*(index[:,0]*img.shape[1]+index[:,1])

        new = np.insert(img, ravel_ind, mean.ravel())
        new = new.reshape((img.shape[0], img.shape[1]+1, 3))
    else:
        coords = np.where(seam)
        coords2 = (coords[0], coords[1] + 1)
        ravel_idx = coords2[0]*img.shape[1]+coords2[1]

        filler = np.zeros_like(img[coords[0], coords[1]]) + 1
        
        new = np.insert(img, ravel_idx, filler.ravel()).reshape(img.shape[0], img.shape[1]+1)
        new[coords[0], coords[1]] += 1
        
        new = np.round(new)
    return new

def expand_vertical(img, seam, colored = True):
    if colored:
        return np.transpose(expand_horizontal(np.transpose(img, axes=(1,0,2)), seam.T), axes=(1,0,2))
    return expand_horizontal(img.T, seam.T, colored).T

def apply_mask(img, mask):
    MAX_GRAD_VAL = img.shape[0]*img.shape[1] * 256
    
    img[mask >= 1] = img[mask >= 1] + MAX_GRAD_VAL
    img[mask == -1] = img[mask == -1] - MAX_GRAD_VAL
    return img

def seam_carve(img, mode, mask = None):
    
    grad_img = get_grad(img)
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=np.float64)
    else:
        mask = mask.astype(np.float64)
        
    grad_img = apply_mask(grad_img, mask)
#     cropped_mask = None
    if mode == 'horizontal shrink':
        vertical_energy = get_vertical_energy(grad_img)

        seam = get_vertical_seam(vertical_energy)
        cropped_img = crop_horizontal(img, seam)
        cropped_mask = crop_horizontal(mask, seam, False)
    elif mode == 'vertical shrink':
        horizontal_energy = get_horizontal_energy(grad_img)
        
        seam = get_horizontal_seam(horizontal_energy)
        cropped_img = crop_vertical(img, seam)
        cropped_mask = crop_vertical(mask, seam, False)
    elif mode == 'horizontal expand':
        vertical_energy = get_vertical_energy(grad_img)

        seam = get_vertical_seam(vertical_energy)
        cropped_img = expand_horizontal(img, seam)
        cropped_mask = expand_horizontal(mask, seam, False)
    else:
        horizontal_energy = get_horizontal_energy(grad_img)
        
        seam = get_horizontal_seam(horizontal_energy)
        cropped_img = expand_vertical(img, seam)
        cropped_mask = expand_vertical(mask, seam, False)
        
    return cropped_img, cropped_mask, seam