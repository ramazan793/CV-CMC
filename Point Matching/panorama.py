import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints = 500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    descriptor_extractor = ORB(n_keypoints = n_keypoints)

    descriptor_extractor.detect_and_extract(rgb2gray(img))
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    points = points.astype(np.float64)
    
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    
    C_x = np.mean(points[:, 0])
    C_y = np.mean(points[:, 1])
    
    N = np.mean(np.sqrt((points[:,0] - C_x)**2 +  (points[:,1] - C_y)**2))
    N = np.sqrt(2) / N
    
    M = np.array([
        [N, 0, -N*C_x],
        [0, N, -N*C_y],
        [0, 0, 1]
    ])
    
    transformed_points = np.dot(M, pointsh)
    return M, transformed_points


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    
    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)
    a_x = np.concatenate([-src.T, np.zeros_like(src.T), src.T * dest[0][:, None]], axis = 1)
    a_y = np.concatenate([np.zeros_like(src.T), -src.T, src.T * dest[1][:, None]], axis = 1)

    A = []
    for i in range(a_x.shape[0]):
        A.extend([a_x[i], a_y[i]])
    A = np.row_stack(A)
    
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    
    h = vh[-1, :].reshape((3,3))
    
    H = np.dot(np.dot(np.linalg.inv(dest_matrix), h), src_matrix)
    return H


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials = 500, residual_threshold = 1.9, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    matches = match_descriptors(src_descriptors, dest_descriptors)
    
    best_inliers = np.zeros((0,))
    
    for i in range(max_trials):
        choise_idx = np.arange(matches.shape[0])
        np.random.shuffle(choise_idx)
        choise_idx = choise_idx[:4]
        
        H = find_homography(src_keypoints[matches[choise_idx, 0]], dest_keypoints[matches[choise_idx, 1]])
        transform = ProjectiveTransform(H)
        
        inliers_idx = np.nonzero(np.linalg.norm(dest_keypoints[matches[:, 1]] - transform(src_keypoints[matches[:, 0]]), axis=1) < residual_threshold)[0]
        if inliers_idx.shape[0] > best_inliers.shape[0]:
            best_inliers = np.copy(inliers_idx)
    
    H = find_homography(src_keypoints[matches[best_inliers, 0]], dest_keypoints[matches[best_inliers, 1]])
    transform = ProjectiveTransform(H)
    
    if return_matches:
        return transform, matches[best_inliers]
    
#     print(best_inliers.shape[0])
#     print()
    return transform
    
def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    
    for i in reversed(range(0, center_index)):
        result[i] = forward_transforms[i] + result[i + 1]
    
    for i in range(center_index + 1, image_count):
        result[i] = ProjectiveTransform(np.linalg.inv(forward_transforms[i - 1].params)) + result[i - 1]
    
    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
            output_shape
        """
    
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    
    # shift to positive quarter
    final_warps = []
    for transform in simple_center_warps:
        final_warps.append(transform + AffineTransform(translation=-min_coords[::-1]))
    
    final_corners = tuple(get_corners(image_collection, final_warps))
    final_min, final_max = get_min_max_coords(final_corners)
    
    return tuple(final_warps), tuple((final_max - final_min)[::-1].astype(np.int32))

def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    # your code here
    mask = np.ones(image.shape[:2])
    rotated_transform_inv = rotate_transform_matrix(ProjectiveTransform(np.linalg.inv(transform.params)))
    
    warped_image = warp(image, rotated_transform_inv, mode='constant', output_shape = output_shape)
    warped_mask = warp(mask, rotated_transform_inv, mode='constant', output_shape = output_shape)

    return warped_image, warped_mask.astype(np.bool8)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    for i, img in enumerate(image_collection):
        warped_image, warped_mask = warp_image(img, final_center_warps[i], output_shape) 
        
        not_intersection_mask = (warped_mask ^ result_mask)
        result += warped_image*not_intersection_mask[:, :, None]
        
        result_mask += not_intersection_mask
    
    return result


def get_gaussian_pyramid(image, n_layers = 4, sigma = 1):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    pyramid = [image]
    for i in range(1, n_layers):
        img = gaussian(pyramid[-1], sigma)
        pyramid.append(gaussian(pyramid[i - 1], sigma))
    
    return pyramid

def get_laplacian_pyramid(image, n_layers = 4, sigma = 1):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(1, n_layers):
        laplacian_pyramid.append(gaussian_pyramid[n_layers - 1 - i] - gaussian_pyramid[n_layers - i])
        
    return laplacian_pyramid[::-1]


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers = 4, image_sigma = 1, merge_sigma = 2.5):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    
    
    for i, img in enumerate(image_collection):
        warped_image, warped_mask = warp_image(img, final_center_warps[i], output_shape) 
        
        if i != 0:
            intersection = (result_mask & warped_mask)
            center_row = np.nonzero(intersection[intersection.shape[0] // 2, :])[0]
            center_col = int(np.mean([center_row[0], center_row[-1]]))
    
            right_half = np.copy(intersection)
            right_half[:, :center_col] = 0
            
            mask = result_mask * (~right_half)
            
            mask_gauss = get_gaussian_pyramid(mask.astype(np.float64), sigma = merge_sigma)
            laplacian1 = get_laplacian_pyramid(result, sigma = image_sigma)
            laplacian2 = get_laplacian_pyramid(warped_image, sigma = image_sigma)
            
            blend_pyramid = []
            for i in range(len(mask_gauss)):
                blend_pyramid.append(laplacian1[i]*mask_gauss[i][:,:,None] + (1 - mask_gauss[i])[:,:,None]*laplacian2[i])
            
            result = sum(blend_pyramid)
            result_mask += (result_mask ^ warped_mask)
        else:
            result += warped_image
            result_mask += warped_mask
    
    
    return np.clip(result, 0, 1)


def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    if scale is None:
        scale = w / np.pi * 1.3
    
    C = np.row_stack([coords[:,0], coords[:, 1], np.ones(coords[:,0].shape)])
    
    K = np.array([
        [scale, 0 , w/2],
        [0, scale, h/2],
        [0, 0, 1]
    ])
    C = np.dot(np.linalg.inv(K), C)
    
    B = np.row_stack([np.tan(C[0]), C[1]/np.cos(C[0]), np.ones(C[0].shape)])
    B = np.dot(K, B)
    
    src_coords = np.zeros_like(coords)
    src_coords[:, 0] = B[0]
    src_coords[:, 1] = B[1]
    
    return src_coords

def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    def inverse_map(coords):
        return cylindrical_inverse_map(coords, h = img.shape[0], w = img.shape[1], scale = scale)
    
    cylindrical = warp(img, inverse_map)
    
    if crop:
        min_w, max_w = np.nonzero(np.max(rgb2gray(cylindrical), axis = 0))[0][[0, -1]]
        min_h, max_h = np.nonzero(np.max(rgb2gray(cylindrical), axis = 1))[0][[0, -1]]
        cylindrical =  cylindrical[min_h:max_h, min_w:max_w]

    return cylindrical