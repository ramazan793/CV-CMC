#!/usr/bin/env python3

import panorama
import plots

from skimage import io
import numpy as np

# ------------------------------------------------------------------------------------------
# Part 0
# ------------------------------------------------------------------------------------------

code = '01'

pano_image_collection = io.ImageCollection(f"imgs/{code}/*.jpg",
                                           load_func=lambda f: io.imread(f).astype(np.float64) / 255)
# plots.plot_collage(pano_image_collection, title=f"Image collection size: {len(pano_image_collection)}")


# ------------------------------------------------------------------------------------------
# Part 1
# ------------------------------------------------------------------------------------------

img = pano_image_collection[0]
keypoints, descriptors = panorama.find_orb(img)

# plots.plot_keypoints(img, keypoints)


# ------------------------------------------------------------------------------------------
# Part 2 and 3
# ------------------------------------------------------------------------------------------

src, dest = pano_image_collection[0], pano_image_collection[1]
src_keypoints, src_descriptors = panorama.find_orb(src)
dest_keypoints, dest_descriptors = panorama.find_orb(dest)

robust_transform, matches = panorama.ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, return_matches=True)

# plots.plot_inliers(src, dest, src_keypoints, dest_keypoints, matches)


# ------------------------------------------------------------------------------------------
# Part 4
# ------------------------------------------------------------------------------------------

keypoints, descriptors = zip(*(panorama.find_orb(img) for img in pano_image_collection))
forward_transforms = tuple(panorama.ransac_transform(src_kp, src_desc, dest_kp, dest_desc)
                           for src_kp, src_desc, dest_kp, dest_desc
                           in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))


simple_center_warps = panorama.find_simple_center_warps(forward_transforms)
corners = tuple(panorama.get_corners(pano_image_collection, simple_center_warps))
min_coords, max_coords = panorama.get_min_max_coords(corners)
center_img = pano_image_collection[(len(pano_image_collection) - 1) // 2]

# plots.plot_warps(corners, min_coords=min_coords, max_coords=max_coords, img=center_img)


final_center_warps, output_shape = panorama.get_final_center_warps(pano_image_collection, simple_center_warps)
corners = tuple(panorama.get_corners(pano_image_collection, final_center_warps))

# plots.plot_warps(corners, output_shape=output_shape)

# ------------------------------------------------------------------------------------------
# Part 5
# ------------------------------------------------------------------------------------------

result = panorama.merge_pano(pano_image_collection, final_center_warps, output_shape)

# plots.plot_result(result)
io.imsave(f"./results/{code}_base_pano.jpeg", result)

# ------------------------------------------------------------------------------------------
# Part 6
# ------------------------------------------------------------------------------------------

img = pano_image_collection[0]

laplacian_pyramid = panorama.get_laplacian_pyramid(img)
merged_img = panorama.merge_laplacian_pyramid(laplacian_pyramid)

# plots.plot_gauss(img, merged_img)
# plots.plot_collage(panorama.increase_contrast(laplacian_pyramid), columns=2, rows=2)

result = panorama.gaussian_merge_pano(pano_image_collection, final_center_warps, output_shape)

# plots.plot_result(result)
io.imsave(f"./results/{code}_improved_pano.jpeg", result)

# ------------------------------------------------------------------------------------------
# Part 7
# ------------------------------------------------------------------------------------------

cylindrical = panorama.warp_cylindrical(result)

# plots.plot_result(cylindrical)
io.imsave(f"./results/{code}_cylindrical_pano.jpeg", cylindrical)
