# Project:
#   VQA
# Description:
#   Image processing tools
# Author: 
#   Sergio Tascon-Morales

from PIL import Image
import numpy as np
import cv2
from skimage.measure import regionprops, label
from sklearn.metrics import pairwise_distances
from scipy import ndimage as ndi


STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.bool)

def normalize_0_255(data):
    x_min = np.min(data)
    x_max = np.max(data)
    data_norm = (data - x_min)/(x_max-x_min)*255
    data_norm = np.round(data_norm).astype(int)
    return data_norm

def normalize_rgb(img):
    if len(img.shape) < 3: # if gray scale image
        img = np.repeat(img[:,:, np.newaxis], 3, axis=2)
    _, _, c = img.shape
    img_norm = np.zeros_like(img)
    for i in range(c): # normalize every channel
        ch = img[:,:,i]
        ch_norm = normalize_0_255(ch)
        img_norm[:,:,i] = ch_norm
    return img_norm

def normalize_and_save(path_in, path_out, resize=True, size = 512, normalize=True, is_mask = False):
    """Normalization function
    """
    im = Image.open(path_in)
    imarray = np.array(im)
    if normalize:
        im_norm = normalize_rgb(imarray)
    else:
        if is_mask:
            im_norm = 255*imarray
        else:
            im_norm = imarray

    im_norm = Image.fromarray(im_norm)

    if resize:
        if is_mask:
            im_resized = im_norm.resize((size, size), Image.NEAREST)
        else:
            im_resized = im_norm.resize((size, size), Image.ANTIALIAS)
    else:
        im_resized = im_norm

    im_resized.save(path_out)


def find_mask_edges(mask):
    # this function returns the edges of a binary mask image.
    # The returning image edge_mask is also a binary image highlighting the edges
    [cols, rows] = np.shape(mask)
    edge_mask = np.zeros(np.shape(mask))
    for ind, value in np.ndenumerate(mask):
        if value and (mask[ind[0] - 1, ind[1]] == 0 or
                        mask[ind[0] + 1, ind[1]] == 0 or
                        mask[ind[0], ind[1] - 1] == 0 or
                        mask[ind[0], ind[1] + 1] == 0):
            edge_mask[ind] = 1

    for y in range(rows):
        for x in range(cols):
            if mask[x, y]:
                if (mask[x - 1, y] == 0 or mask[x + 1, y] == 0 or mask[x, y - 1] == 0 or mask[x, y + 1] == 0):
                    edge_mask[x, y] = 1
    return edge_mask.astype(np.uint8)


def is_object_in_region(gt, mask):
    # consider that several objects can be present in a reduced space
    prod = gt*mask 
    contained = False
    contours, _ = cv2.findContours(prod, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    num_objects_in_region = len(contours)

    if num_objects_in_region == 0:
        pass
    elif num_objects_in_region == 1: # if there is a single object of GT class in the region
        if np.count_nonzero(prod) > 0: # neccesary?
            if np.count_nonzero(gt*find_mask_edges(mask)) == 0: # if the object is not touching the borders
                contained = True
    else: # if there are several objects, check if at least one is contained in the region
        cnt = 0
        for c in contours: # process every contour
            if c.shape[0] > 5: # prevent dot objects
                # create image with this contour only
                temp = np.zeros_like(gt, dtype= np.uint8)
                temp = cv2.fillPoly(temp, pts = [c], color=(255,255,255))
                if np.count_nonzero(temp*find_mask_edges(mask)) == 0: # if the object is not touching the borders
                    cnt += 1
        if cnt > 0:
            contained = True

    return contained


def get_border_image(region):
    convex_hull_mask = region.convex_image
    eroded_image = ndi.binary_erosion(convex_hull_mask, STREL_4, border_value=0)
    border_image = np.logical_xor(convex_hull_mask, eroded_image)
    return border_image


def get_region_diameter(img):

    assert img.dtype == np.bool and len(img.shape) == 2

    label_img = label(img, connectivity=img.ndim)

    region = regionprops(label_img)[0]
    border_image = get_border_image(region)
    perimeter_coordinates = np.transpose(np.nonzero(border_image))
    pairwise_distances_matrix = pairwise_distances(perimeter_coordinates)
    i, j = np.unravel_index(np.argmax(pairwise_distances_matrix), pairwise_distances_matrix.shape)
    ptA, ptB = perimeter_coordinates[i], perimeter_coordinates[j]
    region_offset = np.asarray([region.bbox[0], region.bbox[1]])
    ptA += region_offset
    ptB += region_offset
    return pairwise_distances_matrix[i, j], ptA, ptB