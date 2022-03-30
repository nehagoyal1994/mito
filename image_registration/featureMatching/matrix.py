import cv2, math, imutils
import numpy as np
import os





def matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp):
    
    # compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = tmp.shape[:2]

    aligned = cv2.warpPerspective(img, H, (w, h))
    aligned_mask = cv2.warpPerspective(maskImg, H, (w, h))
    return aligned, aligned_mask



def matrixAffineAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp):
    
    # compute the affine matrix between the two sets of matched points
    (H, mask) = cv2.estimateAffinePartial2D(ptsA, ptsB)

    # use the homography matrix to align the images
    (h, w) = tmp.shape[:2]
        
    aligned = cv2.warpAffine(img, H, (w, h))
    aligned_mask = cv2.warpAffine(maskImg, H, (w, h))
    return aligned, aligned_mask
