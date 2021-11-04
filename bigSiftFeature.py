import cv2, math, imutils
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.spatial import distance
from numpy import mean


def matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp):
    
    # compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = tmp.shape[:2]
    (new_h, new_w) = maskTmp.shape[:2]
    
    aligned = np.zeros_like(img, dtype=img.dtype)
    aligned_mask = np.zeros_like(maskImg, dtype=maskImg.dtype)
    aligned = cv2.warpPerspective(img, H, (w, h))
    aligned_mask = cv2.warpPerspective(maskImg, H, (new_w, new_h))
    return aligned, aligned_mask


def matrixAffineAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp):
    
    # compute the affine matrix between the two sets of matched points
    (H, mask) = cv2.estimateAffinePartial2D(ptsA, ptsB)

    # use the homography matrix to align the images
    (h, w) = tmp.shape[:2]
    (new_h, new_w) = maskTmp.shape[:2]
    
    aligned = np.zeros_like(img, dtype=img.dtype)
    aligned_mask = np.zeros_like(maskImg, dtype=maskImg.dtype)
    aligned = cv2.warpAffine(img, H, (w, h))
    aligned_mask = cv2.warpAffine(maskImg, H, (new_w, new_h))
    return aligned, aligned_mask




def sift_feature(img,tmp, maskImg, maskTmp, matcher, feature, matrix, mask):
    
    start = time.time()
    
    imgA = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    imgB = cv2.cvtColor(tmp.copy(), cv2.COLOR_BGR2GRAY)

    maskA = cv2.cvtColor(maskImg.copy(), cv2.COLOR_BGR2GRAY)
    maskB = cv2.cvtColor(maskTmp.copy(), cv2.COLOR_BGR2GRAY)
   
    brisk = cv2.SIFT_create()
    
    if (mask == True):
        
        (kpsA, descsA) = brisk.detectAndCompute(imgA, mask=maskA )      
        (kpsB, descsB) = brisk.detectAndCompute(imgB, mask=maskB )
        
    else:
        (kpsA, descsA) = brisk.detectAndCompute(imgA, None)        
        (kpsB, descsB) = brisk.detectAndCompute(imgB, None) 

    if(matcher == "BF"):
        distance = cv2.NORM_L1
        matcher = cv2.BFMatcher(distance,crossCheck=True)
        matches = matcher.match(descsA,descsB)

        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        good = []
        good = matches[:10]

        ptsA = np.zeros((len(good), 2), dtype="float")
        ptsB = np.zeros((len(good), 2), dtype="float")

        # loop over the top matches
        for (i, m) in enumerate(good):
            # indicate that the two keypoints in the respective images map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
        
    if (matcher == "FLANN"):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        # Create FLANN object 
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # Matching descriptor vectors using FLANN Matcher 
        matches = matcher.knnMatch(descsA,descsB, k=2)

        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        # Apply ratio test
        good = []
        for m in matches:
            if len(m) ==2 and m[0].distance < 0.8*m[1].distance:
                good.append(m[0])

        good = sorted(good, key=lambda x:x.distance)

        ptsA = np.float32([ kpsA[m.queryIdx].pt for m in good[:10] ]).reshape(-1,1,2)
        ptsB = np.float32([ kpsB[m.trainIdx].pt for m in good[:10] ]).reshape(-1,1,2)

    matchedVis = cv2.drawMatches(imgA,kpsA,imgB,kpsB,good[:10],None)
    matchedVis = imutils.resize(matchedVis, width=1000)
    
    if (matrix=="HOMOGRAPHY"):
        # use the homography matrix to align the images
        aligned, aligned_mask = matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)
    
    if (matrix =="AFFINE"):
        aligned, aligned_mask = matrixAffineAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)

    end = time.time()
    detection_time = end-start
    # return the aligned image
    return aligned, aligned_mask, matchedVis, detection_time



def compute_dice(label_img, pred_img, p_threshold=0.5):
    p = pred_img.astype(np.float)
    l = label_img.astype(np.float)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val 


def FeatureRegistration(IMAGES, MASKS, feature, matcher, matrix, df, mask):
    dice_score_feature = []
    time_list = []
    start = time.time()
#     start = datetime.now()
    for i in range(1, len(IMAGES)):
        print(i)
        img = cv2.imread(IMAGES[i])
        tmp = cv2.imread(IMAGES[i-1])

        maskImg = cv2.imread(MASKS[i])
        maskTmp = cv2.imread(MASKS[i-1])
        
        
        aligned, aligned_mask, matched_points, detected_time = sift_feature(img,tmp, maskImg, maskTmp, matcher, feature, matrix, mask)   
#         if (feature == "SIFT" and mask == False):
#             aligned, aligned_mask, matched_points, detected_time = sift_feature(img,tmp, maskImg, maskTmp, matcher, feature, matrix, mask = False)   
        

        dice_cal = compute_dice(maskTmp, aligned_mask)
        dice_score_feature.append(dice_cal)
        time_list.append(detected_time)
        print("Dice Score = " + str(dice_cal))
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1,5, 1)
#         plt.imshow(img)
#         plt.title("img to be registered")
#         plt.subplot(1, 5, 2)
#         plt.imshow(tmp)
#         plt.title("image accordingly")
#         plt.subplot(1, 5, 3)
#         plt.imshow(matched_points)
#         plt.title("checkpoint image")
#         plt.subplot(1, 5, 4)
#         plt.imshow(aligned)
#         plt.title("warped image")
#         plt.subplot(1, 5, 5)
#         plt.imshow(aligned_mask)
#         plt.title("warped mask")
#         plt.show()
    end = time.time()
#     end = datetime.now()
    avg_dice = mean(dice_score_feature)
    avg_detectedTime = mean(time_list)
    (h,w) = tmp.shape[:2]
    resolution = h * w / 1000000
    throughput_pair =  resolution / avg_detectedTime
    executionTime = end - start
    throughputStack = (resolution * len(IMAGES)) / executionTime
    print("Avg : " + str(avg_dice))
    print("Execution time : " + str(executionTime))
    
    if mask == True:
        plt.title(str(feature) + " " + str(matrix) + " feature with MITO")
    else:
        plt.title(str(feature) + " " + str(matrix) + " feature without MITO")
    x = [i for i in range(1, len(IMAGES))]
    y = dice_score_feature
    plt.ylabel("dice score")
    plt.xlabel("image pair")
    plt.bar(x, y)
    plt.show()
    df = df.append({'avgDice': avg_dice,
                    'throughputPair  MP/s': throughput_pair,
                    'ExecutionTime  sec': executionTime,
                    'throughputStack  MP/s': throughputStack,
                    'feature': feature,
                    'matcher': matcher,
                    'matrix': matrix,
                    'mito': mask,
                   }
                   ,ignore_index=True
                  )
    return  dice_score_feature, df