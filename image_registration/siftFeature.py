import cv2, math, imutils
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from datetime import datetime

from scipy.spatial import distance
from numpy import mean

from matrix import matrixHomographyAligned, matrixAffineAligned
from matcher import BfMatcher, FlannMatcher
from diceScore import compute_dice




def sift_feature(img,tmp, maskImg, maskTmp, matcher, feature, matrix, mask):    
       
    sift = cv2.SIFT_create()
    
    if (mask == True):        
        (kpsA, descsA) = sift.detectAndCompute(img, mask=maskImg )      
        (kpsB, descsB) = sift.detectAndCompute(tmp, mask=maskTmp )
        
    else:
        (kpsA, descsA) = sift.detectAndCompute(img, None)        
        (kpsB, descsB) = sift.detectAndCompute(tmp, None) 
    
    
    if(matcher == "BF"):
        ptsA, ptsB, good = BfMatcher(kpsA, descsA, kpsB, descsB, feature)
        
    if (matcher == "FLANN"):
        ptsA, ptsB, good = FlannMatcher(kpsA, descsA, kpsB, descsB, feature)    
    
    if (matrix=="HOMOGRAPHY"):
        # use the homography matrix to align the images
        aligned, aligned_mask = matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)
    
    if (matrix =="AFFINE"):
        aligned, aligned_mask = matrixAffineAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)

    matchedImg = cv2.drawMatches(imgA,kpsA,imgB,kpsB,good[:10],None)
    
    # return the aligned image
    return aligned, aligned_mask, matchedImg





def FeatureRegistration(IMAGES, MASKS, feature, matcher, matrix, mask):
    
    dice_score_feature = []
    

    for i in range(1, len(IMAGES)):
        print(i)
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        tmp = cv2.imread(IMAGES[i-1], cv2.IMREAD_GRAYSCALE)

        maskImg = cv2.imread(MASKS[i], cv2.IMREAD_GRAYSCALE)
        maskTmp = cv2.imread(MASKS[i-1], cv2.IMREAD_GRAYSCALE)
        
       
        
        aligned, aligned_mask, matchedImg = sift_feature(img,tmp, maskImg, maskTmp, matcher, feature, matrix, mask)
        
        
        dice_cal = compute_dice(maskTmp, aligned_mask)
        dice_score_feature.append(dice_cal)
        
        print("Dice Score = " + str(dice_cal))
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1,5, 1)
        plt.imshow(img)
        plt.title("source image")
        plt.subplot(1, 5, 2)
        plt.imshow(tmp)
        plt.title("target image")
        plt.subplot(1, 5, 3)
        plt.imshow(matchedImg)
        plt.title("matched keypoints")
        plt.subplot(1, 5, 4)
        plt.imshow(aligned)
        plt.title("warped image")
        plt.subplot(1, 5, 5)
        plt.imshow(aligned_mask)
        plt.title("warped mask")
        plt.show()
    
    avgDice = mean(dice_score_feature)
    
        
    print("Avg dice: " + str(avgDice))
    
    
    if mask == True:
        plt.title(str(feature) + " feature using " + str(matrix) + " matrix and " + str(matcher) + " matcher with MITO")
    else:
        plt.title(str(feature) + " feature using " + str(matrix) + " matrix and " + str(matcher) + " matcher without MITO")
    x = [i for i in range(1, len(IMAGES))]
    y = dice_score_feature
    plt.ylabel("dice score")
    plt.xlabel("image pair")
    plt.bar(x, y)
    plt.show()
    