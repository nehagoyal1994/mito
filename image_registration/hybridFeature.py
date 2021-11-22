import cv2, math, imutils
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from datetime import datetime

from scipy.spatial import distance
from numpy import mean

from matrix import matrixHomographyAligned
from matcher import BfMatcher, FlannMatcher
from diceScore import compute_dice





def hybrid_feature(img,tmp, maskImg, maskTmp, matcher, detector, descriptor, mask =False): 
    
    
    if detector == "FAST":
        detector = cv2.FastFeatureDetector_create()
    if detector == "ORB":
        detector = cv2.ORB_create()
    if descriptor  == "BRISK":
        descriptor = cv2.BRISK_create()
    if descriptor  == "FREAK":
        descriptor = cv2.xfeatures2d_FREAK.create()
    
    
    if (mask == True):
        kpsA = detector.detect(img, mask=maskImg)
        (kpsA, descsA) = descriptor.compute(img, kpsA) 
        kpsB = detector.detect(tmp, mask=maskTmp)
        (kpsB, descsB) = descriptor.compute(tmp, kpsB) 
        
    else:
        kpsA = detector.detect(img, None)
        (kpsA, descsA) = descriptor.compute(img, kpsA) 
        kpsB = detector.detect(tmp, None)
        (kpsB, descsB) = descriptor.compute(tmp, kpsB)
        

    if(matcher == "BF"):
        ptsA, ptsB, good = BfMatcher(kpsA, descsA, kpsB, descsB, descriptor)
        
    if (matcher == "FLANN"):
        ptsA, ptsB, good = FlannMatcher(kpsA, descsA, kpsB, descsB, descriptor)        
        
    
    # use the homography matrix to align the images
    aligned, aligned_mask = matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)
    
    matchedImg = cv2.drawMatches(img,kpsA,tmp,kpsB,good[:10],None)

    
    # return the aligned image
    return aligned, aligned_mask, matchedImg
 




def FeatureRegistration(IMAGES, MASKS, detector, descriptor, matcher, mask):
    
    dice_score_feature = []   
    
    for i in range(1, len(IMAGES)):
        print(i)
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        tmp = cv2.imread(IMAGES[i-1], cv2.IMREAD_GRAYSCALE)

        maskImg = cv2.imread(MASKS[i], cv2.IMREAD_GRAYSCALE)
        maskTmp = cv2.imread(MASKS[i-1], cv2.IMREAD_GRAYSCALE)
                
        
        aligned, aligned_mask, matchedImg = hybrid_feature(img,tmp, maskImg, maskTmp, matcher, detector, descriptor, mask)         
           
        
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
        plt.title(str(detector) + " + " + str(descriptor) + " feature using " + str(matcher) + " matcher with MITO")
    else:
        plt.title(str(detector) + " + " + str(descriptor)  + " feature using " + str(matcher) + " matcher without MITO")
    x = [i for i in range(1, len(IMAGES))]
    y = dice_score_feature
    plt.ylabel("dice score")
    plt.xlabel("image pair")
    plt.bar(x, y)
    plt.show()
    
