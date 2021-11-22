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
from diceScore import compute_dice



def features(maskGray, imgGray):

    contours= cv2.findContours(maskGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    
    feature = []
    
    for cnt in contours[0]:
        area = cv2.contourArea(cnt)
        list1 =[]        
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        n = approx.ravel() 
        
        i = 0 
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                
                list1.append([x,y])
            i = i + 1
        list1.append(area)
        feature.append(list1)
    feature = sorted(feature, key=lambda x: x[-1])  
    
        
    return feature


def map_features(img, tmp, maskImg, maskTmp, feature, matcher):    
    
    kpA = features(maskImg, img)
    kpB = features(maskTmp, tmp)
    
    if (feature == "BRISK"):
        feature = cv2.BRISK_create()
    
    if (feature == "FREAK"):
        feature = cv2.xfeatures2d_FREAK.create()
    

    kpsA = [cv2.KeyPoint(x=float(f[i][0]), y=float(f[i][1]), size=float(f[-1])) for f in kpA for i in range(len(f) - 1) ]
    kpsB = [cv2.KeyPoint(x=float(f[i][0]), y=float(f[i][1]), size=float(f[-1])) for f in kpB for i in range(len(f) - 1) ]
    
    kpsA, descsA = feature.compute(img, kpsA)
    kpsB, descsB = feature.compute(tmp, kpsB)

    if(matcher == 'BF'):        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descsA,descsB)

        # sort the matches by their distance (the smaller the distance, the more similar the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        good = []
        good = matches

        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        
        # loop over the top matches
        for (i, m) in enumerate(matches):            
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
    
    if(matcher == 'FLANN'):        
        FLANN_INDEX_KDTREE = 6
             
        index_params= dict(algorithm = FLANN_INDEX_KDTREE,
                                       table_number = 6, # 12
                                       key_size = 12,     # 20
                                       multi_probe_level = 1) #2
        search_params = dict()

        # Create FLANN object 
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Matching descriptor vectors using FLANN Matcher 
        matches = flann.knnMatch(descsA,descsB,k=2)

        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        # Apply ratio test
        good = []
        for m in matches:
            if len(m) ==2 and m[0].distance < 0.8*m[1].distance:
                good.append(m[0])
        
        good = sorted(good, key=lambda x:x.distance)
        
        ptsA = np.float32([ kpsA[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        ptsB = np.float32([ kpsB[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)   
    
    

    aligned, aligned_mask = matrixHomographyAligned(ptsA, ptsB, img, tmp, maskImg, maskTmp)
    
    matchedImg = cv2.drawMatches(imgA,kpsA,imgB,kpsB,good,None)
    return aligned, aligned_mask, matchedImg






def MitoRegistration(IMAGES, MASKS, feature, matcher):
    
    dice_score_feature_mito = []
    
    
    for i in range(1, len(IMAGES)):
        print(i)
        
        img = cv2.imread(IMAGES[i], cv2.IMREAD_GRAYSCALE)
        tmp = cv2.imread(IMAGES[i-1], cv2.IMREAD_GRAYSCALE)

        maskImg = cv2.imread(MASKS[i], cv2.IMREAD_GRAYSCALE)
        maskTmp = cv2.imread(MASKS[i-1], cv2.IMREAD_GRAYSCALE)
        
        
        aligned, aligned_mask, matchedImg = map_features(img, tmp, maskImg, maskTmp, feature, matcher)        
        
        
        dice_cal = compute_dice(maskTmp, aligned_mask)
        dice_score_feature_mito.append(dice_cal)
        
        
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
    
    
    avgDice = mean(dice_score_feature_mito)
    
    
    print("Avg dice: " + str(avgDice))
    
    
    plt.title(str(feature) + " descriptor with MITO detector using " + str(matcher) + " matcher")
    x = [i for i in range(1, len(IMAGES))]
    y = dice_score_feature_mito
    
    plt.ylabel("dice score")
    plt.xlabel("image pair")
    plt.bar(x, y)
    plt.show()
    
    
