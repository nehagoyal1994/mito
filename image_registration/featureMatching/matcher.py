import cv2, math, imutils
import numpy as np
import os





def FlannMatcher(kpsA, descsA, kpsB, descsB, feature):
    
    if(feature =="SIFT"):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
    else:
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

    
    ptsA = np.float32([ kpsA[m.queryIdx].pt for m in good[:10] ]).reshape(-1,1,2)
    ptsB = np.float32([ kpsB[m.trainIdx].pt for m in good[:10] ]).reshape(-1,1,2)
    return ptsA, ptsB, good



def BfMatcher(kpsA, descsA, kpsB, descsB, feature):
    
    if(feature == 'SIFT'):
        distance = cv2.NORM_L1
    else:
        distance = cv2.NORM_HAMMING
    
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
     
    return ptsA, ptsB, good
