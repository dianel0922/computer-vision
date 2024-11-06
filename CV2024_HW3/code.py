import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

img_name = 'hill' #global call file name

def ratio_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2) / (np.linalg.norm(descriptor1) + np.linalg.norm(descriptor2) + 1e-10)

def match(descriptors1, descriptors2):
    matches = []
    tmp = []
    for li in enumerate(descriptors1):
        tmp.append(li)
    feature_size = li[0]
    print(f'the feature number of descriptors1 is {li[0]}')
    tmp = []
    for li in enumerate(descriptors2):
        tmp.append(li)
    print(f'the feature number of descriptors2 is {li[0]}')
    print('start matching feature...')
    for bar, (i, desc1) in zip(tqdm(range(1, feature_size)), enumerate(descriptors1)):
        distances = np.array([ratio_distance(desc1, desc2) for desc2 in descriptors2])
        sorted_indices = np.argsort(distances)[:2]
        matches.append([cv2.DMatch(_queryIdx=i, _trainIdx=idx, _imgIdx=0, _distance=distances[idx]) for idx in sorted_indices])
        
    return matches

def detect_and_match_features(img1, img2, feature='SIFT'):

    # step 1
    if feature == "SIFT":
        print("start run SIFT")
        detector = cv2.SIFT_create()
        
        print("processing image1")
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        
        print("processing image2")
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        
        print("processing complete")
        
    elif feature=='MSER':
        print("start run MSER")
        mser = cv2.MSER_create(min_area=20, max_area=14400)
        sift = cv2.SIFT_create()
        
        print("processing image1")
        keypoints1 = mser.detect(img1)
        keypoints1, descriptors1 = sift.compute(img1, keypoints1)
        
        print("processing image2")
        keypoints2 = mser.detect(img2)
        keypoints2, descriptors2 = sift.compute(img2, keypoints2)
        
        print("processing complete")
        
    else:
        raise ValueError(f"暫不支持{feature}方法，請選擇 'SIFT'or 'MSER'。")
        
        
    #step 2
    
    matches = match(descriptors1, descriptors2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f'the feature number that  is good matching is {len(good_matches)}')
    
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("Feature Matching")
    plt.axis("off")
    plt.show()
    plt.imsave('out/' + feature + '_' + img_name + '_match.jpg', img_matches)

    return keypoints1, keypoints2, good_matches

def ransac_homography(keypoints1, keypoints2, good_matches, threshold=5.0, max_iters=1000):
    # pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold, maxIters=max_iters)
    # inliers = mask.ravel().tolist()

    def sample_correspondences(matches, keypoints1, keypoints2, S=4):
        sampled_matches = random.sample(matches, S)
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in sampled_matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in sampled_matches])
        return pts1, pts2
    
    def compute_homography(pts1, pts2):
        A = []
        for i in range(4):
            X, Y = pts1[i][0], pts1[i][1]
            u, v = pts2[i][0], pts2[i][1]
            A.append([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
            A.append([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])    

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2,2]

    def count_inliers(H, keypoints1, keypoints2, matches, threshold=5.0):
        inliers = 0
        for match in matches:
            pt1 = np.array([*keypoints1[match.queryIdx].pt, 1.0])
            pt2 = np.array([*keypoints2[match.trainIdx].pt, 1.0])
            
            projected_pt1 = H @ pt1
            projected_pt1 /= projected_pt1[2]  
            error = np.linalg.norm(projected_pt1[:2] - pt2[:2])
            if error < threshold:
                inliers += 1
        return inliers
    
    # IV. iterate for N times
    def ransac_homography(matches, keypoints1, keypoints2, threshold=5.0, max_iters=1000):
        best_H = None
        max_inliers = 0

        for _ in range(max_iters):
            pts1, pts2 = sample_correspondences(matches, keypoints1, keypoints2)
            H = compute_homography(pts1, pts2)
            
            inliers = count_inliers(H, keypoints1, keypoints2, matches, threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H

        return best_H

    # V. get the best homography matrix with smallest number of outliers
   
    return ransac_homography(good_matches, keypoints1, keypoints2, threshold=5.0, max_iters=1000)
    

def warp_and_stitch(img1, img2, H):
    h2, w2 = img2.shape[:2]
    corners_img1 = np.float32([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]]).reshape(-1, 1, 2)
    
    corners_img1_enpand = np.concatenate((corners_img1, np.ones((4, 1, 1))), axis=2)
    warped_corners_img1 = []
    for i in range(len(corners_img1_enpand)):
        temp = H @ corners_img1_enpand[i].T
        temp /= temp[2,0]
        warped_corners_img1.append(temp[0:2].T)
    
    corners = np.concatenate((warped_corners_img1, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img1, H_translation @ H, (x_max - x_min, y_max - y_min))
    
    result = np.zeros((y_max - y_min, x_max - x_min, 3))
    temp = H_translation @ H
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            newtemp = temp @ np.array([[j],[i],[1]])
            newtemp /= newtemp[2,0]
            result[int(newtemp[1,0]),int(newtemp[0,0])] = img1[i,j]
    temp = result

    for i in range(len(temp)):
        for j in range(len(temp[0])):
            if temp[i,j].sum() == 0:
                count = 0
                for x in range(-1,2):
                    for y in range(-1,2):
                        if result[min(len(temp)-1,max(0,i+x)),min(len(temp[0])-1,max(0,j+y))].sum() != 0 :
                            count += 1
                            result[i,j] += result[min(len(temp)-1,max(0,i+x)),min(len(temp[0])-1,max(0,j+y))]/25
                if count != 0:
                    result[i,j] /= count
                    result[i,j] *= 25
    
    result[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    return result.astype("uint8")


img1 = cv2.imread('data/'+ img_name +'1.jpg', cv2.COLOR_RGB2BGR)
img2 = cv2.imread('data/'+ img_name +'2.jpg', cv2.COLOR_RGB2BGR)

features = ['SIFT', 'MSER']
feature = features[1]
# step 1,2
keypoints1, keypoints2, good_matches = detect_and_match_features(img1, img2, feature)


# step3
H = ransac_homography(keypoints1, keypoints2, good_matches)
print("最佳單應矩陣 H：")
print(H)

# step 4
result = warp_and_stitch(img1, img2, H)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.title("Panoramic Stitching Result")
plt.show()
plt.imsave('output/' + feature + '_'+ img_name +'_result.jpg', result)