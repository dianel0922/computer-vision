import cv2
import numpy as np
import matplotlib.pyplot as plt


def ratio_distance(descriptor1, descriptor2):
    # 計算自定義的 ratio distance
    return np.linalg.norm(descriptor1 - descriptor2) / (np.linalg.norm(descriptor1) + np.linalg.norm(descriptor2) + 1e-10)

def match(descriptors1, descriptors2):
    matches = []
    for i, desc1 in enumerate(descriptors1):
        distances = np.array([ratio_distance(desc1, desc2) for desc2 in descriptors2])
        sorted_indices = np.argsort(distances)[:2]
        matches.append([cv2.DMatch(_queryIdx=i, _trainIdx=idx, _imgIdx=0, _distance=distances[idx]) for idx in sorted_indices])
    return matches

def detect_and_match_features(img1, img2, feature='SIFT'):

    # step 1
    if feature == "SIFT":
        detector = cv2.SIFT_create()
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    elif feature=='MSER':
        mser = cv2.MSER_create(min_area=20, max_area=14400)
        keypoints1 = mser.detect(img1)
        keypoints2 = mser.detect(img2)
    
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.compute(img1, keypoints1)
        keypoints2, descriptors2 = sift.compute(img2, keypoints2)
        
    else:
        raise ValueError(f"暫不支持{feature}方法，請選擇 'SIFT', 'MSER'。")
        
        
    #step 2
    
    matches = match(descriptors1, descriptors2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("Feature Matching")
    plt.axis("off")
    plt.show()

    return keypoints1, keypoints2, good_matches

def ransac_homography(keypoints1, keypoints2, good_matches, threshold=5.0, max_iters=1000):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold, maxIters=max_iters)

    inliers = mask.ravel().tolist()

    return H, inliers

def warp_and_stitch(img1, img2, H):
    h2, w2 = img2.shape[:2]
    
    corners_img1 = np.float32([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]]).reshape(-1, 1, 2)
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    corners = np.concatenate((warped_corners_img1, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)

    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img1, H_translation @ H, (x_max - x_min, y_max - y_min))
    result[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2

    return result




img1 = cv2.imread('data/TV1.jpg', cv2.COLOR_BGR2RGB)
img2 = cv2.imread('data/TV2.jpg', cv2.COLOR_BGR2RGB)

features = ['SIFT', 'MSER']
# step 1,2
keypoints1, keypoints2, good_matches = detect_and_match_features(img1, img2, features[0])


# step3
H, inliers = ransac_homography(keypoints1, keypoints2, good_matches)
print("最佳單應矩陣 H：")
print(H)

# step 4
result = warp_and_stitch(img1, img2, H)

plt.figure(figsize=(10, 5))
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.title("Panoramic Stitching Result")
plt.show()
plt.imsave('out.jpg', result)