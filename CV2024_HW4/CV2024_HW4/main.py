import cv2
import numpy as np
import matplotlib.pyplot as plt
import random



def detect_and_match_features(img1, img2, ratio_thresh=0.75):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.figure(figsize=(15, 8))
    # plt.imshow(img_matches)
    # plt.title("Feature Matches between Image 1 and Image 2")
    # plt.axis('off')
    # plt.show()
    
    return src_pts, dst_pts, good_matches


def normalize_points(points):
    mean = np.mean(points, axis=0)
    rms = np.sqrt(np.mean(np.linalg.norm(points - mean, axis=1) ** 2))
    scale = np.sqrt(2) / rms
    T = np.array([[scale, 0, -scale * mean[0]], 
                  [0, scale, -scale * mean[1]], 
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.hstack([points, np.ones((points.shape[0], 1))]).T).T[:, :2]
    return normalized_points, T

def compute_fundamental_matrix(src_pts, dst_pts, ransac_iters=1000, threshold=0.01):
    src_pts_norm, T1 = normalize_points(src_pts)
    dst_pts_norm, T2 = normalize_points(dst_pts)

    best_F = None
    best_inliers = []

    for _ in range(ransac_iters):
        sample_indices = random.sample(range(src_pts_norm.shape[0]), 8)
        src_sample = src_pts_norm[sample_indices]
        dst_sample = dst_pts_norm[sample_indices]

        A = []
        for i in range(src_sample.shape[0]):
            x1, y1 = src_sample[i]
            x2, y2 = dst_sample[i]
            A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
        A = np.array(A)

        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)

        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0
        F = np.dot(U, np.dot(np.diag(S), Vt))


        F = np.dot(T2.T, np.dot(F, T1))

        inliers = []
        for i in range(src_pts.shape[0]):
            x1, y1 = src_pts[i]
            x2, y2 = dst_pts[i]
            epipolar_error = np.abs(np.dot(np.dot([x2, y2, 1], F), [x1, y1, 1]))
            if epipolar_error < threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_F = F
            best_inliers = inliers
    

    inlier_pts1 = src_pts[best_inliers]
    inlier_pts2 = dst_pts[best_inliers]


    return best_F, inlier_pts1, inlier_pts2



def compute_epilines(points, F, which_image):
    
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  
    if which_image == 1:
        lines = F.T @ points_hom.T  
    else:
        lines = F @ points_hom.T  
    return lines.T  


def draw_epilines(img1, img2, pts1, pts2, F):
    lines1 = compute_epilines(pts2, F, which_image=1)  

    img1_copy = img1.copy()
    img2_copy = img2.copy()

    for r, pt1 in zip(lines1, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])  
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])  
        img1_copy = cv2.line(img1_copy, (x0, y0), (x1, y1), color, 1)  

    for pt2 in pts2:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt2 = (int(pt2[0]), int(pt2[1]))
        img2_copy = cv2.circle(img2_copy, pt2, 5, color, -1)  

    return img1_copy, img2_copy


def compute_essential_matrix(F, K1, K2):
    E = K1.T @ F @ K2

    return E


def extract_projection_matrices(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    P1 = np.hstack((R1, t.reshape(-1, 1)))
    P2 = np.hstack((R1, -t.reshape(-1, 1)))
    P3 = np.hstack((R2, t.reshape(-1, 1)))
    P4 = np.hstack((R2, -t.reshape(-1, 1)))

    return [P1, P2, P3, P4]


def triangulate_point(P1, P2, pt1, pt2):
    
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    X = X_hom[:3] / X_hom[3]  
    return X

def triangulate_points(P1, P2, pts1, pts2):

    points_3d = []
    for pt1, pt2 in zip(pts1, pts2):
        X = triangulate_point(P1, P2, pt1, pt2)
        points_3d.append(X)
    return np.array(points_3d)

def select_best_projection(P_list, pts1, pts2, K1, K2):
    max_in_front = -1
    best_P = None

    P0 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  

    for P in P_list:
        P1 = K2 @ P

        points_3d = triangulate_points(P0, P1, pts1, pts2)

        Z1 = points_3d[:, 2]  
        Z2 = (P[:3, :3] @ points_3d.T + P[:, 3].reshape(-1, 1)).T[:, 2]  

        num_in_front = np.sum((Z1 > 1e-6) & (Z2 > 1e-6))

        if num_in_front > max_in_front:
            max_in_front = num_in_front
            best_P = P

    return best_P




def triangulate_and_visualize(best_P, pts1, pts2, K1, K2):
    P0 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K2 @ best_P
    points_3d = []
    for pt1, pt2 in zip(pts1, pts2):
        
        
        A = np.zeros((4, 4))
        A[0] = pt1[0] * P0[2] - P0[0]
        A[1] = pt1[1] * P0[2] - P0[1]
        A[2] = pt2[0] * P1[2] - P1[0]
        A[3] = pt2[1] * P1[2] - P1[1]

       
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        X = X_hom[:3] / X_hom[3]  
        points_3d.append(X)

    points_3d = np.array(points_3d)
    # points_4d_hom = cv2.triangulatePoints(P0, K @ best_P, pts1.T, pts2.T)
    # points_3d = points_4d_hom[:3] / points_4d_hom[3]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=40)
    plt.show()

    return points_3d


# img1 = cv2.imread('./data/Mesona1.JPG')
# img2 = cv2.imread('./data/Mesona2.JPG')
# K1 = np.array([[1.4219, 0.0005, 0.5092],
#               [0, 1.4219, 0.3802],
#               [0, 0, 0.001]])
# K2 = np.array([[1.4219, 0.0005, 0.5092],
#               [0, 1.4219, 0.3802],
#               [0, 0, 0.001]])

# img1 = cv2.imread('./data/Statue1.bmp')
# img2 = cv2.imread('./data/Statue2.bmp')
# K1 = np.array([[5426.566895, 0.678017, 330.096680],
#               [0, 5423.133301, 648.950012],
#               [0, 0, 1.000000]])
# K2 = np.array([[5426.566895, 0.678017, 387.430023],
#               [0, 5423.133301, 620.616699],
#               [0, 0, 1.000000]])

img1 = cv2.imread('./my_data/SantaHat1.jpg')
img2 = cv2.imread('./my_data/SantaHat2.jpg')

K1 = np.array([[2701.93, 0.000, 1538.21],
                [0, 2738.09, 1960.13],
                [0, 0, 1]])
K2 = np.array([[2701.93, 0.000, 1538.21],
                [0, 2738.09, 1960.13],
                [0, 0, 1]])



src_pts, dst_pts, good_matches = detect_and_match_features(img1, img2)
F, inlier_pts1, inlier_pts2 = compute_fundamental_matrix(src_pts, dst_pts)
img1_epi, img2_points = draw_epilines(img1, img2, inlier_pts1, inlier_pts2, F)

plt.subplot(121)
plt.imshow(img1_epi, cmap='gray')
plt.title("Image 1 with Epilines")
plt.axis('off')

plt.subplot(122)
plt.imshow(img2_points, cmap='gray')
plt.title("Image 2 with Points")
plt.axis('off')
plt.show()

E = compute_essential_matrix(F, K1, K2)
P_list = extract_projection_matrices(E)
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))


best_P = select_best_projection(P_list, inlier_pts1, inlier_pts2, K1, K2)
points_3d = triangulate_and_visualize(best_P, inlier_pts1, inlier_pts2, K1, K2)


print(f'3d size: {len(points_3d)}, 2d size: {len(inlier_pts1)}')
# folder = 'Mesona'
# folder = 'Statue'
folder = 'SantaHat'

np.savetxt('./output/' + folder + '/points_3d.txt', points_3d, delimiter=',')
np.savetxt('./output/' + folder + '/points_2d.txt', inlier_pts1, delimiter=',')
np.savetxt('./output/' + folder + '/camera_matrix1.txt', K1, delimiter=',')
np.savetxt('./output/' + folder + '/camera_matrix2.txt', K2, delimiter=',')

