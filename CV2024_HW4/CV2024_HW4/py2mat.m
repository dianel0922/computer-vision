%記得改path
dataPath = './output/SantaHat/'; 
img_path = './my_data'; %data/mydata
img_name = {'Mesona1.JPG'; 
            'Mesona2.JPG'; 
            'Statue1.bmp'; 
            'Statue2.bmp';
            'SantaHat1.jpg';
            'SantaHat2.jpg'};

points3dFile = fullfile(dataPath, 'points_3d.txt');
points2dFile = fullfile(dataPath, 'points_2d.txt');
cameraMatrixFile1 = fullfile(dataPath, 'camera_matrix1.txt');
cameraMatrixFile2 = fullfile(dataPath, 'camera_matrix2.txt');

points3d = readmatrix(points3dFile);
points2d = readmatrix(points2dFile);
cameraMatrix1 = readmatrix(cameraMatrixFile1);
cameraMatrix2 = readmatrix(cameraMatrixFile2);
%記得改圖片序號
obj_main(points3d, points2d, cameraMatrix1,fullfile(img_path, img_name{5}), 5, dataPath);
obj_main(points3d, points2d, cameraMatrix2, fullfile(img_path, img_name{6}), 6, dataPath);
