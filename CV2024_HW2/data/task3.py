import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def NCC(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))


def align(a, b, range_x, range_y):
    max_ncc = -np.inf
    
    for i in range(-range_x, range_x+1):
        for j in range(-range_y, range_y+1):
            ncc = NCC(a, np.roll(b, [i, j], axis=(0,1)))
            if ncc > max_ncc:
                max_ncc = ncc
                max_shift = [i, j]
    return max_shift

dir_name = 'task3_colorizing'
save_dir_name = 'output'
img_list = os.listdir(dir_name)
if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)

for img_name in img_list:
    
    print(f"proposing {img_name}")
    name, imgtp = img_name.split('.')
    
    # Read image
    img = cv2.imread(dir_name + "\\" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove border
    h, w = img.shape
    img = img[int(h*0.02):int(h-h*0.02), int(w*0.02):int(w-w*0.02)]

   
    h, w = img.shape
    height = int(h/3)
    blue = img[0:height, :]
    green = img[height:2*height, :]
    red = img[2*height:3*height, :]
    if h > 3000: # 檔案太大，縮小點加速
        issmall = True
        img = cv2.resize(img, (int(w/10), int(h/10)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape
        height = int(h/3)
        blue_use = img[0:height, :]
        green_use = img[height:2*height, :]
        red_use = img[2*height:3*height, :]
    else:
        issmall = False
        blue_use = blue
        green_use = green
        red_use = red
    
    shift_green = align(blue_use, green_use, 20, 20)
    shift_red = align(blue_use, red_use, 20, 20)
    if issmall:
        shift_green = [shift_green[0]*10, shift_green[1]*10]
        shift_red = [shift_red[0]*10, shift_red[1]*10]
        
    
    align_green = np.roll(green, shift_green, axis=(0, 1))
    align_red = np.roll(red, shift_red, axis=(0, 1))
    

    colored = cv2.merge([align_red, align_green, blue])
    colored = colored[int(colored.shape[0]*0.05):int(colored.shape[0]-colored.shape[0]*0.05), int(colored.shape[1]*0.05):int(colored.shape[1]-colored.shape[1]*0.05)]
    plt.imshow(colored)
    plt.axis("off")
    plt.show()
    plt.imsave(save_dir_name + "/" + name + ".jpg", colored)
   
    
    