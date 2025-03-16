import cv2
import numpy as np
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

img_dir = './hw2/original_image'
result_dir = './hw2/result'

def get_Gaussion_blur(img, k_size = 5, sigma = 1):
    g_1D = cv2.getGaussianKernel(k_size, sigma)
    g_2D = np.outer(g_1D, g_1D.T)
    B, G, R= img[:, :, 0], img[:, :, 1], img[:, :, 2]
    B = convolve2d(B, g_2D, mode = 'same', boundary= 'symm')
    G = convolve2d(G, g_2D, mode = 'same', boundary= 'symm')
    R = convolve2d(R, g_2D, mode = 'same', boundary= 'symm')
    B = np.clip(B, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    R = np.clip(R, 0, 255).astype(np.uint8)
    return np.dstack([B, G, R])

def hybric_image(img1, img2):
    tmp_img1 = img1.astype(np.float32)
    img1_high = np.clip(tmp_img1*1.3 - get_Gaussion_blur(tmp_img1, k_size=11, sigma=3), 0, 255).astype(np.float32)
    img2_low = get_Gaussion_blur(img2, 21, 11).astype(np.float32)
    return np.clip(((2/5)*img1_high +  (3/5)* img2_low), 0, 255).astype(np.uint8)

    

def part2_1():
    img = cv2.imread(os.path.join(img_dir, 'taj.jpg'))
    blur_img = get_Gaussion_blur(img, k_size=5, sigma=1)
    img = img.astype(np.int16)
    img_high = np.clip(img-blur_img, 0, 255)
    img_sharp =np.clip(img + img_high*1.5, 0, 255).astype(np.uint8)
    img_sharp_rgb = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
    plt.imshow(img_sharp_rgb)
    plt.show()
    cv2.imwrite(os.path.join(result_dir, 'sharpen.jpg'), img_sharp)

def part2_2():
    img1 = cv2.imread(os.path.join(img_dir, 'DerekPicture.jpg'))
    img2 = cv2.imread(os.path.join(img_dir, 'nutmeg.jpg'))
    img2 = img2[:1024, 250:982, :]
    
    result = hybric_image(img1, img2)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result_rgb)
    plt.show()
    cv2.imwrite(os.path.join(result_dir, 'hybric_Derek.jpg'), result)


#part2_1()
part2_2()