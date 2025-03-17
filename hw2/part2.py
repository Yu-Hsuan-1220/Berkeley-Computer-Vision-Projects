import cv2
import numpy as np
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

img_dir = './hw2/original_image'
result_dir = './hw2/result'

def get_Gaussian_blur(img, k_size = 5, sigma = 1):
    g_1D = cv2.getGaussianKernel(k_size, sigma)
    g_2D = np.outer(g_1D, g_1D.T)
    if len(img.shape) == 2:
        return np.clip(convolve2d(img, g_2D, mode = 'same', boundary='symm'), 0, 255).astype(np.float32)
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
    img1_high = np.clip(tmp_img1*1.3 - get_Gaussian_blur(tmp_img1, k_size=11, sigma=3), 0, 255).astype(np.float32)
    img2_low = get_Gaussian_blur(img2, 21, 11).astype(np.float32)
    return np.clip(((2/5)*img1_high +  (3/5)* img2_low), 0, 255).astype(np.uint8)

def get_Gassian_stack(img, level = 5, k_size = 5, sigma = 1):
    stack = [img.copy()]
    for _ in range(level):
        img = get_Gaussian_blur(img, k_size, sigma)
        stack.append(img.copy())
    return stack

def get_laplacian_stack(img, level = 5, k_size = 5, sigma = 1):
    g_stack = get_Gassian_stack(img, level, k_size, sigma)
    l_stack = [g_stack[i] - g_stack[i+1] for i in range(level-1)]
    l_stack.append(g_stack[-1])
    return l_stack

def part2_1():
    img = cv2.imread(os.path.join(img_dir, 'taj.jpg'))
    blur_img = get_Gaussian_blur(img, k_size=5, sigma=1)
    img = img.astype(np.int16)
    img_high = np.clip(img-blur_img, 0, 255)
    img_sharp =np.clip(img + img_high*1.5, 0, 255).astype(np.uint8)
    img_sharp_rgb = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_sharp_rgb)
    # plt.show()
    cv2.imwrite(os.path.join(result_dir, 'sharpen.jpg'), img_sharp)

def part2_2():
    img1 = cv2.imread(os.path.join(img_dir, 'DerekPicture.jpg'))
    img2 = cv2.imread(os.path.join(img_dir, 'nutmeg.jpg'))
    img2 = img2[:1024, 250:982, :]
    
    result = hybric_image(img1, img2)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result_rgb)
    plt.show()
    #cv2.imwrite(os.path.join(result_dir, 'hybrid_Derek.jpg'), result)

def part2_34():
    level = 5
    img1 = cv2.imread(os.path.join(img_dir, 'apple.jpeg'))
    img2 = cv2.imread(os.path.join(img_dir, 'orange.jpeg'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(img1)
    mask[:, :mask.shape[1]//2] = 1
    mask_stack = get_Gassian_stack(mask, level, 51, 31)
    
    l_stack1 = get_laplacian_stack(img1, level, 11, 7)
    l_stack2 = get_laplacian_stack(img2, level, 11, 7)

    laplacian_blend = []
    for i in range(level):
        laplacian_blend.append(l_stack1[i]*mask_stack[i] + l_stack2[i]*(1-mask_stack[i]))

    blend_img = laplacian_blend[-1]
    for i in reversed(range(level-1)):
        blend_img = get_Gaussian_blur(blend_img, 11, 7) + laplacian_blend[i]
    plt.imshow(blend_img, cmap = 'gray')
    plt.show()
    #cv2.imwrite(os.path.join(result_dir, 'blend.jpg'), blend_img)    

#part2_1()
part2_2()
#part2_34()
