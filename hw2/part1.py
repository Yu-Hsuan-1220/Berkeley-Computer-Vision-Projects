import cv2
import numpy as np
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

img_dir = './hw2/original_image'
result_dir = './hw2/result'

def pd_x(image):
    D_x = [[-1, 1]]
    return convolve2d(image, D_x, mode='same', boundary='symm') ## I(x,y) = I(x+1, y) - I(x, y)

def pd_y(image):
    D_y = [[-1],[1]]
    return convolve2d(image, D_y, mode='same', boundary='symm')


def convert(Ix, Iy, threshold = None):
    
    
    if threshold == None:
        return np.zeros_like(Ix)
    gradient_magnitude = Ix**2 + Iy**2
    result = np.where(gradient_magnitude >= threshold, 255, 0).astype(np.uint8)
    return result
    
def part1_1():
    img = cv2.imread(os.path.join(img_dir, 'cameraman.png'), cv2.IMREAD_GRAYSCALE)
    #print(img.shape)

    #### Part 1-1 Finite Difference Operator

    threshold = 3000

    Ix = pd_x(img)
    Iy = pd_y(img)

    result = convert(Ix, Iy, threshold)
    plt.imshow(result, cmap='gray')
    plt.show()
    print(result.shape)
    cv2.imwrite(os.path.join(result_dir, 'edge_cameraman.png'), result)

#### Part 2-2
def part1_2():
    img = cv2.imread(os.path.join(img_dir, 'cameraman.png'), cv2.IMREAD_GRAYSCALE)
    k_size = 5
    sigma = 1
    g1d = cv2.getGaussianKernel(k_size, sigma)
    G = np.outer(g1d, g1d.T)
    blur_img = convolve2d(img, G, mode='same', boundary='symm')
    plt.imshow(blur_img, cmap='gray')
    plt.show()
    threshold = 2000

    Ix = pd_x(blur_img)
    Iy = pd_y(blur_img)

    result = convert(Ix, Iy, threshold)
    
    cv2.imwrite(os.path.join(result_dir, 'Gaussian_cameraman.png'), result)

def part1_3():
    img = cv2.imread(os.path.join(img_dir, 'cameraman.png'), cv2.IMREAD_GRAYSCALE)
    k_size = 5
    sigma = 1
    g1d = cv2.getGaussianKernel(k_size, sigma)
    G = np.outer(g1d, g1d.T)
    DoG_x = convolve2d(G, [[1, -1]], mode='same', boundary='symm')
    DoG_y = convolve2d(G, [[1], [-1]], mode='same', boundary='symm')

    Ix = convolve2d(img, DoG_x, mode='same', boundary='symm')
    Iy = convolve2d(img, DoG_y, mode='same', boundary='symm')
    
    threshold = 2000

    result = convert(Ix, Iy, threshold)
    plt.imshow(result, cmap='gray')
    plt.show()
    cv2.imwrite(os.path.join(result_dir, 'DoG_cameraman.png'), result)

part1_2()
