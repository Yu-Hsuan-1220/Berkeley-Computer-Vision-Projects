import cv2
import numpy as np
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

img_dir = './hw2/original_image'
result_dir = './hw2/result'