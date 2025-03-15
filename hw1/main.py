import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import floor
import cv2
import os
def ncc(arr1, arr2):
    avg1 = np.average(arr1)
    avg2 = np.average(arr2)
    norm_both = (arr1-avg1) * (arr2-avg2)
    norm1 = (arr1-avg1) ** 2
    norm2 = (arr2-avg2) ** 2
    return np.sum(norm_both)/(((np.sum(norm1))**(1/2))*((np.sum(norm2))**(1/2))) 

# def shift_image(img, dx, dy):
#     h, w = img.shape[:2] 
#     shifted = np.zeros_like(img)
  
#     x_start, x_end = max(0, dx), min(w, w + dx)
#     y_start, y_end = max(0, dy), min(h, h + dy)

#     src_x_start, src_x_end = max(0, -dx), min(w, w - dx)
#     src_y_start, src_y_end = max(0, -dy), min(h, h - dy)

#     shifted[y_start:y_end, x_start:x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]

#     return shifted
def shift_image(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def crop(img, percentage):
    h, w = img.shape
    start_x, end_x = floor(h*percentage), floor(h*(1-percentage))
    start_y, end_y = floor(w*percentage), floor(w*(1-percentage))

    return img[start_x:end_x+1, start_y:end_y+1]


Image_path = './hw1/original_Image/01112v.jpg'
result_dir = './hw1/result'
img = Image.open(Image_path).convert('L')


img_array = np.array(img, dtype=np.uint8)

h, w = img_array.shape
h_cropped = (h // 3) * 3
img_array = img_array[:h_cropped, :]
h, w = img_array.shape

print(f"height {h}, weight {w}")

B_array, G_array, R_array = np.array_split(img_array, 3, axis=0)

crop_percentage = 0.1

B_array = crop(B_array, crop_percentage)
G_array = crop(G_array, crop_percentage)
R_array = crop(R_array, crop_percentage)

### naive
RGB_array = np.zeros((3, B_array.shape[0], B_array.shape[1]), dtype=np.uint8)
RGB_array[0] = R_array
RGB_array[1] = G_array
RGB_array[2] = B_array


### apply NCC
scope = 20

ncc_RGB_array = np.zeros_like(RGB_array)
# print(ncc_RGB_array.shape)
ncc_RGB_array[0] = R_array ## fixed R

#align R & G array
value, best_x, best_y = -1, 0, 0

for i in range(-scope, scope+1):
    for j in range(-scope, scope+1):
        shift_G = shift_image(G_array, i, j)
        cur_val = ncc(R_array, shift_G)
        if cur_val > value:
            best_x = i
            best_y = j
            value = cur_val

ncc_RGB_array[1] = shift_image(G_array, best_x, best_y)
print(value, best_x, best_y)

#align R & B array
value, best_x, best_y = -1, 0, 0

for i in range(-scope, scope+1):
    for j in range(-scope, scope+1):
        shift_B = shift_image(B_array, i, j)
        cur_val = ncc(R_array, shift_B)
        if cur_val > value:
            best_x = i
            best_y = j
            value = cur_val
ncc_RGB_array[2] = shift_image(B_array, best_x, best_y)
print(value, best_x, best_y)


plt.imshow(np.transpose(ncc_RGB_array, (1, 2, 0)))
plt.show()

ncc_RGB_array = np.transpose(ncc_RGB_array, (1, 2, 0))
ncc_BGR_array = cv2.cvtColor(ncc_RGB_array, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(result_dir, '01112v.jpg'), ncc_BGR_array)