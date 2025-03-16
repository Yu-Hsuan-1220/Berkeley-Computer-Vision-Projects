# Berkeley-Intro-to-Computer-Vision-and-Computational-Photography

Course Code: CS194-26

## Project1

Image alignment

Given spilt R、G、B channel gray-scale image, use NCC(Normalized Cross-Correlation) to align the images and covert to a RGB image

#### Input

![Alt text](hw1/original_Image/00351v.jpg)

#### Result:

![Alt text](hw1/result/00351v.jpg)

## Project2
### Part 1

First, apply Gaussian blur to reduce noise, then use convolution to approximate the partial derivatives in the x and y directions, and finally set a threshold for edge detection.

#### Input
![Alt text](hw2/original_image/cameraman.png)

#### Result
![Alt text](hw2/result/Gaussian_cameraman.png)
