import os
import numpy as np
import cv2
from PIL import Image


import torch
import torch.nn as nn
from torchvision import transforms


def morphology(im):
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_rescaled = cv2.resize(im_gray,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
    kernel = np.ones((23,23),np.uint8)
    close_re = cv2.morphologyEx(im_rescaled, cv2.MORPH_CLOSE, kernel)
    ret,th_otsu_re = cv2.threshold(close_re,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    processed = cv2.resize(th_otsu_re,None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_CUBIC)

    return processed

def find_contours(im):

    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    contours = sorted(contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)[:5]
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    maxArea = 100.0
    good_cnts = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            good_cnts.append(contours[i])
    return good_cnts

def get_mini_boxes(contour):
    rect = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    width, height = rect[1]
    angle = rect[2]
    if abs(angle)==0.0 or abs(angle)==90.0:
        angle = 0.0
    else:
        if int(height)>=int(width):
            if angle < -48:
                angle = (-90 - angle)
            else:
                angle = angle
        else:
            if angle < -48:
                angle = (90+angle)
            else:
                angle = -angle

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    if cv2.arcLength(contour, True)>530:
        point_common_1 = [(points[index_2][0]+points[index_1][0])//2, (points[index_2][1]+points[index_1][1])//2]
        point_common_2 = [(points[index_3][0]+points[index_4][0])//2, (points[index_3][1]+points[index_4][1])//2]
        box1 = [points[index_1], point_common_1, point_common_2,points[index_4]]
        box2 = [point_common_1,points[index_2],points[index_3],point_common_2]
        angle = 0.0
        box=[box1, box2] 
    else:
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        box = [box]
    return box, angle

def crop_rect(img, rect):
    
    # Rotation
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # print(height,width)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img_rot

def padding(im,exp_size=(100,100)):
    exp_height, exp_width = exp_size
    height, width = im.shape[0], im.shape[1]
    if height>exp_height or width>exp_width:
        im = cv2.resize(im, (90, 90), interpolation = cv2.INTER_AREA)
        height, width = im.shape[0], im.shape[1]
    del_ht = exp_height-height
    del_wd = exp_width-width

    pad_top = del_ht//2
    pad_bottom = del_ht - pad_top

    pad_left = del_wd//2
    pad_right = del_wd - pad_left

    img_padded = cv2.copyMakeBorder(im, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    # print(img_padded.shape)
    return img_padded

def extract_letters(im):
    # im = cv2.imread(file,-1)
    initial_pad_hor = np.ones((100,20),np.uint8)*255
    initial_pad_ver = np.ones((20,440),np.uint8)*255
    im = cv2.hconcat([initial_pad_hor,im,initial_pad_hor])
    im = cv2.vconcat([initial_pad_ver,im,initial_pad_ver])
    ret,thresh_inv = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
    image_copy = thresh_inv.copy()
    good_cnts = find_contours(image_copy)
    letters = []
    i=0
    for cnt in good_cnts:
        boxes, angle = get_mini_boxes(cnt)
        for j in range(len(boxes)):    
            box = np.int0(boxes[j])

            origin_x = min(box[0][0],box[1][0],box[2][0],box[3][0])
            origin_y = min(box[0][1],box[1][1],box[2][1],box[3][1])
            # print(origin_x,origin_y)
            max_x = max(box[0][0],box[1][0],box[2][0],box[3][0])
            max_y = max(box[0][1],box[1][1],box[2][1],box[3][1])
            # print(max_x,max_y)

            width = max_x - origin_x
            height = max_y - origin_y
            centre = (width/2, height/2)
            rect = (centre,(height, width),(angle))
            img = thresh_inv[origin_y:max_y, origin_x:max_x]
            img_rot = crop_rect(img,rect)
            img_padded = padding(img_rot)
            letters.append(img_padded)

    return letters

def predict_letter(model, letter):
    image_transforms = {
        "train": transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((100, 100)),
            transforms.ToTensor()
            ])
        }
    idx2class = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 'A', 9: 'C', 10: 'D',
                 11: 'E', 12: 'F', 13: 'G', 14: 'H', 15: 'J', 16: 'K', 17: 'M', 18: 'N', 19: 'P', 20: 'Q',
                 21: 'R', 22: 'S', 23: 'T', 24: 'U', 25: 'V', 26: 'W', 27: 'X', 28: 'Y', 29: 'Z'}
    img = Image.fromarray(letter)
    img_t = image_transforms["train"](img)
    batch_t = torch.unsqueeze(img_t, 0)

    out = model(batch_t)
    prediction = int(torch.max(out.data, 1)[1].numpy())
    predicted_letter = idx2class[prediction]
    return predicted_letter

