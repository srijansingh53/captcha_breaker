import numpy as np
import cv2
from PIL import Image
import argparse

import torch

from inference import morphology, extract_letters, predict_letter
from model_arch import train1, train2, train4, train6, train7, train8, train9, CustomNet

def predict(arch_class,model_path,image_path):
    arch_class_object = arch_class()
    model = arch_class_object.getmodel()
    # print(model)
    full_model_path = model_path
    model = torch.load(full_model_path, map_location=torch.device('cpu'))
    model.eval()

    im = cv2.imread(image_path,-1)
    processed = morphology(im)
    letters = extract_letters(processed)

    captcha = ''
    for letter in letters:
        predicted_letter = predict_letter(model, letter)
        captcha = captcha+predicted_letter
        # print(predicted_letter)
    return captcha

if __name__=='__main__':

    model_archs = {'train1': train1, 'train2': train2, 'train3': train2, 'train4': train4, 'train5': train4,
    'train6': train6, 'train7': train7, 'train8': train8, 'train9':train9, 'train10': train4}
    parser = argparse.ArgumentParser()

    parser.add_argument('architecture', help='model architecture', choices = model_archs.keys())
    parser.add_argument('model_path', help='complete path for the model')
    parser.add_argument('image_path', help='complete path of image')

    args = parser.parse_args()
    arch_class = model_archs[args.architecture]
    captcha = predict(arch_class, args.model_path, args.image_path)
    print('The predicted captcha is:',captcha)
