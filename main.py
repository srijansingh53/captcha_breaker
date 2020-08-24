import numpy as np
import cv2
from PIL import Image
import argparse

import torch

from inference import CustomNet, morphology, extract_letters, predict_letter

def predict(model_path,image_path):
    model = CustomNet()
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

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='complete path for the model')
    parser.add_argument('image_path', help='complete path of image')

    args = parser.parse_args()

    captcha = predict(args.model_path, args.image_path)
    print('The predicted captcha is:',captcha)
