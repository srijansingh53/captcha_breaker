import numpy as np
import cv2
from PIL import Image
import argparse
import glob
import shutil

import torch

from inference import morphology, extract_letters
from active_training import train

def save_letters(image_path, labels, k):

    im = cv2.imread(image_path,-1)
    processed = morphology(im)
    letters = extract_letters(processed)

    i=0
    for letter in letters:
       cv2.imwrite('active letter2/'+labels[i]+'/'+str(k)+'.jpg', letter)
       i+=1
       k+=1
    return k

def count(folder):
    return len(glob.glob(folder+'/*.jpg'))

# extract letters and save
def prepare_data():
    filenames = glob.glob('active folder/*.jpg')
    k=0
    for file in filenames:
        #print(file)
        labels = file.split('_')[1]
        labels = list(labels)[:5]
        k = save_letters(file, labels, k)

    # making datset and balancing it
    active_folders = sorted(glob.glob('active letter/*'))

    letters_count = {}
    for folder in active_folders:
        letters_count[folder] = count(folder)

    max_count = max(letters_count.values())
    for folder in active_folders:
        #print(folder)
        diff = max_count - letters_count[folder]
        #print(diff)
        original_folder = sorted(glob.glob('letters/'+folder.split('/')[1]+'/*.jpg'), reverse=True)
        copy_files = original_folder[:diff]
        #print(len(copy_files))
        for files in copy_files:
            # print('f')
            shutil.copy(files,folder)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--prepare_data', help='make letters folder for training')
    parser.add_argument('--train_active', help='train with prepared letter folder')


    args = parser.parse_args()
    if args.prepare_data:
        prepare_data()
    if args.train_active:
        train()
        
