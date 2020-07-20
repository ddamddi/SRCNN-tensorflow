import numpy as np
import tensorflow as tf
import random
import h5py
import cv2
import os

def time_calculate(sec):
    s = sec % 60
    m = sec // 60
    h = m // 60
    m = m % 60
    return h, m, s

def str2int_list(str):
    num_list = str.split('-')
    for idx, elem in enumerate(num_list):
        num_list[idx] = int(elem)
    return num_list

def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

''' For Load Train and Test Datasets '''
def load_T91(scale=2):
    filename = 'T91_' + str(scale) + 'x.h5'
    file_dir = os.path.join(os.getcwd(), 'data', 'Train', filename)
    
    with h5py.File(file_dir, 'r') as f:
        label = list(f['label'])
        data = list(f['data'])
    
    label, data = shuffle(label), shuffle(data)
    return label, data

def load_set5(scale=2, with_cbcr=False):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set5', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set5', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            label.append(read_img)
        else:
            label.append(read_img[:,:,2:3])
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            data.append(read_img)
        else:
            data.append(read_img[:,:,2:3])

    return label, data

def load_set14(scale=2, with_cbcr=False):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set14', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set14', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            label.append(read_img)
        else:
            label.append(read_img[:,:,2:3])
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            data.append(read_img)
        else:
            data.append(read_img[:,:,2:3])

    return label, data

def load_bsds200(scale=2, with_cbcr=False):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'BSDS200', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'BSDS200', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            label.append(read_img)
        else:
            label.append(read_img[:,:,2:3])
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            data.append(read_img)
        else:
            data.append(read_img[:,:,2:3])

    return label, data

def load_urban100(scale=2, with_cbcr=False):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Urban100', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Urban100', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            label.append(read_img)
        else:
            label.append(read_img[:,:,2:3])
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if with_cbcr:
            data.append(read_img)
        else:
            data.append(read_img[:,:,2:3])

    return label, data

''' For Dataset Pre-processing'''
def preprocessing(x, patch_size=33, stride=14):
    HR, LR = prepare_patches(x, patch_size=patch_size, stride=stride)
    
    HR = normalize(HR)
    LR = normalize(LR)
    
    HR = shuffle(HR)
    LR = shuffle(LR)
    return HR, LR

def prepare_patches(x, patch_size=33, stride=14):
    HR, LR = x
    HR_patches = []
    LR_patches = []

    for idx in range(len(HR)):
        HR_img = HR[idx]
        LR_img = LR[idx]
        row = HR_img.shape[0]
        col = HR_img.shape[1]
        row_cnt = row//patch_size
        col_cnt = col//patch_size

        for i in range(row_cnt):
            for j in range(col_cnt):
                HR_img_crop = HR_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                LR_img_crop = LR_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                HR_patches.append(HR_img_crop)
                LR_patches.append(LR_img_crop)

    return HR_patches, LR_patches

def shuffle(x):
    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    return x

def normalize(x):
    return np.array(x) / 255.

def denormalize(x):
    x *= 255
    x = x.astype('uint8')
    x = np.clip(x, 0, 255)
    return np.array(x)

''' Invert Color Channel '''
def bgr2ycrcb(img):
    cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
    return cvt_img

def ycrcb2bgr(img):
    cvt_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return cvt_img


if __name__ == '__main__' :
    label, data = load_T91(2)
    print(np.array(label).shape)
    print(np.array(data).shape)