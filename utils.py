import numpy as np
from keras.datasets import mnist, cifar10, cifar100
import cv2
import tensorflow as tf


def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_cifar100() :
    (train_data, train_labels), (test_data, test_labels) = cifar100.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0
    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 100)
    test_labels = to_categorical(test_labels, 100)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels


def mse(X, Y):
    return np.mean(np.square(X-Y))

def psnr(X, Y, peak=255):
    return 10 * np.log10(np.square(peak) / MSE(X,Y))

# def SSIM():


# https://sistenix.com/rgb2ycbcr.html
def RGB2YCbCr(x):
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]

    Y  = 16 + 65.738*R/256 + 129.057*G/256 + 25.064*B/256
    Cb = 128 - 37.945*R/256 - 74.494*G/256 + 112.439*B/256
    Cr = 128 + 112.439*R/256 - 94.154*G/256 - 18.285*B/256

    return (Y, Cb, Cr)

# def YCbCr2RGB(x):


def str2list(str):
    x = str.split(',')
    for i, elem in enumerate(x):
        x[i] = elem.strip()
 
    return x

def str2int(str_list):
    num_list = []
    for elem in str_list:
        num_list.append(int(elem))
    
    return num_list

def str2float(str_list):
    num_list = []
    for elem in str_list:
        num_list.append(float(elem))
    
    return num_list

if __name__ == '__main__':
    img = cv2.imread('./data/sample_img/cropped-dog.jpg')
    cv2.imshow(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
