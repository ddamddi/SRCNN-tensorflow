from network import SRCNN
import argparse
import tensorflow as tf
import numpy as np
from utils import *

def parse_args():
    desc = "Tensorflow implementation of SRCNN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='mnist', help='[cifar10, cifar100, mnist]')
    
    parser.add_argument('--filter_size', type=str, default='9,1,5', help='The filter sizes in CNNs')
    parser.add_argument('--filter_num', type=str, default='32,64', help='The filter numbers in CNNs')

    parser.add_argument('--epoch', type=int, default=82, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch per gpu')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    # return check_args(parser.parse_args())
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    sess = tf.Session()
    
    srcnn = SRCNN(sess, args)
    srcnn.build_model()
    show_all_variables()

    print(tf.get_variable('layer1'))
    
    # if(args.phase == 'train'):
    #     srcnn.train()

    
    # srcnn.test()