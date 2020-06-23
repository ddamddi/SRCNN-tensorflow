from network import SRCNN
from utils import *
import argparse


def check_phase(args):
    args.phase = args.phase.lower()
    assert args.phase == 'train' or args.phase == 'test', 'Choose TRAIN or TEST phase'


def check_channel(args):
    args.channel = args.channel.lower()
    assert args.channel == 'y' or args.channel == 'rgb', 'Color Channel Should be Y(in YCbCr) or RGB'


def check_network_arch(args):
    args.kernel_size = str2int_list(args.kernel_size)
    args.filter_size = str2int_list(args.filter_size)
    assert args.depth == len(args.kernel_size), 'Depth and the number of kernel_size should be match'
    assert len(args.kernel_size) == len(args.filter_size) + 1, 'The number of filter_num (except last Conv) must be 1 less than the number of kernel_size'

    return args


def check_args(args):
    check_phase(args)
    check_channel(args)
    args = check_network_arch(args)
    return args


def parse_args():
    desc = "Tensorflow implementation of SRCNN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test?')

    parser.add_argument('--channel', type=str, default='Y', help='RGB or Y(in YCbCr)?')
    parser.add_argument('--depth', type=int, default=3, help='The number of Conv')
    parser.add_argument('--kernel_size', type=str, default='9-1-5', help='The kernel sizes of Conv')
    parser.add_argument('--filter_size', type=str, default='32-64', help='The filter sizes of Conv (except last Conv)')

    parser.add_argument('--epoch', type=int, default=15000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch per gpu')
    parser.add_argument('--scale', type=int, default=4, help='The Scale factor')

    parser.add_argument('--val_interval', type=int, default=200, help='Validation Batch Interval')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    print(args)
    # quit()
    with tf.Session() as sess:
    
        srcnn = SRCNN(sess, args)
        srcnn.build_model()
        show_all_variables()

        if(args.phase == 'train'):
            srcnn.train()

        srcnn.test()