from utils import *
from ops import *

class SRCNN:
    def __init__(self, sess, args):
        self.sess = sess
        self.filter_size = str2int(str2list(args.filter_size))
        self.filter_num = str2int(str2list(args.filter_num))
        self.lr = args.lr
        self.momentum = 0.9
        self.batch_size = args.batch_size
        self.img_size = 32
        self.c_dim = 3

    def network(self, x, reuse=False):
        with tf.variable_scope("SRCNN", reuse=reuse):         
            # F_1(Y) : Patch extraction and representation
            with tf.variable_scope("layer1", reuse=reuse):
                x = conv(x, self.filter_num[0], kernel=self.filter_size[0], stride=1, padding='SAME', use_bias=True, scope="conv_0")
                x = relu(x)

            # F_2(Y) : Non-linear mapping
            with tf.variable_scope("layer2", reuse=reuse):
                x = conv(x, self.filter_num[1], kernel=self.filter_size[1], stride=1, padding='SAME', use_bias=True, scope="conv_0")
                x = relu(x)

            # F_3(Y) : Reconstruction
            with tf.variable_scope("layer3", reuse=reuse):
                x = conv(x, 3, kernel=self.filter_size[2], stride=1, padding='SAME', use_bias=True, scope="conv_0")

            return x

    def build_model(self):
        """ Graph Input """
        # ground_truth : X
        self.train_ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_ground_truth')
        self.test_ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='test_ground_truth')
        # low_resolution : Y
        self.train_low_resolution = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_low_resolution')
        self.test_low_resolution = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='test_low_resolution')


        """ Model """
        self.train_logits = self.network(self.train_low_resolution)
        self.test_logits = self.network(self.test_low_resolution, reuse=True)

        self.train_loss = tf.reduce_mean(tf.square(self.train_logits-self.train_ground_truth))
        self.test_loss = tf.reduce_mean(tf.square(self.test_logits-self.test_ground_truth))

        """ Training """
        self.var_list1 = []
        self.var_list2 = []

        self.train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum).minimize(self.train_loss)
        # self.train_op1 = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum).minimize(self.train_loss, var_list=self.var_list1)
        # self.train_op2 = tf.train.MomentumOptimizer(learning_rate=self.lr*0.1, momentum=self.momentum).minimize(self.train_loss, var_list=self.var_list2)
        # self.train_op = tf.group(train_op1, train_op2)

        """ Summary """
        # self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        # self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        # self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        # self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        # self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        # self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    # def train(self):
    #     # initialize all variables
    #     tf.global_variables_initializer().run()

    #     # saver to save model
    #     self.saver = tf.train.Saver()

    #     # summary writer
    #     self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

    # def test(self):



if __name__ == '__main__':
    sess = tf.Session()
    
    cnn = SRCNN()

