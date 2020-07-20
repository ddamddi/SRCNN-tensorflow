from utils import *
from ops import *
import time

class SRCNN:
    def __init__(self, sess, args):
        self.model_name = "SRCNN"
        self.train_dataset = 'T91'
        self.sess = sess

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.infer_dir = args.infer_dir

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.initial_lr = args.lr
        self.momentum = args.momentum
        self.val_interval = args.val_interval

        self.pixel_max = 1.
        self.pixel_min = 0.
        self.patch_size = 33
        self.stride = 14
        self.scale = args.scale
        self.img_c = 1
        
        self.depth = args.depth
        self.kernel_size = args.kernel_size
        self.filter_size = args.filter_size
        self.filter_size.append(self.img_c)
        
        if args.phase == 'train':
            ''' Load T91 Dataset for Training '''
            self.train_HR, self.train_LR = load_T91(scale=self.scale)

            ''' Load Datasets for Validation (Set5)'''
            self.test_HR, self.test_LR = load_set5(scale=self.scale)
            self.test_HR, self.test_LR = preprocessing((self.test_HR, self.test_LR))

            print("---------\nDatasets\n---------")
            print("TRAIN LABEL : ", str(np.array(self.train_HR).shape))
            print("TRAIN INPUT : ", str(np.array(self.train_LR).shape))
            print("TEST LABEL  : ", str(np.array(self.test_HR).shape))
            print("TEST INPUT  : ", str(np.array(self.test_LR).shape))

    def network(self, x, reuse=False):
        with tf.variable_scope("SRCNN", reuse=reuse):
            for i in range(0, self.depth-1):
                with tf.variable_scope("layer" + str(i+1), reuse=reuse):
                    x = conv(x, self.filter_size[i], kernel=self.kernel_size[i], stride=1, padding='SAME', use_bias=True, scope="conv_0")
                    x = relu(x)
            
            with tf.variable_scope("layer" + str(self.depth), reuse=reuse):
                x = conv(x, self.filter_size[self.depth-1], kernel=self.kernel_size[self.depth-1], stride=1, padding='SAME', use_bias=True, scope="conv_0")

            return x

    def build_model(self):
        """ Graph Input
                High_Resolution(Ground_Truth) : X
                Interpolated_Low_Resolution   : Y  """
        self.train_X = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.img_c], name='train_X')
        self.train_Y = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.img_c], name='train_Y')

        self.test_X  = tf.placeholder(tf.float32, [None, None, None, self.img_c], name='test_X')
        self.test_Y  = tf.placeholder(tf.float32, [None, None, None, self.img_c], name='test_Y')

        self.first_lr  = tf.placeholder(tf.float32, name='first_lr')
        self.second_lr = tf.placeholder(tf.float32, name='second_lr')

        """ Model """
        self.train_logits = self.network(self.train_Y)
        self.test_logits  = tf.clip_by_value(self.network(self.test_Y, reuse=True), self.pixel_min, self.pixel_max)
        # self.test_logits = self.network(self.test_Y, reuse=True)

        self.train_loss = tf.reduce_mean(tf.square(self.train_logits - self.train_X))
        self.test_loss  = tf.reduce_mean(tf.square(self.test_logits - self.test_X))

        self.train_psnr = tf.reduce_mean(tf.image.psnr(self.train_logits, self.train_X, max_val=self.pixel_max-self.pixel_min))
        self.test_psnr = tf.reduce_mean(tf.image.psnr(self.test_logits, self.test_X, max_val=self.pixel_max-self.pixel_min))

        self.train_ssim = tf.reduce_mean(tf.image.ssim(self.train_logits, self.train_X, max_val=self.pixel_max-self.pixel_min))
        self.test_ssim = tf.reduce_mean(tf.image.ssim(self.test_logits, self.test_X, max_val=self.pixel_max-self.pixel_min))

        """ Training """
        self.var_list1 = tf.trainable_variables()[:4]
        self.var_list2 = tf.trainable_variables()[4:]

        self.train_op1 = tf.train.MomentumOptimizer(learning_rate=self.first_lr, momentum=self.momentum).minimize(self.train_loss, var_list=self.var_list1)
        self.train_op2 = tf.train.MomentumOptimizer(learning_rate=self.second_lr, momentum=self.momentum).minimize(self.train_loss, var_list=self.var_list2)
        self.train_op = tf.group(self.train_op1, self.train_op2)

        """ Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_psnr = tf.summary.scalar("train_psnr", self.train_psnr)
        self.summary_train_ssim = tf.summary.scalar("train_ssim", self.train_ssim)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_psnr = tf.summary.scalar("test_psnr", self.test_psnr)
        self.summary_test_ssim = tf.summary.scalar("test_ssim", self.test_ssim)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_psnr, self.summary_train_ssim])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_psnr, self.summary_test_ssim])

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()
        
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        total_batch_iter = len(self.train_HR) // self.batch_size
        
        # Restore checkpoints if exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)        
        if could_load:
            start_epoch = checkpoint_counter
            train_counter = checkpoint_counter * total_batch_iter + 1
            val_counter = checkpoint_counter * (total_batch_iter // self.val_interval + 1) + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            train_counter = 1
            val_counter = 1
            print(" [!] Load failed...")
        
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            for batch_idx in range(total_batch_iter):
                batch_HR = self.train_HR[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size]
                batch_LR = self.train_LR[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size]

                train_feed_dict = {
                    self.train_X : batch_HR,
                    self.train_Y : batch_LR,
                    self.first_lr : self.initial_lr,
                    self.second_lr : self.initial_lr / 10
                }

                _, train_loss, train_psnr, summary_str = self.sess.run(
                    [self.train_op, self.train_loss, self.train_psnr, self.train_summary], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, train_counter)
                train_counter += 1

                if batch_idx % self.val_interval == 0:
                    test_loss, test_psnr = self.validate(val_counter)
                    val_counter += 1
                    print("Epoch: [%2d] [%5d/%5d] train_loss: %.7f, train_psnr: %.7f, test_loss: %.7f, test_psnr: %.7f" % (epoch+1, batch_idx+1, total_batch_iter, train_loss, train_psnr, test_loss, test_psnr))
                else:
                    print("Epoch: [%2d] [%5d/%5d] train_loss: %.7f, train_psnr: %.7f" % (epoch+1, batch_idx+1, total_batch_iter, train_loss, train_psnr))
        
            self.save(self.checkpoint_dir, epoch+1)

        self.save(self.checkpoint_dir, self.epoch)
        print("Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

    def validate(self, val_counter):
        test_feed_dict = {
            self.test_X : self.test_HR,
            self.test_Y : self.test_LR
        }

        summary_str, test_loss, test_psnr = self.sess.run([self.test_summary, self.test_loss, self.test_psnr], feed_dict=test_feed_dict)
        self.writer.add_summary(summary_str, val_counter)

        return test_loss, test_psnr

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        test_datasets = ['Set5', 'Set14', 'BSDS200', 'Urban100']

        ''' TEST DATA LOAD '''
        for dataset in test_datasets:
            if dataset == 'Set5':
                self.test_HR, self.test_LR = load_set5(scale=self.scale)
            if dataset == 'Set14':
                self.test_HR, self.test_LR = load_set14(scale=self.scale)
            if dataset == 'BSDS200':
                self.test_HR, self.test_LR = load_bsds200(scale=self.scale)
            if dataset == 'Urban100':
                self.test_HR, self.test_LR = load_urban100(scale=self.scale)
            
            self.test_HR, self.test_LR = preprocessing((self.test_HR, self.test_LR))

            test_loss_mean = 0.
            test_psnr_mean = 0.
            test_ssim_mean = 0.

            start_time = time.time()
            for idx in range(len(self.test_HR)):
                h, w, c = self.test_HR[idx].shape
                _h, _w, _c = self.test_LR[idx].shape
                h = min(h, _h)
                w = min(w, _w)

                HR = self.test_HR[idx][:h,:w]
                LR = self.test_LR[idx][:h,:w]

                HR = HR.reshape([1,h,w,c])
                LR = LR.reshape([1,h,w,c])

                test_feed_dict = {
                    self.test_X : LR,
                    self.test_Y : HR
                }

                test_output, test_loss, test_psnr, test_ssim = self.sess.run([self.test_logits, self.test_loss, self.test_psnr, self.test_ssim], feed_dict=test_feed_dict)

                test_loss_mean += test_loss
                test_psnr_mean += test_psnr
                test_ssim_mean += test_ssim

            test_loss_mean /= len(self.test_HR)
            test_psnr_mean /= len(self.test_HR)
            test_ssim_mean /= len(self.test_HR)

            print("{} Average - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, test_loss_mean, test_psnr_mean, test_ssim_mean))
            print("     Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

    def infer(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        infer_datasets = ['Set5', 'Set14', 'BSDS200', 'Urban100']
        # infer_datasets = ['Set5']

        test_loss_mean = 0.
        test_psnr_mean = 0.
        test_ssim_mean = 0.
        
        for dataset in infer_datasets:
            if dataset == 'Set5':
                self.test_HR, self.test_LR = load_set5(scale=self.scale, with_cbcr=True)
            if dataset == 'Set14':
                self.test_HR, self.test_LR = load_set14(scale=self.scale, with_cbcr=True)
            if dataset == 'BSDS200':
                self.test_HR, self.test_LR = load_bsds200(scale=self.scale, with_cbcr=True)
            if dataset == 'Urban100':
                self.test_HR, self.test_LR = load_urban100(scale=self.scale, with_cbcr=True)

            start_time = time.time()
            for idx in range(len(self.test_HR)):
                original = ycrcb2bgr(np.concatenate((self.test_HR[idx][:,:,2:],self.test_HR[idx][:,:,:2]), axis=2))
                
                test_HR = normalize(self.test_HR[idx][:,:,2:3]) # Y value of HR
                test_crcb = self.test_LR[idx][:,:,0:2]          # Color value of LR
                test_LR = normalize(self.test_LR[idx][:,:,2:3]) # Y value of LR

                h, w, c = test_HR.shape
                _h, _w, _c = test_LR.shape
                h = min(h, _h)
                w = min(w, _w)

                HR = test_HR[:h,:w]
                LR = test_LR[:h,:w]
                test_crcb = test_crcb[:h,:w]

                HR = HR.reshape([1,h,w,c])
                LR = LR.reshape([1,h,w,c])
                test_crcb = test_crcb.reshape([1,h,w,2])

                test_feed_dict = {
                    self.test_X : LR,
                    self.test_Y : HR
                }

                test_output, test_loss, test_psnr, test_ssim = self.sess.run([self.test_logits, self.test_loss, self.test_psnr, self.test_ssim], feed_dict=test_feed_dict)

                test_output = denormalize(test_output)
                inference_img = np.concatenate((test_output, test_crcb), axis=3)
                inference_img = inference_img.reshape([h,w,3])
                inference_img = ycrcb2bgr(inference_img)

                if not os.path.exists(os.path.join(self.infer_dir, dataset, str(self.scale)+'x')):
                    os.makedirs(os.path.join(self.infer_dir, dataset, str(self.scale)+'x'))

                infer_path = os.path.join(self.infer_dir, dataset, str(self.scale)+'x')
                cv2.imwrite(os.path.join(infer_path, 'SRCNN_' + str(self.scale) + 'x_' + str(idx) + '_' + str(test_psnr) + '_' + str(test_ssim) + '.png'), inference_img)
                print("{}[{}] - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, idx, test_loss, test_psnr, test_ssim))
                
                if len(self.test_HR) <= 20:
                    cv2.imshow('original', original)
                    cv2.imshow('inference - {}[{}]'.format(dataset, idx), inference_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                test_loss_mean += test_loss
                test_psnr_mean += test_psnr
                test_ssim_mean += test_ssim

            test_loss_mean /= len(self.test_HR)
            test_psnr_mean /= len(self.test_HR)
            test_ssim_mean /= len(self.test_HR)

            print("{} Average - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, test_loss_mean, test_psnr_mean, test_ssim_mean))
            print("     Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

    @property
    def model_dir(self):
        return "{}_{}x_{}_{}_{}".format(self.model_name, self.scale, self.train_dataset, self.batch_size, self.initial_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print(" [*] Model Saving...")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)
        print(" [*] Save complete")
    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
