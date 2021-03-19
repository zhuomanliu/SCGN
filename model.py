import os
import time
import cv2
import numpy as np
from math import ceil, isnan
import tensorflow as tf
from networks import SCGN
from loss import Loss, Inception_Score
from dataset import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(object):
    def __init__(self, args):
        self.mode = args.mode
        self.params = {'input_size': args.input_size, 'batch_size': args.batch_size, 'epochs': args.epochs,
                       'gen_lr': args.gen_lr, 'disc_lr': args.disc_lr,
                       'gen_decay_ep': ceil(args.epochs/2), 'disc_decay_ep': ceil(args.epochs/2),
                       'w_vc': args.w_vc, 'w_adv': args.w_adv, 'w_sharp': args.w_sharp,
                       'incep_batch': 64, 'dataset': args.dataset,
                       'sharp_loss': args.sharp_loss}
        self.demo_params = {'data_path': args.data_path,
                            'input_l_name': args.input_l_name, 'input_r_name': args.input_r_name,
                            'output_name': args.output_name}

        if self.mode == 'train':
            self.params['save_folder'] = os.path.join('./ckpts', args.save_folder)
            self.params['epoch_save'] = args.epoch_save
            if not os.path.exists(self.params['save_folder']):
                os.mkdir(self.params['save_folder'])
            self.params['model_path'] = os.path.join('./ckpts', args.save_folder)
            self.params['output_folder'] = os.path.join('./results', '%s-train' % args.save_folder)
            if not os.path.exists(self.params['output_folder']):
                os.mkdir(self.params['output_folder'])
        elif self.mode == 'test' or self.mode == 'demo':
            self.params['model_path'] = os.path.join('./ckpts', args.model)
            self.params['output_folder'] = os.path.join('./results', args.output_folder)
            if not os.path.exists(self.params['output_folder']) and self.mode == 'test':
                os.mkdir(self.params['output_folder'])
        else:
            raise IOError("Illegal input, please input a text among of [train, test, demo].")

        self.build_model()
        if self.mode != 'demo':
            self.data = Dataset(self.params, self.mode)
        if self.mode == 'train':
            self.data_val = Dataset(self.params, 'test')

    def build_model(self):
        input_size = self.params['input_size']
        g_init_lr, g_dc_step = self.params['gen_lr'], self.params['gen_decay_ep']
        d_init_lr, d_dc_step = self.params['disc_lr'], self.params['disc_decay_ep']

        self.inputs_l = tf.placeholder(tf.float32, (None, input_size, input_size, 3), name="inputs_l")
        self.inputs_r = tf.placeholder(tf.float32, (None, input_size, input_size, 3), name="inputs_r")
        self.targets = tf.placeholder(tf.float32, (None, input_size, input_size, 3), name="targets")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.mode == 'test':
            self.inception_score = Inception_Score(self.params['incep_batch'])

        tf.set_random_seed(1)

        self.g_lr = tf.train.exponential_decay(g_init_lr, global_step=self.global_step,
                                               decay_steps=g_dc_step, decay_rate=0.1, staircase=True)
        self.d_lr = tf.train.exponential_decay(d_init_lr, global_step=self.global_step,
                                               decay_steps=d_dc_step, decay_rate=0.1, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=0.9, beta2=0.999)
            g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=0.9, beta2=0.999)

        with tf.variable_scope(tf.get_variable_scope()):
            pred_tanh, real_logits, fake_logits, inv_l, inv_r = \
                SCGN(self.inputs_l, self.inputs_r, self.targets)
            self.pred_tanh, self.inv_l, self.inv_r = pred_tanh, inv_l, inv_r

            d_loss, g_loss, l1_loss, inv_loss, adv_loss, sharp_loss, metrics, pred_gau, iter_metrics = \
                Loss([self.params['w_adv'], self.params['w_vc'], self.params['w_sharp']],
                      self.targets, pred_tanh, real_logits, fake_logits,
                      self.inputs_l, self.inputs_r, self.inv_l, self.inv_r,
                      mode=self.mode, choice=self.params['sharp_loss'])

            self.d_loss, self.g_loss = d_loss, g_loss
            self.l1_loss, self.inv_loss, self.adv_loss, self.sharp_loss = l1_loss, inv_loss, adv_loss, sharp_loss
            self.metrics = metrics
            self.pred_gau = pred_gau
            self.iter_metrics = iter_metrics

            self.train_d = d_opt.minimize(d_loss)
            self.train_g = g_opt.minimize(g_loss)

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
        elif self.mode == 'demo':
            self.demo()
        else:
            raise IOError("Illegal mode, please input a text among of [train, test, demo].")

    def train(self):
        print("---------------------------\nStart Training ...")
        saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            psnr_msssim_train_f = open(os.path.join(self.params['save_folder'], 'psnr_msssim_train.txt'), 'w')
            psnr_msssim_f = open(os.path.join(self.params['save_folder'], 'psnr_msssim.txt'), 'w')

            epochs = self.params['epochs']
            for ep in range(epochs):
                for iter in range(0, self.data.image_num, self.params['batch_size']):
                    batch = self.data.load_batch(iter)

                    _, disc_loss, dlr, glr = sess.run([self.train_d, self.d_loss, self.d_lr, self.g_lr],
                                                          feed_dict={self.inputs_l: batch[0],
                                                                     self.inputs_r: batch[1],
                                                                     self.targets: batch[2]})

                    _, invl, invr, gen_loss, l1, inv, adv, sharp, pred_res, pred_gau, metrics = \
                        sess.run([self.train_g, self.inv_l, self.inv_r,
                                  self.g_loss, self.l1_loss, self.inv_loss, self.adv_loss, self.sharp_loss,
                                  self.pred_tanh, self.pred_gau, self.iter_metrics],
                                  feed_dict={self.inputs_l: batch[0],
                                             self.inputs_r: batch[1],
                                             self.targets: batch[2]})

                    print("\r[%03d/%05d]" % (ep + 1, iter + 1),
                          "G: %.5f" % gen_loss, "D: %.5f" % disc_loss,
                          "L1: %.5f" % l1, "VC: %.5f" % inv, "Adv: %.5f" % adv, "S: %.5f" % sharp, end="")

                    psnr_msssim_train_f.write('%f,%f\n' % (metrics[0], metrics[1]))

                    if isnan(gen_loss):
                        cv2.imwrite(os.path.join(self.params['output_folder'], "pred_res_gau_ep%03d_iter%04d.png" %
                                                 (ep + 1, iter + 1)),
                                    np.array(np.hstack((pred_res.squeeze(), pred_gau.squeeze())),
                                             dtype=np.float64) * 127.5 + 127.5)
                        exit(-1)

                print("\n")
                if (ep + 1) % self.params['epoch_save'] == 0 or (ep + 1) == epochs:
                    # eval
                    eval_metrics = self.test('eval_iter', sess)
                    psnr_msssim_f.write('%f,%f\n' % (eval_metrics[0], eval_metrics[1]))

                    savemodel_flag = False
                    if (ep + 1) == epochs:
                        savemodel_flag = True

                    if savemodel_flag:
                        print("Saving checkpoint ...\n")
                        filename = 'final' if (ep + 1) == epochs else 'ep%03d' % (ep + 1)
                        saver.save(sess, os.path.join(self.params['save_folder'], filename), global_step=ep,
                                   write_meta_graph=False)

    def test(self, mode='test', sess=None):
        if mode == 'test':
            print("---------------------------\nStart Testing ...")
            saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                if mode == 'test':
                    print("Loading model ...")
                saver.restore(sess, tf.train.latest_checkpoint(self.params['model_path']))

                tot_time = 0.
                tot_metrics = np.asarray([0., 0., 0., 0.])
                results = []

                for k in range(0, self.data.image_num):
                    batch = self.data.load_batch(k)
                    per_st_time = time.time()
                    pred_res, inv_l, inv_r, metrics = sess.run(["generator/decoder/de_tanh:0",
                                                                "generator_vcs/decoder/de_l_tanh:0",
                                                                "generator_vcs/decoder_1/de_r_tanh:0",
                                                                self.metrics],
                                                               feed_dict={self.inputs_l: batch[0],
                                                                          self.inputs_r: batch[1],
                                                                          self.targets: batch[2]})
                    per_time = time.time() - per_st_time
                    print("[Image %d costs: %.5f s]" % (k, per_time), end='\r', flush=True)
                    if k != 0:
                        tot_time += per_time

                    tot_metrics += np.asarray(metrics)

                    pred_res = pred_res.squeeze()
                    results.append(pred_res)
                    cv2.imwrite(os.path.join(self.params['output_folder'], "%05d.png" % k),
                                np.array(pred_res, dtype=np.float64) * 127.5 + 127.5)
                    # cv2.imwrite(os.path.join(self.params['output_folder'], "%05d_inv_l.png" % k),
                    #             np.array(inv_l.squeeze(), dtype=np.float64) * 127.5 + 127.5)
                    # cv2.imwrite(os.path.join(self.params['output_folder'], "%05d_inv_r.png" % k),
                    #             np.array(inv_r.squeeze(), dtype=np.float64) * 127.5 + 127.5)

                psnr, msssim, mse, l1 = tot_metrics / float(self.data.image_num)
                incep_mean, incep_std = self.inception_score.run(
                    np.array(results, dtype=np.float32) * 127.5 + 127.5)
                print("PSNR: %.2f | MS-SSIM: %.4f | IS: %.2f (%.2f) | MSE: %.2f | L1: %.3f" %
                      (psnr, msssim, incep_mean, incep_std, mse, l1))
                print('Speed: %.2f FPS' % ((self.data.image_num - 1) / tot_time))
                print("Finish Testing ...")

        elif mode == 'eval_iter':  # eval
            tot_metrics = np.asarray([0. for _ in range(2)])
            for iter in range(0, self.data_val.image_num, self.params['batch_size']):
                batch = self.data.load_batch(iter)

                metrics = sess.run(self.iter_metrics, feed_dict={self.inputs_l: batch[0],
                                                                 self.inputs_r: batch[1],
                                                                 self.targets: batch[2]})

                tot_metrics += np.asarray(metrics)

            num_iters = ceil(self.data_val.image_num / self.params['batch_size'])

            return tot_metrics / num_iters

        else:  # eval
            tot_disc_loss, tot_gen_loss, tot_l1, tot_inv, tot_adv, tot_sharp = [0. for _ in range(6)]
            for iter in range(0, self.data_val.image_num, self.params['batch_size']):
                batch = self.data.load_batch(iter)

                disc_loss = sess.run(self.d_loss, feed_dict={self.inputs_l: batch[0],
                                                             self.inputs_r: batch[1],
                                                             self.targets: batch[2]})

                gen_loss, l1, inv, adv, sharp, metrics = \
                    sess.run([self.g_loss, self.l1_loss, self.inv_loss, self.adv_loss, self.sharp_loss],
                             feed_dict={self.inputs_l: batch[0],
                                        self.inputs_r: batch[1],
                                        self.targets: batch[2]})

                tot_disc_loss += disc_loss
                tot_gen_loss += gen_loss
                tot_l1 += l1
                tot_inv += inv
                tot_adv += adv
                tot_sharp += sharp

            num_iters = ceil(self.data_val.image_num / self.params['batch_size'])
            print("\r--> Val:",
                  "G: %.5f" % (tot_gen_loss / num_iters), "D: %.5f" % (tot_disc_loss / num_iters),
                  "L1: %.5f" % (tot_l1 / num_iters), "VC: %.5f" % (tot_inv / num_iters),
                  "Adv: %.5f" % (tot_adv / num_iters), "Sharp: %.5f" % (tot_sharp / num_iters))
            # print("\n")
            return tot_l1 / num_iters

    def demo(self):
        print("---------------------------\nStart Inference ...")

        inputs = [cv2.imread(os.path.join(self.demo_params['data_path'], self.demo_params['input_l_name'])),
                  cv2.imread(os.path.join(self.demo_params['data_path'], self.demo_params['input_r_name']))]

        # preprocess
        for i, img in enumerate(inputs):
            h, w = img.shape[:2]
            sz = min(h, w)

            if h != w:  # if is not squared
                pad_h = int((h - sz) / 2.)
                pad_w = int((w - sz) / 2.)
                img = img[pad_h:-pad_h, pad_w:-pad_w]
            if sz != self.params['input_size']:
                img = cv2.resize(img, (self.params['input_size'], self.params['input_size']))
            inputs[i] = (np.asarray(img, dtype=np.float64)[np.newaxis, ...] - float(127.5)) / float(127.5)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print("Loading model ...")
            saver.restore(sess, tf.train.latest_checkpoint(self.params['model_path']))

            pred_res, inv_l, inv_r = sess.run(["generator/decoder/de_tanh:0",
                                               "generator_vcs/decoder/de_l_tanh:0",
                                               "generator_vcs/decoder_1/de_r_tanh:0"],
                                               feed_dict={self.inputs_l: inputs[0],
                                                          self.inputs_r: inputs[1]})

            pred_res = pred_res.squeeze()

            cv2.imwrite(os.path.join(self.demo_params['data_path'], "%s.png" % self.demo_params['output_name']),
                        np.array(pred_res, dtype=np.float64) * 127.5 + 127.5)
            cv2.imwrite(os.path.join(self.demo_params['data_path'], "%s_inv_l.png" % self.demo_params['output_name']),
                        np.array(inv_l.squeeze(), dtype=np.float64) * 127.5 + 127.5)
            cv2.imwrite(os.path.join(self.demo_params['data_path'], "%s_inv_r.png" % self.demo_params['output_name']),
                        np.array(inv_r.squeeze(), dtype=np.float64) * 127.5 + 127.5)

            print("Finish Inference ...")