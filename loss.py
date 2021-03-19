import tensorflow as tf
import functools
import numpy as np
import math
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops

def Loss(weights, targets, pred_tanh, real_logits, fake_logits,
         inputs_l, inputs_r, inv_l, inv_r, mode='train', choice='l1'):
    # losses for disc.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_logits), logits=real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_logits), logits=fake_logits))
    d_loss = d_loss_real + d_loss_fake

    # losses for gen.
    l1_loss = tf.reduce_mean(tf.abs(pred_tanh - targets))
    inv_loss = inverse_loss(inputs_l, inputs_r, inv_l, inv_r)
    adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_logits), logits=fake_logits))
    sharp_loss, pred_gau = sharpness_loss(pred_tanh, targets, choice) if weights[2] > 0 else [tf.zeros_like(adv_loss), tf.zeros_like(pred_tanh)]

    g_loss = l1_loss + weights[0] * adv_loss + weights[1] * inv_loss + weights[2] * sharp_loss

    metrics = compute_metrics(pred_tanh, targets) if mode == 'test' else None
    iter_metrics = compute_metrics(pred_tanh, targets, 'iter')
    return d_loss, g_loss, l1_loss, inv_loss, adv_loss, sharp_loss, metrics, pred_gau, iter_metrics

def inverse_loss(input_l, input_r, dc_l, dc_r):
    perp_l = tf.abs(dc_l - input_l)
    perp_r = tf.abs(dc_r - input_r)
    return tf.reduce_mean(perp_l) + tf.reduce_mean(perp_r)

def sharpness_loss(pred, targ, choice='l1'):
    pred_s, pred_gau = sharpness(pred)
    targ_s, _ = sharpness(targ)

    if choice == 'l1':
        return tf.reduce_mean(tf.abs(pred_s - targ_s)), pred_gau
    elif choice == 'mse':
        return tf.reduce_mean((pred_s - targ_s)**2), pred_gau
    else:
        AssertionError("Invalid choice for sharpness loss!")

def sharpness(x, block_size=8):
    """global sharpness (ref. from logs model)"""

    def gaussian_kernel(size: int, mean: float, std: float):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(loc=float(mean), scale=float(std))
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        return gauss_kernel / (tf.reduce_sum(gauss_kernel)+1e-8)

    def calc_std(img):
        '''calculate sharpness std. in blocks (higher score for better quality)
       --> sqrt(conv(x - avg_pool(x, k, s=1))^2, ones(k,k), s=k) / Z)'''
        img_mean = tf.layers.average_pooling2d(img, block_size, strides=1, padding="SAME")
        sum_filter = tf.tile(tf.ones([block_size, block_size])[:, :, tf.newaxis, tf.newaxis],
                             (1,1,img.shape[3],1))
        img_sum = tf.nn.conv2d(tf.pow(img - img_mean, 2), sum_filter,
                               strides=[1, block_size, block_size, 1], padding="SAME")

        std = tf.sqrt((tf.abs(img_sum) + 1e-8) / (block_size**2))
        return std

    # obtain reblurred synthesized image
    # gauss_kernel = tf.tile(gaussian_kernel(3, 0, 5)[:, :, tf.newaxis, tf.newaxis],
    #                        (1,1,x.shape[3],x.shape[3]))
    gauss_kernel = gaussian_kernel(3, 0, 5)[:, :, tf.newaxis, tf.newaxis]
    x_r, x_g, x_b = tf.expand_dims(x[...,0],-1), tf.expand_dims(x[...,1],-1), tf.expand_dims(x[...,2],-1)
    x_r_blur = tf.nn.conv2d(x_r, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    x_g_blur = tf.nn.conv2d(x_g, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    x_b_blur = tf.nn.conv2d(x_b, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    x_gau = tf.concat([x_r_blur, x_g_blur, x_b_blur], axis=3)

    # calculate sharpness score between x and x_gau
    std_x = calc_std(x)
    std_x_gau = calc_std(x_gau)
    Z = tf.cast(tf.floor(tf.shape(x)[1] / block_size) * tf.floor(tf.shape(x_gau)[2] / block_size), tf.float32)
    # Z = 28**2
    score = tf.sqrt(tf.abs(std_x - std_x_gau) + 1e-8) / Z

    return score, x_gau


def compute_metrics(pred_tanh, targets, type='epoch'):
    pred_255 = pred_tanh * 127.5 + 127.5
    targets_255 = targets * 127.5 + 127.5

    psnr = image_psnr(tf.reduce_mean((pred_255 - targets_255) ** 2))
    msssim = tf.reduce_mean(tf.image.ssim_multiscale(pred_255, targets_255, 255))
    if type == 'iter':
        return psnr, msssim
    mse = tf.reduce_mean((pred_tanh - targets) ** 2) * 127.5 + 127.5
    l1 = tf.reduce_mean(tf.abs(pred_tanh - targets))
    return psnr, msssim, mse, l1

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def image_psnr(mse):
    return 10 * log10(255.0 * 255.0 / (mse))

class Inception_Score(object):
    '''
    From https: // github.com / tsc2017 / inception - score
    '''
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.sess = tf.InteractiveSession()
        self.images = tf.placeholder(tf.float32, [batch_size, None, None, 3])
        self.logits = self.inception_logits()

    def inception_logits(self, num_splits=1):
        size = 299
        tfgan = tf.contrib.gan

        images = tf.image.resize_bilinear(self.images, [size, size])
        generated_images_list = array_ops.split(
            images, num_or_size_splits=num_splits)

        logits = functional_ops.map_fn(
            fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
        logits = array_ops.concat(array_ops.unstack(logits), 0)
        return logits

    def get_inception_probs(self, images):
        preds = []
        num = np.shape(images)[0]
        n_batches = num // self.batch_size

        for i in range(n_batches):
            inp = images[i * self.batch_size:(i + 1) * self.batch_size]
            pred = self.logits.eval({self.images:inp}, self.sess)[:,:1000]
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        preds=np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        return preds

    @staticmethod
    def preds2score(preds, splits):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def run(self, images, splits=10):
        assert(type(images) == np.ndarray)
        assert(len(images.shape) == 4)

        preds = self.get_inception_probs(images)
        mean, std = self.preds2score(preds, splits)
        return mean, std