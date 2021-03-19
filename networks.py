import tensorflow as tf
from layers import *

def SCGN(inputs_l, inputs_r, targets):
    pred_tanh = VSN(inputs_l, inputs_r)

    real_logits = Discriminator(targets, reuse=False)
    fake_logits = Discriminator(pred_tanh, reuse=True)

    inv_l, inv_r = VDN(pred_tanh)

    return pred_tanh, real_logits, fake_logits, inv_l, inv_r

def VSN(inputs_l, inputs_r):
    with tf.variable_scope("generator"):
        xavier_init = tf.contrib.layers.xavier_initializer(seed=1)

        with tf.variable_scope('encoder_reuse'):
            out_l, ec1fea_l, ec2fea_l, ec3fea_l = shared_encoder(inputs_l, xavier_init, reuse=False)
            out_r, ec1fea_r, ec2fea_r, ec3fea_r = shared_encoder(inputs_r, xavier_init, reuse=True)

        print("Concatenating Encoder Outputs...")
        encoder_out = tf.concat([out_l, out_r], axis=3)

        feats = [ec1fea_l, ec1fea_r, ec2fea_l, ec2fea_r, ec3fea_l, ec3fea_r]
        dc_tanh, dc_out = decoder(encoder_out, xavier_init, feats)
        return dc_tanh

def VDN(input_layer):
    with tf.variable_scope("generator_vcs"):
        xavier_init = tf.contrib.layers.xavier_initializer(seed=1)

        ed_out, ec1fea, ec2fea, ec3fea = shared_encoder(input_layer, xavier_init,
                                                        reuse=False, name="ed_inv_")

        ed_left = tf.layers.conv2d(inputs=ed_out, kernel_size=[1, 1], strides=1,
                                     filters=128, padding="same", name="ed_left")
        ed_right = tf.layers.conv2d(inputs=ed_out, kernel_size=[1, 1], strides=1,
                                      filters=128, padding="same", name="ed_right")

        feats_l = [ec1fea, ec2fea, ec3fea]

        dc_left, _ = decoder(ed_left, xavier_init, feats_l, reuse=False, name="de_l_")
        dc_right, _ = decoder(ed_right, xavier_init, feats_l, reuse=False, name="de_r_")
        return dc_left, dc_right

def Discriminator(input_layer, reuse=False):
    print("Building Discriminator Network...")
    with tf.variable_scope("discriminator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        xavier_init = tf.contrib.layers.xavier_initializer(seed=1)

        disc1=lrelu(conv2d(input_layer, 32, name="d1_conv"), name="d1_lrelu")
        disc2=lrelu(batch_norm(conv2d(disc1, 64, name="d2_conv"), name="d2_bn"), name="d2_lrelu")
        disc3=lrelu(batch_norm(conv2d(disc2, 128, name="d3_conv"), name="d3_bn"), name="d3_lrelu")
        disc4=lrelu(batch_norm(conv2d(disc3, 256, name="d4_conv"), name="d4_bn"), name="d4_lrelu")

        logits = tf.layers.conv2d(inputs=disc4, kernel_size=[14,14],strides=1, filters=1,
                                    padding="valid", kernel_initializer=xavier_init)
        return logits


def shared_encoder(input_layer, kernel_init, reuse=False, name="ed_"):
    print("Building Shared Encoder...")
    with tf.variable_scope('encoder'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        k_size=3

        ec1 = tf.layers.conv2d(inputs=input_layer, kernel_size=[7, 7], strides=1,
                                 filters=32, padding="same",
                                 activation=tf.nn.leaky_relu, name=name + "ec1",
                                 kernel_initializer=kernel_init)
        ec1fea = tf.layers.conv2d(inputs=ec1, kernel_size=[1, 1], strides=1,
                                    filters=16, padding="same",
                                    activation=tf.nn.leaky_relu, name=name + "ec1fea")

        ep1 = tf.layers.max_pooling2d(inputs=ec1, pool_size=[3, 3], strides=2,
                                        padding="same", name=name + "mp1")
        ep1 = resblock(resblock(ep1, name=name + "ep1res1"), name=name + "ep1res2")
        # 128x64

        ec2 = tf.layers.conv2d(inputs=ep1, kernel_size=[5, 5], strides=1,
                                 filters=64, padding="same",
                                 activation=tf.nn.leaky_relu, name=name + "ec2",
                                 kernel_initializer=kernel_init)
        ec2fea = tf.layers.conv2d(inputs=ec2, kernel_size=[1, 1], strides=1,
                                    filters=32, padding="same",
                                    activation=tf.nn.leaky_relu,name=name + "ec2fea")

        ep2 = tf.layers.max_pooling2d(inputs=ec2, pool_size=[3, 3], strides=2,
                                        padding="same", name=name + "mp2")
        ep2 = resblock(resblock(ep2, name=name + "ep2res1"), name=name + "ep2res2")
        # 64x128

        ec3 = tf.layers.conv2d(inputs=ep2, kernel_size=[k_size, k_size], strides=1,
                                 filters=128, padding="same", name=name + "ec3")
        ec3fea = tf.layers.conv2d(inputs=ec3, kernel_size=[1, 1], strides=1,
                                    filters=128, padding="same",
                                    activation=tf.nn.leaky_relu, name=name + "ec3fea")

        ec3 = resblock(resblock(ec3, name=name + "ec3res1"), name=name + "ec3res2")
        # 64x256

        if name=="ed_inv_":
            ed_out = tf.layers.conv2d(inputs=ec3, kernel_size=[k_size, k_size], strides=1,
                                        filters=256, padding="same", name=name + "ec4")
        else:
            ec4 = atrous_conv2d(ec3, 128, rate=2,name=name+"aconv2d_1")
            ec5 = atrous_conv2d(ec4, 128, rate=2,name=name+"aconv2d_2")
            ed_out = atrous_conv2d(ec5, 128, rate=2, name=name + "aconv2d_3")

        for i in range(4):
            ed_out = resblock(ed_out, k_h=k_size, k_w=k_size, name=name + "res%d" % (i+1))
    return ed_out,ec1fea,ec2fea,ec3fea

def decoder(encoder_out, kernel_init, feats, im_size=224, reuse=False, name="de_"):
    print("Building Decoder...")
    with tf.variable_scope('decoder'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # ed_out 64x512
        k_size=3

        dc3 = tf.layers.conv2d(inputs=encoder_out, filters=128, kernel_size=[k_size, k_size],
                                 padding='same', activation=tf.nn.relu,
                                 kernel_initializer=kernel_init, name=name + "dc3")
        upsample3 = tf.image.resize_nearest_neighbor(dc3, (int(im_size / 4), int(im_size / 4)), name=name + "nn3")
        upsample3 = resblock_relu(resblock_relu(upsample3, name=name + "up3res1"), name=name + "up3res2")
        # print("feats len=",len(feats))
        if len(feats) == 6:
            cat3 = tf.concat([upsample3, feats[4], feats[5]], axis=3)
        elif len(feats) == 3:
            cat3 = tf.concat([upsample3, feats[2]], axis=3)
        else:
            print("Error in features concat!")
            exit(1)

        dc4 = tf.layers.conv2d(inputs=cat3, filters=64, kernel_size=[k_size, k_size],
                                 padding='same', activation=tf.nn.relu,
                                 kernel_initializer=kernel_init, name=name + "dc4")
        upsample4 = tf.image.resize_nearest_neighbor(dc4, (int(im_size / 2), int(im_size / 2)), name=name + "nn4")
        upsample4 = resblock_relu(resblock_relu(upsample4, name=name + "up4res1"), name=name + "up4res2")

        if len(feats) == 6:
            cat4 = tf.concat([upsample4, feats[2], feats[3]], axis=3)
        elif len(feats) == 3:
            cat4 = tf.concat([upsample4, feats[1]], axis=3)
        else:
            print("Error in features concat!")
            exit(1)

        dc5 = tf.layers.conv2d(inputs=cat4, filters=32, kernel_size=[5, 5],
                                 padding='same', activation=tf.nn.relu,
                                 kernel_initializer=kernel_init, name=name + "dc5")
        upsample5 = tf.image.resize_nearest_neighbor(dc5, (im_size, im_size), name=name + "nn5")
        upsample5 = resblock_relu(resblock_relu(upsample5, name=name + "up5res1"), name=name + "up5res2")

        if len(feats) == 6:
            cat5 = tf.concat([upsample5, feats[0], feats[1]], axis=3)
        elif len(feats) == 3:
            cat5 = tf.concat([upsample5, feats[0]], axis=3)
        else:
            print("Error in features concat!")
            exit(1)

        dc6 = tf.layers.conv2d(inputs=cat5, filters=32, kernel_size=[7, 7],
                                 padding='same', activation=tf.nn.relu,
                                 kernel_initializer=kernel_init, name=name + "dc6")

        logits = tf.layers.conv2d(inputs=dc6, filters=3, kernel_size=[3, 3],
                                  padding='same', name=name + "logit")
        pred_tanh = tf.nn.tanh(logits, name=name + "tanh")
        return pred_tanh, logits