import scipy.misc
import cv2
import glob
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import itertools
import sklearn.model_selection

input_image_shape = (160, 576)
num_classes = 2
init = tf.truncated_normal_initializer(stddev=0.01)
regularizer = tf.contrib.layers.l2_regularizer(1e-3)

class Image:
    def __init__(self, image_array):
        self.image = image_array

    def to_one_hot(self):
        gt_bg = np.all(self.image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        self.image = gt_image
        return self

def read_image(image_file):
    image_array = scipy.misc.imread(image_file)
    image_array = scipy.misc.imresize(image_array, input_image_shape)
    return Image(image_array)

def plot_images(images):
    n = len(images)
    cols = 3
    rows = round(n/cols) + 1
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(images[i].image)


class Network:

    def convolution_layers(self):
        reg = tf.contrib.layers.l2_regularizer(1e-3),
        self.conv11 = tf.layers.conv2d(self.input, 64, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv11')
        self.conv12 = tf.layers.conv2d(self.conv11, 64, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv12')
        self.pool1 = tf.layers.max_pooling2d(self.conv12, pool_size=(2,2), strides=(2,2), padding='same', name='pool1')
        self.conv21 = tf.layers.conv2d(self.pool1, 128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv21')
        self.conv22 = tf.layers.conv2d(self.conv21, 128, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv22')
        self.pool2 = tf.layers.max_pooling2d(self.conv22, pool_size=(2,2), strides=(2,2), padding='same', name='pool2')
        self.conv31 = tf.layers.conv2d(self.pool2, 256, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv31')
        self.conv32 = tf.layers.conv2d(self.conv31, 256, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv32')
        self.pool3 = tf.layers.max_pooling2d(self.conv32, pool_size=(2,2), strides=(2,2), padding='same', name='pool3')
        self.conv41 = tf.layers.conv2d(self.pool3, 512, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv41')
        self.conv42 = tf.layers.conv2d(self.conv41, 512, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv42')
        self.pool4 = tf.layers.max_pooling2d(self.conv42, pool_size=(2,2), strides=(2,2), padding='same', name='pool4')
        self.conv51 = tf.layers.conv2d(self.pool4, 4096, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv51')
        self.conv52 = tf.layers.conv2d(self.conv51, 4096, kernel_size=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init, name='conv52')
        self.pool5 = tf.layers.max_pooling2d(self.conv52, pool_size=(2,2), strides=(2,2), padding='same', name='pool5')

    def vgg_convolution_layers(self, vgg_path):
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

        self.input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
        self.keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
        self.pool3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
        self.pool4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
        self.pool5 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    def prediction_layers(self):
        self.pred1 = tf.layers.conv2d(self.pool3, num_classes, kernel_size=(1,1), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)
        self.pred2 = tf.layers.conv2d(self.pool4, num_classes, kernel_size=(1,1), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)
        self.pred3 = tf.layers.conv2d(self.pool5, num_classes, kernel_size=(1,1), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)

    def deconvolution_layers(self):
        self.deconv1 = tf.layers.conv2d_transpose(self.pred3, num_classes, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=init, kernel_regularizer=regularizer, name='deconv1')
        add1 = tf.add(self.pred2, self.deconv1)
        self.deconv2 = tf.layers.conv2d_transpose(add1, num_classes, kernel_size=4, strides=2, padding='same',
                                                  kernel_initializer=init, kernel_regularizer=regularizer, name='deconv2')
        add2 = tf.add(self.pred1, self.deconv2)
        self.deconv3 = tf.layers.conv2d_transpose(add2, num_classes, kernel_size=16, strides=8, padding='same',
                                                  kernel_initializer=init, kernel_regularizer=regularizer, name='deconv3')
        self.output = self.deconv3

    def loss_layer(self):
        self.logits = tf.reshape(self.output, (-1, num_classes))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
        self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy_loss)

    def outputs(self):
        self.y_true = tf.reshape(self.label, (-1, num_classes))
        self.im_pred = tf.one_hot(tf.argmax(tf.nn.softmax(self.output), axis=3), depth=2)
        self.y_pred = tf.nn.softmax(self.logits)

    def metrics(self):
        intersection = tf.reduce_sum(self.y_pred * self.y_true)
        union = tf.reduce_sum(self.y_pred) + tf.reduce_sum(self.y_true)
        self.dice_coeff = (2*intersection + 1)/(union + 1)

    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, input_image_shape[0], input_image_shape[1], 3], "input")
        self.label = tf.placeholder(tf.float32, [None, input_image_shape[0], input_image_shape[1], 2], "label")
        self.learning_rate = tf.placeholder(tf.float32, [], "learning_rate")

        #self.convolution_layers()
        self.vgg_convolution_layers("./data/vgg")
        self.prediction_layers()
        self.deconvolution_layers()
        self.loss_layer()
        self.outputs()
        self.metrics()

    def train(self, sess, input_gen, label_gen, learning_rate, keep_prob,
              epochs, batch_size, n_batches, save_path=None, restore_path=None):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver()
        if restore_path: saver.restore(sess, restore_path)
        losses = []
        for e in range(epochs):
            print("Epoch ", e+1)
            input_gen, input_gen_temp = itertools.tee(input_gen)
            label_gen, label_gen_temp = itertools.tee(label_gen)
            for _ in range(n_batches):
                start = time.time()
                batch_input = np.array([im.image for _, im in zip(range(batch_size), input_gen_temp)])
                batch_label = np.array([im.image for _, im in zip(range(batch_size), label_gen_temp)])
                _, loss, dice = sess.run([self.train_op, self.cross_entropy_loss, self.dice_coeff], feed_dict={
                    self.input: batch_input,
                    self.label: batch_label,
                    self.learning_rate: learning_rate,
                    self.keep_prob: keep_prob
                })
                print("training: loss - ", loss, ", dice - ", dice, " in ", time.time() - start)
                losses.append(loss)
            if save_path: saver.save(sess, save_path)
        return losses

    def validate(self, sess, image_gen, label_gen):
        inferred = []
        for im, lb in zip(image_gen, label_gen):
            image_input = np.array(im.image)
            image_shape = image_input.shape
            im_pred, loss, dice = sess.run(
                [self.im_pred, self.cross_entropy_loss, self.dice_coeff],
                     feed_dict={
                         self.input: [image_input],
                         self.keep_prob: 1.0,
                         self.label: [lb.image]})
            print('validation: loss - ', loss, 'dice - ', dice, 'shape - ', im_pred.shape)
            segmentation = im_pred[0][:,:,1].reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[255, 0, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image_input)
            street_im.paste(mask, box=None, mask=mask)
            inferred.append(Image(image_input))
            inferred.append(Image(mask))
            inferred.append(Image(street_im))
        return inferred


    def test(self, sess, image_gen):
        inferred = []
        for im in image_gen:
            image_input = np.array(im.image)
            image_shape = image_input.shape
            im_pred = sess.run(self.im_pred, feed_dict={ self.input: [image_input], self.keep_prob: 1.0 })
            print(im_pred[0].shape)
            segmentation = im_pred[0][:,:,1].reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[255, 0, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image_input)
            street_im.paste(mask, box=None, mask=mask)
            inferred.append(Image(image_input))
            inferred.append(Image(mask))
            inferred.append(Image(street_im))
        return inferred


    def print_shape(self, sess, layers, imageset):
        input_images = [image.im for image in imageset.images]
        sh = sess.run([tf.shape(l) for l in layers], feed_dict={ self.input: input_images })
        print(sh)
        sh = sess.run(layers, feed_dict={self.input: input_images})
        print(sh)

    def debug_shapes(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            l1, l2 = sess.run([tf.shape(self.input), tf.shape(self.output)], feed_dict={self.input: np.ones(((1, 128, 128, 3)))})
            print(l1, l2)

# background_color = np.array([0, 0, 0])
# features = glob.glob(os.path.join("data/train", "*.jpg"))
# labels = glob.glob(os.path.join("data/train_masks", "*.gif"))
# tests = glob.glob(os.path.join("data/test", "*.jpg"))

background_color = np.array([255, 0, 0])
features = glob.glob(os.path.join("data/data_road/training/image_2", "*.png"))
labels = glob.glob(os.path.join("data/data_road/training/gt_image_2", "*road*.png"))
tests = glob.glob(os.path.join("data/data_road/testing/image_2", "*.png"))[:5]

trains, vals, label_trains, label_vals = sklearn.model_selection.train_test_split(features, labels, test_size=0.33, random_state=42)
X_train = (read_image(f) for f in trains)
y_train = (read_image(f).to_one_hot() for f in label_trains)
X_val = (read_image(f) for f in vals)
y_val = (read_image(f).to_one_hot() for f in label_vals)
X_test = (read_image(f) for f in tests)


with tf.Session() as sess:
    nw = Network()
    epochs = 2
    batch_size = 5
    n_batches = int((len(trains) - 1) / batch_size + 1)
    nw.train(sess, X_train, y_train, 0.0001, 0.5, epochs, batch_size, n_batches, save_path="./road2.ckpt")
    #nw.validate(sess, X_val, y_val)
    inferred = nw.test(sess, X_test)
    print("inferred ", len(inferred))
    plot_images(inferred)
    plt.show()
