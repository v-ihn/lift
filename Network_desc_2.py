import tensorflow as tf
import numpy as np


class NetworkDesc2:
    def __init__(self, margin=4.0):
        self.model_file = 'models/desc2/desc2'
        self.margin = margin

        self.is_training = tf.placeholder(tf.bool, name='desc_pair_is_training')
        self.matching_pairs = tf.placeholder(tf.bool, name='desc_pair_matching')
        self.input1 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='desc_pair_input1')
        self.input2 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='desc_pair_input2')

        self.output_1, self.output_2 = self.init_network()
        self.loss = self.init_loss()
        self.optimizer = self.init_optimizer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.restore_model()

    def init_network(self):
        with tf.variable_scope("network_desc_pair") as scope:
            output_1 = self.single_network(self.input1)
            scope.reuse_variables()
            output_2 = self.single_network(self.input2)
        return output_1, output_2

    def init_loss(self):
        with tf.variable_scope("desc_pair_loss"):
            loss = tf.cond(self.matching_pairs, true_fn=self.loss_matching, false_fn=self.loss_non_matching)
        return loss

    def loss_matching(self):
        with tf.variable_scope("desc_pair_loss_matching"):
            loss = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_2), axis=1))
        return loss

    def loss_non_matching(self):
        with tf.variable_scope("desc_pair_loss_non_matching"):
            pair_dist_1_to_3 = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_2), axis=1))
            loss = tf.nn.relu(self.margin - pair_dist_1_to_3)
        return loss

    def init_optimizer(self, learning_rate=0.001):
        with tf.variable_scope("desc_pair_optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                return optimizer

    def single_network(self, input):
        with tf.variable_scope('desc_cnn_layer1'):
            input_normalized = tf.layers.batch_normalization(input, training=self.is_training)
            conv1 = tf.layers.conv2d(inputs=input_normalized, filters=32, kernel_size=7, padding='valid', activation=tf.nn.relu)
            conv1_normalized = tf.layers.batch_normalization(conv1, training=self.is_training)
            # conv1_activ = tf.nn.relu(conv1_normalized)
            pool1 = tf.layers.average_pooling2d(inputs=conv1_normalized, pool_size=2, strides=2)

        with tf.variable_scope('desc_cnn_layer2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=6, padding='valid', activation=tf.nn.relu)
            conv2_normalized = tf.layers.batch_normalization(conv2, training=self.is_training)
            # conv2_activ = tf.nn.relu(conv2_normalized)
            pool2 = tf.layers.average_pooling2d(inputs=conv2_normalized, pool_size=3, strides=3)

        with tf.variable_scope('desc_cnn_layer3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=5, padding='valid', activation=tf.nn.relu)
            conv3_normalized = tf.layers.batch_normalization(conv3, training=self.is_training)
            # conv3_activ = tf.nn.relu(conv3_normalized)
            pool3 = tf.layers.average_pooling2d(inputs=conv3_normalized, pool_size=4, strides=4)

        return tf.reshape(pool3, (-1, 128))

    def train_model(self, input1, input2, matching_pairs):
        _, train_loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.input1: input1, self.input2: input2,
                                                                              self.matching_pairs: matching_pairs,
                                                                              self.is_training: True})
        return np.array(train_loss).mean()

    def hardmine_train(self, input1, input2, matching_pairs, iteration):
        mining_ratio = min(2 ** int(iteration / 15000), 4)
        if mining_ratio == 1:
            return self.train_model(input1, input2, matching_pairs)
        else:
            losses = self.sess.run(self.loss, feed_dict={self.input1: input1, self.input2: input2,
                                                         self.matching_pairs: matching_pairs,
                                                         self.is_training: True})
            train_with = int(losses.size / mining_ratio)
            training_indices = np.argsort(losses)[::-1][:train_with]
            training_input1 = input1[training_indices]
            training_input2 = input2[training_indices]
            return self.train_model(training_input1, training_input2, matching_pairs)

    def save_model(self):
        self.saver.save(self.sess, self.model_file)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_file)
