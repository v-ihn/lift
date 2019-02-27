import tensorflow as tf
import numpy as np


class NetworkDesc3:
    def __init__(self):
        self.model_file = 'models/desc3/desc3'

        self.is_training = tf.placeholder(tf.bool, name='desc_triplet_is_training')
        self.input1 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='desc_triplet_input1')
        self.input2 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='desc_triplet_input2')
        self.input3 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='desc_triplet_input3')

        self.output_1, self.output_2, self.output_3 = self.init_network()
        self.loss = self.init_loss2()
        self.optimizer = self.init_optimizer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.restore_model()

    def init_network(self):
        with tf.variable_scope("network_desc_triplet") as scope:
            output_1 = self.single_network(self.input1)
            scope.reuse_variables()
            output_2 = self.single_network(self.input2)
            output_3 = self.single_network(self.input3)
        return output_1, output_2, output_3

    def init_loss(self, margin=5.0):
        with tf.variable_scope("desc_triplet_loss"):
            d_pos = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_2), axis=1))

            pair_dist_1_to_3 = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_3), axis=1))
            pair_dist_2_to_3 = tf.sqrt(tf.reduce_sum(tf.square(self.output_2 - self.output_3), axis=1))
            d_neg = tf.minimum(pair_dist_1_to_3, pair_dist_2_to_3)

            return tf.nn.relu(d_pos - d_neg + margin)

    def init_loss2(self, margin=5.0):
        with tf.variable_scope("desc_triplet_loss"):
            loss_pos = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_2), axis=1))

            pair_dist_1_to_3 = tf.sqrt(tf.reduce_sum(tf.square(self.output_1 - self.output_3), axis=1))
            pair_dist_2_to_3 = tf.sqrt(tf.reduce_sum(tf.square(self.output_2 - self.output_3), axis=1))
            d_neg = tf.minimum(pair_dist_1_to_3, pair_dist_2_to_3)
            loss_neg = tf.nn.relu(margin - d_neg)

            return loss_pos + loss_neg

    def init_optimizer(self, learning_rate=0.005):
        with tf.variable_scope("desc_triplet_optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def single_network(self, input):
        with tf.variable_scope('desc_triplet_cnn_layer1'):
            input_normalized = tf.layers.batch_normalization(input, training=self.is_training)
            conv1 = tf.layers.conv2d(inputs=input_normalized, filters=32, kernel_size=7, padding='valid', activation=tf.nn.relu)
            conv1_normalized = tf.layers.batch_normalization(conv1, training=self.is_training)
            # conv1_activ = tf.nn.relu(conv1_normalized)
            pool1 = tf.layers.average_pooling2d(inputs=conv1_normalized, pool_size=2, strides=2)

        with tf.variable_scope('desc_triplet_cnn_layer2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=6, padding='valid', activation=tf.nn.relu)
            conv2_normalized = tf.layers.batch_normalization(conv2, training=self.is_training)
            # conv2_activ = tf.nn.relu(conv2_normalized)
            pool2 = tf.layers.average_pooling2d(inputs=conv2_normalized, pool_size=3, strides=3)

        with tf.variable_scope('desc_triplet_cnn_layer3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=5, padding='valid', activation=tf.nn.relu)
            conv3_normalized = tf.layers.batch_normalization(conv3, training=self.is_training)
            # conv3_activ = tf.nn.relu(conv3_normalized)
            pool3 = tf.layers.average_pooling2d(inputs=conv3_normalized, pool_size=4, strides=4)

        return tf.reshape(pool3, (-1, 128))

    def train_model(self, input1, input2, input3):
        _, train_loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.input1: input1, self.input2: input2,
                                                                              self.input3: input3, self.is_training: True})
        return np.array(train_loss).mean()

    def test_model(self, input1):
        output = self.sess.run(self.output_1, feed_dict={self.input1: input1, self.is_training: False})
        return output

    def test_model_loss(self, input1, input2, input3):
        loss = self.sess.run(self.loss, feed_dict={self.input1: input1, self.input2: input2,
                                                   self.input3: input3, self.is_training: False})
        return loss

    def hardmine_train(self, input1, input2, input3, iteration):
        mining_ratio = min(2 ** int(iteration / 20000), 4)
        if mining_ratio == 1:
            return self.train_model(input1, input2, input3)
        else:
            losses = self.sess.run(self.loss, feed_dict={self.input1: input1, self.input2: input2,
                                                         self.input3: input3, self.is_training: True})
            train_with = int(losses.size / mining_ratio)
            training_indices = np.argsort(losses)[::-1][:train_with]
            training_input1 = input1[training_indices]
            training_input2 = input2[training_indices]
            training_input3 = input3[training_indices]
            return self.train_model(training_input1, training_input2, training_input3)

    def hardmine_train2(self, input1, input2, input3, mining_ratio):
        if mining_ratio == 1:
            return self.train_model(input1, input2, input3)
        else:
            losses = self.sess.run(self.loss, feed_dict={self.input1: input1, self.input2: input2,
                                                         self.input3: input3, self.is_training: True})
            train_with = int(losses.size / mining_ratio)
            training_indices = np.argsort(losses)[::-1][:train_with]
            training_input1 = input1[training_indices]
            training_input2 = input2[training_indices]
            training_input3 = input3[training_indices]
            return self.train_model(training_input1, training_input2, training_input3)

    def save_model(self):
        self.saver.save(self.sess, self.model_file)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_file)
