from tensorflow.python import keras as K
import numpy as np
import os


class NetworkDesc3Keras:
    def __init__(self, learning_rate=0.001, model_file='desc3_keras.h5'):
        self.model_dir = 'models/desc3_keras/'
        self.model_file = model_file

        self.p1 = K.Input(shape=(64, 64, 1), dtype='float32', name='desc_triplet_p1')
        self.p2 = K.Input(shape=(64, 64, 1), dtype='float32', name='desc_triplet_p2')
        self.p3 = K.Input(shape=(64, 64, 1), dtype='float32', name='desc_triplet_p3')

        self.output1, self.output2, self.output3 = self.init_outputs()
        self.training_loss = self.init_training_loss()
        self.optimizer = self.init_optimizer(learning_rate)

        self.training_model, self.desc_model = self.init_models()
        self.restore_weights()

    def single_branch_model(self):
        model = K.Sequential()
        model.add(K.layers.BatchNormalization(input_shape=(64, 64, 1)))

        model.add(K.layers.Conv2D(filters=32, kernel_size=7, activation='relu'))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.AveragePooling2D(pool_size=2, strides=2))

        model.add(K.layers.Conv2D(filters=64, kernel_size=6, activation='relu'))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.AveragePooling2D(pool_size=3, strides=3))

        model.add(K.layers.Conv2D(filters=128, kernel_size=5, activation='relu'))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.AveragePooling2D(pool_size=4, strides=4))
        model.add(K.layers.Reshape((128, )))

        return model

    def init_outputs(self):
        single_branch_model = self.single_branch_model()
        output1 = single_branch_model(self.p1)
        output2 = single_branch_model(self.p2)
        output3 = single_branch_model(self.p3)
        return output1, output2, output3

    def init_training_loss(self, margin=4.0):
        def calculate_loss(outputs):
            loss_pos = K.backend.sqrt(K.backend.sum(K.backend.square(outputs[0] - outputs[1]), axis=1, keepdims=True))

            pair_dist_1_to_3 = K.backend.sqrt(K.backend.sum(K.backend.square(outputs[0] - outputs[2]), axis=1, keepdims=True))
            pair_dist_2_to_3 = K.backend.sqrt(K.backend.sum(K.backend.square(outputs[1] - outputs[2]), axis=1, keepdims=True))
            d_neg = K.backend.minimum(pair_dist_1_to_3, pair_dist_2_to_3)
            loss_neg = K.backend.relu(margin - d_neg)

            return loss_pos + loss_neg

        return K.layers.Lambda(calculate_loss)([self.output1, self.output2, self.output3])

    def init_optimizer(self, learning_rate):
        return K.optimizers.Adam(learning_rate)

    def init_models(self):
        training_model = K.Model(inputs=[self.p1, self.p2, self.p3], outputs=self.training_loss)
        training_model.compile(optimizer=self.optimizer, loss='mean_absolute_error')
        # training_model.summary()

        desc_model = K.Model(inputs=[self.p1], outputs=[self.output1])
        # desc_model.summary()

        return training_model, desc_model

    def train_model(self, input1, input2, input3):
        return self.training_model.train_on_batch([input1, input2, input3], np.zeros(input1.shape[0]))

    def test_model(self, input1, input2, input3):
        return self.training_model.test_on_batch([input1, input2, input3], np.zeros(input1.shape[0]))

    def get_losses(self, input1, input2, input3):
        return self.training_model.predict_on_batch([input1, input2, input3])

    def hardmine_train(self, input1, input2, input3, mining_ratio):
        if mining_ratio == 1:
            return self.train_model(input1, input2, input3)
        else:
            losses = self.get_losses(input1, input2, input3).flatten()
            train_with = int(losses.size / mining_ratio)
            training_indices = np.argsort(losses)[::-1][:train_with]
            training_input1 = input1[training_indices]
            training_input2 = input2[training_indices]
            training_input3 = input3[training_indices]
            return self.train_model(training_input1, training_input2, training_input3)

    def save_weights(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.training_model.save_weights(self.model_dir + self.model_file)

    def restore_weights(self):
        if os.path.isfile(self.model_dir + self.model_file):
            self.training_model.load_weights(self.model_dir + self.model_file)
            self.desc_model.load_weights(self.model_dir + self.model_file)
