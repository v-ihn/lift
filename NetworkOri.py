from tensorflow.python import keras as K
import numpy as np
import os


class NetworkOri:
    def __init__(self, network_desc, learning_rate=0.001, model_file='ori.h5'):
        self.model_dir = 'models/ori/'
        self.model_file = model_file

        self.network_desc = network_desc

        self.p1 = K.Input(shape=(64, 64, 1), dtype='float32', name='ori_input_p1')
        self.p2 = K.Input(shape=(64, 64, 1), dtype='float32', name='ori_input_p2')
        self.p3 = K.Input(shape=(64, 64, 1), dtype='float32', name='ori_input_p3')

        # self.output1, self.output2, self.output3 = self.init_outputs()
        # self.training_loss = self.init_training_loss()
        # self.optimizer = self.init_optimizer(learning_rate)
        #
        # self.training_model, self.desc_model = self.init_models()
    #     self.restore_weights()
    #
    # def save_weights(self):
    #     if not os.path.isdir(self.model_dir):
    #         os.makedirs(self.model_dir)
    #     self.training_model.save_weights(self.model_dir + self.model_file)
    #
    # def restore_weights(self):
    #     if os.path.isfile(self.model_dir + self.model_file):
    #         self.training_model.load_weights(self.model_dir + self.model_file)
    #         self.desc_model.load_weights(self.model_dir + self.model_file)
