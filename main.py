try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import tensorflow as tf

from Dataset import *
from Network_desc_2 import *
from Network_desc_3 import *


folder = 'img'
kp_file = 'world_points_complete.txt'

dataset = Dataset(folder, kp_file)
network_desc_2 = NetworkDesc2()
# network_desc_3 = NetworkDesc3()

for i in range(0, 100000000):
    # labels, patches = dataset.get_batch_desc_triplets(128)
    # # train_loss = network_desc_3.train_model(patches[0], patches[1], patches[2])
    # train_loss = network_desc_3.hardmine_train(patches[0], patches[1], patches[2], i)
    # if i % 10 == 0:
    #     print('Iteration {}, Train loss {}'.format(i, train_loss))
    # if i % 2000 == 0:
    #     network_desc_3.save_model()

    matching = True if i % 2 == 0 else False
    labels, patches = dataset.get_batch_desc_pairs(128, matching)
    # train_loss = network_desc_2.train_model(patches[0], patches[1], matching)
    train_loss = network_desc_2.hardmine_train(patches[0], patches[1], matching, i)
    if i % 10 in (0, 1):
        print('Iteration {}, Train loss {}'.format(i, train_loss))
    if i % 2000 == 0:
        network_desc_2.save_model()


# labels, patches = dataset.get_triplet(True)
# out1 = network_desc_3.test_model([patches[0]])
# out2 = network_desc_3.test_model([patches[1]])
# out3 = network_desc_3.test_model([patches[2]])
#
# d_pos = (np.sum(np.square(out1[0] - out2[0]))) ** 0.5
# d_neg = (np.sum(np.square(out1[0] - out3[0]))) ** 0.5
# print(d_pos)

# labels, patches = dataset.get_matching_pair()
# img = patches[0]
# print(labels[0])
# print(img.shape)
# cv2.imshow('Img', img)
# cv2.waitKey()
