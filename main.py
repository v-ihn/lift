try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import sys
from tensorflow.python import keras as K

import datasets
from Dataset import *
from NetworkDesc import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist()
# train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
# train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=True)
train_images, train_labels, test_images, test_labels = datasets.brown('trevi')

train_dataset = Dataset(train_images, train_labels)
test_dataset = Dataset(test_images, test_labels, mean=train_dataset.mean, std=train_dataset.std)

# train_dataset.create_fixed_epoch(38400)

test_batch = test_dataset.get_batch_triplets(1024)

network_desc = NetworkDesc(learning_rate=0.001)
mining_ratio = 2
batch_size = 128

for i in range(1, 15001):
    patches = train_dataset.get_batch_triplets(batch_size)
    # patches = train_dataset.get_batch_triplets_from_fixed_epoch(128)

    train_loss = network_desc.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
    if i % 10 == 0:
        print('Iteration {}, Train loss {}'.format(i, train_loss))
    if i % 1000 == 0:
        network_desc.save_weights()
        print('Test loss: {}'.format(network_desc.test_model(test_batch[0], test_batch[1], test_batch[2])))
    # if i == 5000:
    #     mining_ratio = 8
    #     batch_size = 1024
    # if i == 10000 and K.backend.get_value(network_desc.training_model.optimizer.lr) > 0.001:
    #     K.backend.set_value(network_desc.training_model.optimizer.lr, 0.001)
    # if i % 5000 == 0 and mining_ratio < 4:
    #     mining_ratio *= 2
    #     batch_size *= 2


print('Data from training dataset: ')
patches = train_dataset.get_batch_triplets(256)
loss_test = network_desc.test_model(patches[0], patches[1], patches[2])
loss_train = network_desc.train_model(patches[0], patches[1], patches[2])
print('Test: {}, Train: {}'.format(loss_test, loss_train))

triplet = train_dataset.get_batch_triplets(1)
d1 = network_desc.get_descriptor(triplet[0])
d2 = network_desc.get_descriptor(triplet[1])
d3 = network_desc.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.minimum(d_1_3, d_2_3)
# d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)
print('--------------------------------------------------------')


print('Data from testing dataset: ')
print('Test loss: {}'.format(network_desc.test_model(test_batch[0], test_batch[1], test_batch[2])))

triplet = test_dataset.get_batch_triplets(1)
d1 = network_desc.get_descriptor(triplet[0])
d2 = network_desc.get_descriptor(triplet[1])
d3 = network_desc.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.minimum(d_1_3, d_2_3)
# d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)
