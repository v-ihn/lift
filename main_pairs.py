try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import sys
from tensorflow.python import keras as K

import datasets
from Dataset import *
from NetworkDesc2 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# train_images, train_labels, test_images, test_labels = datasets.mnist()
# train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
# train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=False)
train_images, train_labels, test_images, test_labels = datasets.brown('trevi')

train_dataset = Dataset(train_images, train_labels)
test_dataset = Dataset(test_images, test_labels, mean=train_dataset.mean, std=train_dataset.std)

# train_dataset.create_fixed_epoch(38400)

test_batch_matching = test_dataset.get_batch_pairs(128, True)
test_batch_non_matching = test_dataset.get_batch_pairs(128, False)

network_desc2 = NetworkDesc2(learning_rate=0.01)
mining_ratio = 1
batch_size = 128

for i in range(1, 20001):
    matching = True if i % 2 == 0 else False
    patches = train_dataset.get_batch_pairs(batch_size, matching)

    train_loss = network_desc2.hardmine_train(patches[0], patches[1], matching, mining_ratio)
    if i % 10 in (0, 1):
        print('Iteration {}, Matching {}, Train loss {}'.format(i, matching, train_loss))
    if i % 1000 == 0:
        network_desc2.save_weights()
        print('Test loss matching: {}'.format(network_desc2.test_model(test_batch_matching[0], test_batch_matching[1], True)))
        print('Test loss non matching: {}'.format(network_desc2.test_model(test_batch_non_matching[0], test_batch_non_matching[1], False)))
    # if i == 10000:
    #     mining_ratio = 8
    #     batch_size = 1024
    # if i == 15000 and K.backend.get_value(network_desc2.training_model.optimizer.lr) > 0.001:
    #     K.backend.set_value(network_desc2.training_model.optimizer.lr, 0.001)
    if i % 5000 == 0 and mining_ratio < 8:
        mining_ratio *= 2
        batch_size *= 2


print('Data from training dataset: ')
patches_matching = train_dataset.get_batch_pairs(256, True)
patches_non_matching = train_dataset.get_batch_pairs(256, False)

loss_test_matching = network_desc2.test_model(patches_matching[0], patches_matching[1], True)
loss_train_matching = network_desc2.train_model(patches_matching[0], patches_matching[1], True)

loss_test_non_matching = network_desc2.test_model(patches_non_matching[0], patches_non_matching[1], False)
loss_train_non_matching = network_desc2.train_model(patches_non_matching[0], patches_non_matching[1], False)

print('Matching, Test: {}, Train: {}'.format(loss_test_matching, loss_train_matching))
print('Non matching, Test: {}, Train: {}'.format(loss_test_non_matching, loss_train_non_matching))

triplet = train_dataset.get_batch_triplets(1)
d1 = network_desc2.get_descriptor(triplet[0])
d2 = network_desc2.get_descriptor(triplet[1])
d3 = network_desc2.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.minimum(d_1_3, d_2_3)
# d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)
print('--------------------------------------------------------')


print('Data from testing dataset: ')
print('Matching test loss: {}'.format(network_desc2.test_model(test_batch_matching[0], test_batch_matching[1], True)))
print('Non matching test loss: {}'.format(network_desc2.test_model(test_batch_non_matching[0], test_batch_non_matching[1],False)))

triplet = test_dataset.get_batch_triplets(1)
d1 = network_desc2.get_descriptor(triplet[0])
d2 = network_desc2.get_descriptor(triplet[1])
d3 = network_desc2.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.minimum(d_1_3, d_2_3)
# d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)
