try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import sys

from Dataset import *
from DatasetMnist import *
from NetworkDesc import *
from DatasetShapes import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = K.datasets.mnist.load_data()
dataset = DatasetMnist(x_train_mnist, y_train_mnist)
test_dataset = DatasetMnist(x_test_mnist, y_test_mnist)

# x_train_shapes, y_train_shapes, x_test_shapes, y_test_shapes = get_shapes_train_test('img/shapes/')
# dataset = DatasetMnist(x_train_shapes, y_train_shapes)
# test_dataset = DatasetMnist(x_test_shapes, y_test_shapes)

# folder = 'img'
# kp_train_file = 'train_keypoints.txt'
# kp_test_file = 'test_keypoints.txt'
#
# dataset = Dataset(folder, kp_train_file)
# # dataset.create_fixed_epoch_desc(38400,  rotated=True)
#
# test_dataset = Dataset(folder, kp_test_file, mean=dataset.mean, std=dataset.std)
test_batch = test_dataset.get_batch_desc_triplets(64)

network_desc = NetworkDesc(learning_rate=0.003)
mining_ratio = 1

for i in range(1, 15001):
    patches = dataset.get_batch_desc_triplets(128)
    # patches = dataset.get_batch_desc_triplets_from_fixed_epoch(128)
    train_loss = network_desc.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
    if i % 10 == 0:
        print('Iteration {}, Train loss {}'.format(i, train_loss))
    if i % 1000 == 0:
        network_desc.save_weights()
        print('Test loss: {}'.format(network_desc.test_model(test_batch[0], test_batch[1], test_batch[2])))
    if i % 5000 == 0 and mining_ratio < 4:
        mining_ratio *= 2


patches = dataset.get_batch_desc_triplets(256)
loss_test = network_desc.test_model(patches[0], patches[1], patches[2])
loss_train = network_desc.train_model(patches[0], patches[1], patches[2])
print('Test: {}, Train: {}'.format(loss_test, loss_train))

triplet = dataset.get_batch_desc_triplets(1)
d1 = network_desc.get_descriptor(triplet[0])
d2 = network_desc.get_descriptor(triplet[1])
d3 = network_desc.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.maximum(d_1_3, d_2_3)
d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)


print('Test loss: {}'.format(network_desc.test_model(test_batch[0], test_batch[1], test_batch[2])))

triplet = test_dataset.get_batch_desc_triplets(1)
d1 = network_desc.get_descriptor(triplet[0])
d2 = network_desc.get_descriptor(triplet[1])
d3 = network_desc.get_descriptor(triplet[2])

d_pos = np.sqrt(np.sum(np.square(d1 - d2), axis=1, keepdims=True))
d_1_3 = np.sqrt(np.sum(np.square(d1 - d3), axis=1, keepdims=True))
d_2_3 = np.sqrt(np.sum(np.square(d2 - d3), axis=1, keepdims=True))
d_neg = np.minimum(d_1_3, d_2_3)
d_neg = 4.0 - d_neg

print(d_pos)
print(d_neg)
