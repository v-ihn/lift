try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import sys
import tensorflow as tf

import datasets
from Dataset import *
from NetworkDesc import *
from NetworkDescPN import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_datasets(dataset_name):
    if dataset_name == 'mnist':
        train_images, train_labels, test_images, test_labels = datasets.mnist()
    elif dataset_name == 'shapes':
        train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
    elif dataset_name == 'alienator':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=False)
    elif dataset_name == 'alienator_custom':
        train_images, train_labels, test_images, test_labels = datasets.alienator('.', 'train_keypoints_custom.txt', 'test_keypoints_custom.txt', rotated=False, kp_size_multiplier=30)
    elif dataset_name == 'alienator_custom_ns':
        train_images, train_labels, test_images, test_labels = datasets.alienator('.', 'train_keypoints_custom_ns.txt', 'test_keypoints_custom_ns.txt', rotated=False, kp_size_multiplier=30)
    elif dataset_name == 'alienator2':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints2.txt', 'test_keypoints2.txt', rotated=False)
    else:
        train_images, train_labels, test_images, test_labels = datasets.brown(dataset_name)

    train = Dataset(train_images, train_labels, size=64)
    test = Dataset(test_images, test_labels, mean=train.mean, std=train.std, size=64)

    return train, test


if __name__ == '__main__':
    train_dataset, test_dataset = load_datasets('alienator_custom_ns')

    test_batch = test_dataset.get_batch_triplets(100)

    desc_file = 'desc_ali_custom_ns2_hinge_relu_avg_batch_adam.h5'
    network_desc = NetworkDesc(learning_rate=0.001, model_file=desc_file)
    mining_ratio = 1
    batch_size = 100

    for i in range(1, 6001):
        patches = train_dataset.get_batch_triplets(batch_size)

        train_loss = network_desc.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
        if i % 10 == 0:
            print('Iteration {}, Train loss {}'.format(i, train_loss))
        if i % 1000 == 0:
            network_desc.save_weights()
            print('Test loss: {}'.format(network_desc.test_model(test_batch[0], test_batch[1], test_batch[2])))
        if i % 2000 == 0 and mining_ratio < 4:
            mining_ratio *= 2
            batch_size *= 2

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
