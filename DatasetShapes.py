from tensorflow.python import keras as K
import cv2
import numpy as np
import random
import os


def get_shapes_train_test(shapes_dir):
    labels, images = [], []
    shapes = ['square', 'circle', 'star', 'triangle']

    for shape in shapes:
        print('Getting data for: ', shape)
        for path in os.listdir(shapes_dir + shape):
            images.append(cv2.imread(shapes_dir + shape + '/' + path, 0))
            labels.append(shapes.index(shape))

    train_test_ratio, to_train = 5, 0
    train_images, test_images, train_labels, test_labels = [], [], [], []
    for image, label in zip(images, labels):
        if to_train < train_test_ratio:
            train_images.append(image)
            train_labels.append(label)
            to_train += 1
        else:
            test_images.append(image)
            test_labels.append(label)
            to_train = 0

    return train_images, train_labels, test_images, test_labels
