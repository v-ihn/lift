try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import sys
from tensorflow.python import keras as K
import matplotlib.pyplot as plt

import datasets
from Dataset import *
from NetworkDesc import *
from NetworkDesc2 import *


def get_net_descriptors(net, img_batch):
    step = 5000
    assert img_batch.shape[0] % step == 0
    all_descriptors = []

    for i in range(0, img_batch.shape[0], step):
        descriptors = net.get_descriptor(img_batch[i:i+step])
        print(descriptors.shape)
        all_descriptors.append(descriptors)

    all_descriptors = np.array(all_descriptors)
    return all_descriptors.reshape(-1, all_descriptors.shape[2])


def get_sift_descriptors(img_batch):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = cv2.KeyPoint(31.5, 31.5, 4, 0)
    sift_descriptors = []

    for image in img_batch:
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, descriptor = sift.compute(img, [kp])
        sift_descriptors.append(descriptor[0])

    return np.array(sift_descriptors)


def get_positives_negatives(desc_p1, desc_p2, desc_p3):
    positives = np.sqrt(np.sum(np.square(desc_p1 - desc_p2), axis=1, keepdims=True))
    d_1_3 = np.sqrt(np.sum(np.square(desc_p1 - desc_p3), axis=1, keepdims=True))
    d_2_3 = np.sqrt(np.sum(np.square(desc_p2 - desc_p3), axis=1, keepdims=True))
    negatives = np.minimum(d_1_3, d_2_3)

    return positives, negatives


def get_tpr_fpr_precision_recall(positives, negatives, thresh_steps):
    positives = np.sort(positives.flatten())
    negatives = np.sort(negatives.flatten())

    min_positive = positives[0]
    max_negative = negatives[-1]
    thresh_range = np.linspace(min_positive, max_negative, thresh_steps)

    tpr = []
    fpr = []
    precision = []
    recall = []
    for threshold in thresh_range:
        # print('Thresh {}'.format(threshold))
        true_positives = np.count_nonzero(positives < threshold)
        false_positives = np.count_nonzero(negatives <= threshold)
        true_negatives = np.count_nonzero(negatives > threshold)
        false_negatives = np.count_nonzero(positives >= threshold)
        # print('FP {}, FN {}'.format(false_positives, false_negatives))

        tpr.append(true_positives / (true_positives + false_negatives))
        fpr.append(false_positives / (false_positives + true_negatives))
        if (true_positives + false_positives) != 0:
            precision.append(true_positives / (true_positives + false_positives))
            recall.append(true_positives / (true_positives + false_negatives))

    return tpr, fpr, precision, recall


if __name__ == "__main__":
    # (train_images, train_labels), (test_images, test_labels) = datasets.mnist()
    # train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
    # train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=True)
    train_images, train_labels, test_images, test_labels = datasets.brown('trevi')

    train_dataset = Dataset(train_images, train_labels)
    test_dataset = Dataset(test_images, test_labels, mean=train_dataset.mean, std=train_dataset.std)

    network_desc = NetworkDesc(learning_rate=0.01, model_file='desc_trevi_hinge_relu_avg_batch_adam.h5')
    # network_desc = NetworkDesc2(learning_rate=0.01)

    batch = test_dataset.get_batch_triplets(50000)

    positives_net, negatives_net = get_positives_negatives(get_net_descriptors(network_desc, batch[0]), get_net_descriptors(network_desc, batch[1]), get_net_descriptors(network_desc, batch[2]))
    positives_sift, negatives_sift = get_positives_negatives(get_sift_descriptors(batch[0]), get_sift_descriptors(batch[1]), get_sift_descriptors(batch[2]))

    tpr_net, fpr_net, precision_net, recall_net = get_tpr_fpr_precision_recall(positives_net, negatives_net, 5000)
    tpr_sift, fpr_sift, precision_sift, recall_sift = get_tpr_fpr_precision_recall(positives_sift, negatives_sift, 5000)

    auc_roc_net = np.trapz(tpr_net, fpr_net)
    auc_pr_net = np.trapz(precision_net, recall_net)
    auc_roc_sift = np.trapz(tpr_sift, fpr_sift)
    auc_pr_sift = np.trapz(precision_sift, recall_sift)
    print('NET AUC ROC: {}, AUC PR: {}'.format(auc_roc_net, auc_pr_net))
    print('SIFT AUC ROC: {}, AUC PR: {}'.format(auc_roc_sift, auc_pr_sift))

    plt.ylim((0.9, 1.01))
    plt.xlim((-0.01, 0.1))
    plt.plot(fpr_net, tpr_net, 'b')
    # plt.plot(recall_net, precision_net, 'b--')
    plt.plot(fpr_sift, tpr_sift, 'k')
    # plt.plot(recall_sift, precision_sift, 'k--')
    plt.show()
