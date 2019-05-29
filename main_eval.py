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
from NetworkDescPN import *


def get_net_descriptors(net, img_batch):
    step = 1000
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

    tpr_indices = np.argsort(tpr)
    fpr = np.array(fpr)[tpr_indices]
    tpr = np.array(tpr)[tpr_indices]

    return tpr, fpr, precision, recall


def save_net_results(model_name, dataset_name):
    if dataset_name == 'mnist':
        train_images, train_labels, test_images, test_labels = datasets.mnist()
    elif dataset_name == 'shapes':
        train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
    elif dataset_name == 'alienator':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=False)
    elif dataset_name == 'alienator2':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints2.txt', 'test_keypoints2.txt', rotated=False)
    else:
        train_images, train_labels, test_images, test_labels = datasets.brown(dataset_name)

    train_dataset = Dataset(train_images, train_labels)
    test_dataset = Dataset(test_images, test_labels, mean=train_dataset.mean, std=train_dataset.std)

    network_desc = NetworkDesc(model_file=model_name + '.h5')
    # network_desc = NetworkDescPN(model_file=model_name + '.h5')

    batch = test_dataset.get_batch_triplets(1000)

    positives_net, negatives_net = get_positives_negatives(get_net_descriptors(network_desc, batch[0]), get_net_descriptors(network_desc, batch[1]),
                                                           get_net_descriptors(network_desc, batch[2]))

    results_dir = 'results/{}/'.format(model_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    np.save('{}{}.positives'.format(results_dir, dataset_name), positives_net)
    np.save('{}{}.negatives'.format(results_dir, dataset_name), negatives_net)


def save_sift_results(dataset_name):
    if dataset_name == 'mnist':
        train_images, train_labels, test_images, test_labels = datasets.mnist()
    elif dataset_name == 'shapes':
        train_images, train_labels, test_images, test_labels = datasets.shapes('img/shapes')
    elif dataset_name == 'alienator':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints.txt', 'test_keypoints.txt', rotated=False)
    elif dataset_name == 'alienator2':
        train_images, train_labels, test_images, test_labels = datasets.alienator('img', 'train_keypoints2.txt', 'test_keypoints2.txt', rotated=False)
    else:
        train_images, train_labels, test_images, test_labels = datasets.brown(dataset_name)

    train_dataset = Dataset(train_images, train_labels)
    test_dataset = Dataset(test_images, test_labels, mean=train_dataset.mean, std=train_dataset.std)

    batch = test_dataset.get_batch_triplets(1000)

    positives_sift, negatives_sift = get_positives_negatives(get_sift_descriptors(batch[0]), get_sift_descriptors(batch[1]), get_sift_descriptors(batch[2]))

    results_dir = 'results/sift/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    np.save('{}{}.positives'.format(results_dir, dataset_name), positives_sift)
    np.save('{}{}.negatives'.format(results_dir, dataset_name), negatives_sift)


def load_positives_negatives_from_files(directory, dataset_name):
    positives = np.load('{}{}.positives.npy'.format(directory, dataset_name))
    negatives = np.load('{}{}.negatives.npy'.format(directory, dataset_name))

    # plt.hist(positives, bins=250, alpha=0.75)
    # plt.hist(negatives, bins=250, alpha=0.75)
    # plt.xlabel('Distance')
    # plt.ylabel('Count')
    # plt.show()

    return positives, negatives


def plot_saved_data(directories, dataset_names, labels, type):
    assert len(directories) == len(dataset_names) == len(labels)

    for i in range(len(directories)):
        positives, negatives = load_positives_negatives_from_files(directories[i], dataset_names[i])
        tpr, fpr, precision, recall = get_tpr_fpr_precision_recall(positives, negatives, 500)

        auc_roc = np.trapz(tpr, fpr)
        auc_pr = np.trapz(precision, recall)
        fpr_95 = np.interp(0.95, tpr, fpr)
        print('{} - FPR {} @ 95% TPR, AUC ROC: {}, AUC PR: {}'.format(labels[i], fpr_95, auc_roc, auc_pr))

        if type == 'ROC':
            plt.plot(fpr, tpr, linewidth=1, label=labels[i])
        elif type == 'PR':
            plt.plot(precision, recall, linewidth=1, label=labels[i])

    if type == 'ROC':
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # plt.xlim(-0.01, 0.5)
        # plt.ylim(0.5, 1.01)
        plt.legend(loc='lower right')
    elif type == 'PR':
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.xlim(0.5, 1.01)
        plt.ylim(0.5, 1.01)
        plt.legend(loc='lower left')

    plt.show()


if __name__ == "__main__":
    # save_net_results('desc_ali2_hinge_relu_avg_batch_adam', 'alienator2')
    # save_sift_results('alienator2')

    directories = [
                   'results/desc_liberty_hinge_relu_avg_batch_adam/',
                   # 'results/desc_liberty_hinge_relu_avg_batch_sgd/',
                   # 'results/desc_liberty_hinge_tanh_l2_batch_adam/',
                   # 'results/desc_liberty_softpn_hm_relu_avg_batch_adam/',
                   # 'results/desc_liberty_softpn_hm_tanh_l2_batch_adam/',
                   # 'results/desc_liberty_softpn_relu_avg_batch_adam/',
                   # 'results/desc_liberty_softpn_relu_avg_batch_sgd/',
                   # 'results/desc_liberty_softpn_tanh_l2_batch_adam/',
                   'results/sift/',
                   'results/desc_ali_hinge_relu_avg_batch_adam/',
                   # 'results/desc_ali_softpn_relu_avg_batch_adam/',
                   'results/sift/',
                   'results/desc_ali2_hinge_relu_avg_batch_adam/',
                   'results/sift/'
    ]
    dataset_names = [
                     'notredame',
                     # 'notredame',
                     # 'notredame',
                     # 'notredame',
                     # 'notredame',
                     # 'notredame',
                     # 'notredame',
                     # 'notredame',
                     'notredame',
                     'alienator',
                     # 'alienator',
                     'alienator',
                     'alienator2',
                     'alienator2'
    ]
    labels = [
        'Network - Notre Dame',
        # 'Hinge new SGD',
        # 'Hinge - old',
        # 'SoftPN - new',
        # 'SoftPN - old',
        # 'SoftPN new Adam',
        # 'SoftPN new SGD',
        # 'SoftPN old Adam',
        'SIFT - Notre Dame',
        'Network - Alienator',
        # 'Alienator SoftPN',
        'SIFT - Alienator',
        'Network - Alienator reduced',
        'SIFT - Alienator reduced'
    ]

    plot_saved_data(directories, dataset_names, labels, 'ROC')
