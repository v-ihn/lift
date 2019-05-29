try:
    import cv2.__init__ as cv2
except ImportError:
    pass
from tensorflow.python import keras as K
import numpy as np
import os


def mnist():
    (train_images, train_labels), (test_images, test_labels) = K.datasets.mnist.load_data()
    return train_images, train_labels, test_images, test_labels


def shapes(shapes_dir):
    labels, images = [], []
    shapes = ['square', 'circle', 'star', 'triangle']

    for shape in shapes:
        print('Getting data for: ', shape)
        for path in os.listdir(shapes_dir + '/' + shape):
            images.append(cv2.imread(shapes_dir + '/' + shape + '/' + path, 0))
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


def alienator(data_folder, train_filename, test_filename, rotated=True, kp_size_multiplier=15, clamp_crop_size=True):
    def create_patch(img, x, y, kp_angle, kp_size):
        x_center = int(x + 0.5)
        y_center = int(y + 0.5)

        if clamp_crop_size:
            crop_size = int(np.clip(kp_size_multiplier * kp_size, kp_size_multiplier * 2.0, kp_size_multiplier * 3.5))
        else:
            crop_size = int(kp_size_multiplier * kp_size)
        if x_center - crop_size < 0 or x_center + crop_size >= img.shape[1] or y_center - crop_size < 0 or y_center + crop_size >= img.shape[0]:
            return None

        if rotated:
            m = cv2.getRotationMatrix2D((x_center, y_center), 360 - kp_angle, 1)
            img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))

        img = img[y_center - crop_size:y_center + crop_size, x_center - crop_size:x_center + crop_size]
        # img = cv2.resize(img, (size, size))
        # img = cv2.normalize(img.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        return img

    train_images, test_images, train_labels, test_labels = [], [], [], []
    loaded_img_files = {}

    train_file = open(data_folder + '/' + train_filename, 'r')
    for line in train_file.read().splitlines():
        line_arr = line.split(',')

        label = line_arr[0] + ',' + line_arr[1] + ',' + line_arr[2]

        if line_arr[3] not in loaded_img_files:
            loaded_img_files[line_arr[3]] = cv2.imread(data_folder + '/' + line_arr[3], flags=cv2.IMREAD_GRAYSCALE)
        patch = create_patch(loaded_img_files[line_arr[3]].copy(), float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]))

        if patch is not None:
            train_labels.append(label)
            train_images.append(patch)
    train_file.close()

    test_file = open(data_folder + '/' + test_filename, 'r')
    for line in test_file.read().splitlines():
        line_arr = line.split(',')

        label = line_arr[0] + ',' + line_arr[1] + ',' + line_arr[2]

        if line_arr[3] not in loaded_img_files:
            loaded_img_files[line_arr[3]] = cv2.imread(data_folder + '/' + line_arr[3], flags=cv2.IMREAD_GRAYSCALE)
        patch = create_patch(loaded_img_files[line_arr[3]].copy(), float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]))

        if patch is not None:
            test_labels.append(label)
            test_images.append(patch)
    test_file.close()

    return train_images, train_labels, test_images, test_labels


def brown(data_folder):
    if os.path.isfile('{}/train_patches.npy'.format(data_folder)) and os.path.isfile('{}/test_patches.npy'.format(data_folder)) and os.path.isfile('{}/train_labels.npy'.format(data_folder)) and os.path.isfile('{}/test_labels.npy'.format(data_folder)):
        train_images = np.load('{}/train_patches.npy'.format(data_folder))
        test_images = np.load('{}/test_patches.npy'.format(data_folder))
        train_labels = np.load('{}/train_labels.npy'.format(data_folder))
        test_labels = np.load('{}/test_labels.npy'.format(data_folder))
    else:
        all_labels = []
        train_images, test_images, train_labels, test_labels = [], [], [], []

        patches_in_row = 16
        patches_in_column = 16
        patch_size = 64

        data_file = open(data_folder + '/info.txt', 'r')
        for line in data_file.readlines():
            all_labels.append(line.split(' ')[0])
        data_file.close()

        unique_labels = list(set(all_labels))
        # labels_to_test = np.random.choice(unique_labels, int(len(unique_labels) / 20), replace=False)
        labels_to_test = [unique_labels[i] for i in range(len(unique_labels)) if i % 10 == 0]

        img_index = 0
        label_index = 0
        while os.path.isfile('{}/patches{:04d}.bmp'.format(data_folder, img_index)):
            img = cv2.imread('{}/patches{:04d}.bmp'.format(data_folder, img_index), flags=cv2.IMREAD_GRAYSCALE)

            for img_patch_index in range(patches_in_row * patches_in_column):
                if label_index >= len(all_labels):
                    break

                label = all_labels[label_index]

                patch_x = (img_patch_index % patches_in_row) * patch_size
                patch_y = int(img_patch_index / patches_in_column) * patch_size
                patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

                if label in labels_to_test:
                    test_labels.append(label)
                    test_images.append(patch)
                else:
                    train_labels.append(label)
                    train_images.append(patch)

                label_index += 1

            img_index += 1

        np.save('{}/train_patches.npy'.format(data_folder), train_images)
        np.save('{}/test_patches.npy'.format(data_folder), test_images)
        np.save('{}/train_labels.npy'.format(data_folder), train_labels)
        np.save('{}/test_labels.npy'.format(data_folder), test_labels)

    return train_images, train_labels, test_images, test_labels



