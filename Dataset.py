import cv2
import numpy as np
import random


class Dataset:
    def __init__(self, images, labels, size=64, rand_rot_range=None, mean=None, std=None):
        self.available_patches = {}
        resized_images = np.zeros((len(images), size, size), np.float32)
        for i in range(len(images)):
            resized_images[i] = cv2.resize(images[i], (size, size))

        self.mean = np.mean(resized_images) if mean is None else mean
        self.std = np.std(resized_images) if std is None else std

        for i in range(len(labels)):
            if labels[i] not in self.available_patches:
                self.available_patches[labels[i]] = []

            patch = resized_images[i]
            if rand_rot_range is not None:
                patch = self.rotate_patch(patch, random.randint(rand_rot_range[0], rand_rot_range[1]))
            # patch = (patch - self.mean) / self.std
            patch = np.expand_dims(patch, axis=2)

            self.available_patches[labels[i]].append(patch)

        self.unique_labels = list(set(labels))

    def get_pair(self, matching):
        if matching:
            label = random.choice(self.unique_labels)
            patches = random.sample(self.available_patches[label], 2)
        else:
            labels = random.sample(self.unique_labels, 2)
            patches = [random.choice(self.available_patches[labels[0]]), random.choice(self.available_patches[labels[1]])]

        return patches[0], patches[1]

    def get_triplet(self):
        labels = random.sample(self.unique_labels, 2)
        matching_patches = random.sample(self.available_patches[labels[0]], 2)
        different_patch = random.choice(self.available_patches[labels[1]])

        return matching_patches[0], matching_patches[1], different_patch

    def get_batch_pairs(self, batch_size: int, matching: bool):
        patches = []
        for i in range(batch_size):
            patches_pair = self.get_pair(matching)
            patches.append(patches_pair)
        patches_batch = np.array(list(zip(*patches)))
        return patches_batch

    def get_batch_triplets(self, batch_size: int):
        patches = []
        for i in range(batch_size):
            patches_triplet = self.get_triplet()
            patches.append(patches_triplet)
        patches_batch = np.array(list(zip(*patches)))
        return patches_batch

    def rotate_patch(self, patch, angle):
        x_center = patch.shape[1] / 2
        y_center = patch.shape[0] / 2
        m = cv2.getRotationMatrix2D((x_center, y_center), 360 - angle, 1)
        rotated_patch = cv2.warpAffine(patch, m, (patch.shape[1], patch.shape[0]))
        return np.expand_dims(rotated_patch, axis=2)

