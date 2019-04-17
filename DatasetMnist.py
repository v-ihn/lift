from tensorflow.python import keras as K
import cv2
import numpy as np
import random


class DatasetMnist:
    def __init__(self, x, y):
        self.available_patches = {}
        for i in range(len(y)):
            if y[i] not in self.available_patches:
                self.available_patches[y[i]] = []
            patch = cv2.resize(x[i], (64, 64))
            patch = cv2.normalize(patch.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
            patch = np.expand_dims(patch, axis=2)
            self.available_patches[y[i]].append(patch)

        self.available_indices = {}
        self.init_patch_indices()

    def init_patch_indices(self):
        self.available_indices = {}
        for label in self.available_patches:
            num_imgs = len(self.available_patches[label])
            patch_indices = [i for i in range(num_imgs)]
            self.available_indices[label] = patch_indices

    def get_triplet(self):
        labels = random.sample(list(self.available_patches), 2)
        matching_patches = random.sample(self.available_patches[labels[0]], 2)
        different_patch = random.choice(self.available_patches[labels[1]])

        return matching_patches[0], matching_patches[1], different_patch

    def get_triplet2(self):
        labels = random.sample(list(self.available_indices), 2)
        matching_patches_indices = random.sample(self.available_indices[labels[0]], 2)
        different_patch_index = random.choice(self.available_indices[labels[1]])

        self.available_indices[labels[0]] = [index for index in self.available_indices[labels[0]] if index not in matching_patches_indices]
        if len(self.available_indices[labels[0]]) < 2:
            del self.available_indices[labels[0]]

        self.available_indices[labels[1]].remove(different_patch_index)
        if len(self.available_indices[labels[1]]) < 2:
            del self.available_indices[labels[1]]

        if len(self.available_indices) < 2:
            self.init_patch_indices()

        matching_patches = [self.available_patches[labels[0]][index] for index in matching_patches_indices]
        different_patch = self.available_patches[labels[1]][different_patch_index]

        return matching_patches[0], matching_patches[1], different_patch

    def get_batch_desc_triplets(self, batch_size: int):
        patches = []
        for i in range(batch_size):
            patches_triplet = self.get_triplet2()
            patches.append(patches_triplet)
        patches_batch = np.array(list(zip(*patches)))
        return patches_batch

